import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from MGNN_class import MLP, MG_NN
from datetime import datetime
import os
import pickle
from minmax_opt_class_mgnn_2d import MinMaxOptProb2D
from my_helpers import plot_u_theta, plot_loss_history
import matplotlib.pyplot as plt
import wandb
import matplotlib.gridspec as gridspec




class SeqMinMaxOptProb:
    def __init__(self, input_dim=2, num_segments=1, total_t_min=0, total_t_max=1, x_min=0, x_max=1, y_min=None, y_max=None, seq_T=None, N_t=10, N_x=10, N_y=None, 
                 N_bc=128, N_ic=256, hidden_dims_list=[(64, 64, 64, 64)], seed=0, 
                 sigma=5, N_perturb=10000, k_h_rf_times=2, f_adaptive_weight=1000, f_opt_kh=1, u0=None, u1=None, u2=None, project="seq_minmax_burgers",
                 lambda_ic=20.0, lambda_bc=20.0, lambda_eqn=1.0, alpha=0.9, gamma=0.0, N_test_gamma=1000, adjust_gamma=True,
                 L1reg=0, sampling_method='uniform', notes='', pre_epochs=1000, nu=0.01/np.pi,
                 init_lrs=[1e-3, 1e-3, 1e-3], grade_epochs=[10, 10, 10], decay_rates=[1e-3, 1e-3, 1e-3], init_lr_stage2=1e-3, epoch_stage2=100, decay_rate_stage2=1e-3,
                 res=None, loss_type='interior',  k=0, use_a=True, f_plot=1000, plt_err_dist=False, plt_mesh=False,
                 partition='tri',
                 M_t=1000, M_x=2000, set_wandb=False):
        """
        Initialize the sequential training class.

        Args:
            num_segments (int): Number of time segments.
            total_t_max (float): Total time horizon (e.g., 2 for [0,1] and [1,2]).
            u0 (callable): Initial condition function for the first segment.
            u1 (callable): Boundary condition at x_min.
            u2 (callable): Boundary condition at x_max.
            ... (other parameters passed to MinMaxOptProb2D)
        """
        self.input_dim = input_dim
        self.set_wandb = set_wandb
        self.partition = partition
        
        self.total_t_min = total_t_min
        self.total_t_max = total_t_max
        self.seq_T = np.array(seq_T) if seq_T is not None else np.linspace(total_t_min, total_t_max, num_segments + 1)
        self.num_segments = len(self.seq_T) - 1

        self.u0 = u0
        self.u1 = u1
        self.u2 = u2
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.segments = []  # List to store trained segment instances
        # Store other parameters for segment creation
        self.N_t = N_t
        self.N_x = N_x
        self.N_y = N_y
        
        self.hidden_dims_list = hidden_dims_list
        self.seed = seed
        self.sigma = sigma
        self.N_perturb = N_perturb
        self.N_bc = N_bc
        self.N_ic = N_ic
        self.k_h_rf_times = k_h_rf_times
        self.lambda_ic = lambda_ic
        self.lambda_bc = lambda_bc
        self.lambda_eqn = lambda_eqn
        self.gamma = gamma
        self.alpha = alpha
        self.N_test_gamma = N_test_gamma
        self.adjust_gamma = adjust_gamma
        self.L1reg = L1reg
        self.sampling_method = sampling_method
        self.notes = notes
        self.pre_epochs = pre_epochs
        self.nu = nu
        self.init_lrs = init_lrs
        self.grade_epochs = grade_epochs
        self.decay_rates = decay_rates
        self.init_lr_stage2 = init_lr_stage2
        self.epoch_stage2 = epoch_stage2
        self.decay_rate_stage2 = decay_rate_stage2
        self.f_adaptive_weight = f_adaptive_weight
        self.f_opt_kh = f_opt_kh
        self.res = res

        self.loss_type = loss_type
        self.k = k
        self.use_a = use_a
        self.M_t = M_t
        self.M_x = M_x
        self.f_plot = f_plot
        self.plt_err_dist = plt_err_dist
        self.plt_mesh = plt_mesh
        self.run_name = f"({total_t_max:.1f}-{total_t_min:.1f})/{num_segments}_Nt{N_t}_Nx{N_x}_lrs{init_lrs[0]}_eps{grade_epochs[0]}_L1reg{L1reg}_sig{sigma}_gam{gamma}_Np{N_perturb}_sd{seed}"
        self.project = project
        if self.set_wandb:
            wandb.init(project=self.project, 
                       name=self.run_name,
                       config={"num_segments": self.num_segments, 
                               "total_t_min": self.total_t_min,
                               "total_t_max": self.total_t_max,
                               "seq_T": self.seq_T.tolist(),
                               "N_t": N_t,
                               "N_x": N_x,
                               "N_bc": N_bc,
                               "N_ic": N_ic,
                               "hidden_dims_list": hidden_dims_list,
                               "seed": seed,
                               "sigma": sigma,
                               "N_perturb": N_perturb,
                               "k_h_rf_times": k_h_rf_times,
                               "lambda_ic": lambda_ic,
                               "lambda_bc": lambda_bc,
                               "lambda_eqn": lambda_eqn,
                               "gamma": gamma,
                               "alpha": alpha,
                               "N_test_gamma": N_test_gamma,
                               "adjust_gamma": adjust_gamma,
                               "L1reg": L1reg,
                               "sampling_method": sampling_method,
                              })

    def train_seq(self, grade_epochs=None, epoch_stage2=None, save_model=True):
        """
        Train all segments sequentially, copying the previous segment's parameters to the next.

        Args:
            save_model (bool): Whether to save model parameters.
        """
        if grade_epochs is None:
            grade_epochs = self.grade_epochs
        if epoch_stage2 is None:
            epoch_stage2 = self.epoch_stage2
        for i in range(self.num_segments):
            # Define time interval for the current segment
            t_min = self.seq_T[i]
            t_max = self.seq_T[i+1]

            # Set initial condition: use u0 for first segment, else use previous segment's final state
            if i == 0:
                current_u0 = self.u0
            else:
                prev_segment = self.segments[i - 1]
                current_u0 = lambda x: prev_segment.model.forward(
                    jnp.stack([jnp.ones_like(x) * prev_segment.pwp.t_max, x], axis=1)
                )

            # Create a new MinMaxOptProb2D instance for this segment
            segment = MinMaxOptProb2D(
                input_dim=self.input_dim,
                t_min=t_min, t_max=t_max, x_min=self.x_min, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max, N_t=self.N_t, N_x=self.N_x, N_y=self.N_y
                N_bc=self.N_bc, N_ic=self.N_ic, hidden_dims_list=self.hidden_dims_list, seed=self.seed, 
                sigma=self.sigma, N_perturb=self.N_perturb, k_h_rf_times=self.k_h_rf_times, u0=current_u0, u1=self.u1, u2=self.u2, project=self.project,
                gamma=self.gamma, alpha=self.alpha, N_test_gamma=self.N_test_gamma, adjust_gamma=self.adjust_gamma, 
                lambda_ic=self.lambda_ic, lambda_bc=self.lambda_bc, lambda_eqn=self.lambda_eqn,
                L1reg=self.L1reg, sampling_method=self.sampling_method, notes=self.notes, pre_epochs=self.pre_epochs, nu=self.nu,
                init_lrs=self.init_lrs, grade_epochs=grade_epochs, decay_rates=self.decay_rates, 
                init_lr_stage2=self.init_lr_stage2, epoch_stage2=epoch_stage2, decay_rate_stage2=self.decay_rate_stage2,
                f_adaptive_weight=self.f_adaptive_weight, f_opt_kh=self.f_opt_kh, res=self.res, 
                loss_type=self.loss_type, k=self.k, use_a=self.use_a, 
                f_plot=self.f_plot, plt_err_dist=self.plt_err_dist, plt_mesh=self.plt_mesh, partition=self.partition, M_t=self.M_t, M_x=self.M_x,
                seq_train=True, set_wandb=self.set_wandb,
            )
             # Store the segment and update previous_params with the trained parameters
            self.segments.append(segment)

            # Plot before training
            print(f"\nPlotting before training segment [{t_min:.4f}, {t_max:.4f}]")
            plot_u_theta(
                prob=segment,
                M_t=self.M_t,
                M_x=self.M_x,
                t_min=t_min,
                t_max=t_max,
                x_min=self.x_min,
                x_max=self.x_max,
                upload_fig=False
            )

            # Train the segment
            print(f"\nTraining segment [{t_min:.4f}, {t_max:.4f}]")
            segment.train_process_stage1()
            segment.train_process_stage2()

            # Plot after training
            print(f"\nPlotting after training segment [{t_min:.4f}, {t_max:.4f}]")
            plot_u_theta(
                prob=segment,
                M_t=self.M_t,
                M_x=self.M_x,
                t_min=t_min,
                t_max=t_max,
                x_min=self.x_min,
                x_max=self.x_max,
                upload_fig=self.set_wandb
            )

        # After training all segments, plot the combined output
        print("\nPlotting combined output across all segments from t=0 to total_t_max")
        plot_u_theta(self, M_t_per_segment=self.M_t, M_x=self.M_x, upload_fig=self.set_wandb)
        wandb.finish()


def save_seq_model(seq_obj, model_path_prefix):
    """
    Save the model parameters for all segments.

    Args:
        seq_obj: Instance of SeqMinMaxOptProb.
        model_path_prefix (str): Prefix for the model file paths.
    """
    os.makedirs(os.path.dirname(model_path_prefix), exist_ok=True)
    save_dict = {
        'seq_T': seq_obj.seq_T,
        'num_segments': seq_obj.num_segments,
        'segments_params': [segment.save_model_dict() for segment in seq_obj.segments],
        'total_t_min': seq_obj.total_t_min,
        'total_t_max': seq_obj.total_t_max,
        'x_min': seq_obj.x_min,
        'x_max': seq_obj.x_max,
        'config': {
            'u0': seq_obj.u0,
            'u1': seq_obj.u1,
            'u2': seq_obj.u2,
            'res': seq_obj.res,
            'N_t': seq_obj.N_t,
            'N_x': seq_obj.N_x,
            'hidden_dims_list': seq_obj.hidden_dims_list,
            'seed': seq_obj.seed,
            'sigma': seq_obj.sigma,
            'N_perturb': seq_obj.N_perturb,
            'N_bc': seq_obj.N_bc,
            'N_ic': seq_obj.N_ic,
            'k_h_rf_times': seq_obj.k_h_rf_times,
            'lambda_ic': seq_obj.lambda_ic,
            'lambda_bc': seq_obj.lambda_bc,
            'lambda_eqn': seq_obj.lambda_eqn,
            'gamma': seq_obj.gamma,
            'alpha': seq_obj.alpha,
            'N_test_gamma': seq_obj.N_test_gamma,
            'adjust_gamma': seq_obj.adjust_gamma,
            'L1reg': seq_obj.L1reg,
            'sampling_method': seq_obj.sampling_method,
            'set_wandb': seq_obj.set_wandb
        }
    }
    model_path = f"{model_path_prefix}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"Saved entire sequence model to {model_path}")

def load_seq_model(model_path):
    """
    Load the model parameters for all segments into the seq_obj.

    Args:
        model_path (str): Path to the saved sequence model file.
    """
    with open(model_path, 'rb') as f:
        loaded_data = pickle.load(f)
    config = loaded_data['config']
    seq_obj = SeqMinMaxOptProb(num_segments=loaded_data['num_segments'], total_t_min=loaded_data['total_t_min'], total_t_max=loaded_data['total_t_max'], 
                               x_min=loaded_data['x_min'], x_max=loaded_data['x_max'], **config)
    seq_obj.seq_T = loaded_data['seq_T']
    seq_obj.num_segments = loaded_data['num_segments']
    
    seq_obj.segments = []
    for segment_params in loaded_data['segments_params']:
        segment = MinMaxOptProb2D(
            **config
        )
        segment.load_model_dict(segment_params)
        seq_obj.segments.append(segment)
    print(f"Loaded entire sequence model from {model_path}")
    return seq_obj