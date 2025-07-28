import jax
import jax.numpy as jnp
import numpy as np
import optax
from datetime import datetime
import os
import pickle
from minmax_opt_class_mgnn import MinMaxOptProb
from my_helpers import plot_u_theta, plot_loss_history
import matplotlib.pyplot as plt
import wandb
import matplotlib.gridspec as gridspec




class SeqMinMaxOptProb:
    def __init__(self, num_segments, total_t_min, total_t_max, x_min=0, x_max=1, seq_T=None, N_t=10, N_x=10, N_bc=2048, N_ic=4096, hidden_dims_list=[(64, 64, 64, 64)], seed=0, 
                 sigma=0.1, N_perturb=10000, k_h_rf_times=2, f_adaptive_weight=1000, f_opt_kh=1, u0=None, u1=None, u2=None, project="seq_minmax_burgers",
                 lambda_ic=10.0, lambda_bc=10.0, lambda_eqn=1.0, alpha=0.9, gamma=0.0, N_test_gamma=1000, adjust_gamma=True,
                 L1reg=0, sampling_method='normal', notes='', lambda_tv=1.0, 
                 init_lrs=[1e-3, 1e-3, 1e-3], grade_epochs=[10, 10, 10], decay_rates=[1e-3, 1e-3, 1e-3], init_lr_stage2=1e-3, epoch_stage2=100, decay_rate_stage2=1e-3,
                 res=None, res_visc=None, use_loss_fn_elem_int=True, total_grade=1, k=0, use_a=True,
                 M_t=1000, M_x=2000, set_wandb=True):
        """
        Initialize the sequential training class.

        Args:
            num_segments (int): Number of time segments.
            total_t_max (float): Total time horizon (e.g., 2 for [0,1] and [1,2]).
            u0 (callable): Initial condition function for the first segment.
            u1 (callable): Boundary condition at x_min.
            u2 (callable): Boundary condition at x_max.
            ... (other parameters passed to MinMaxOptProb)
        """
        self.set_wandb = set_wandb
        
        self.total_t_min = total_t_min
        self.total_t_max = total_t_max
        self.seq_T = np.array(seq_T) if seq_T is not None else np.linspace(total_t_min, total_t_max, num_segments + 1)
        self.num_segments = len(self.seq_T) - 1

        self.u0 = u0
        self.u1 = u1
        self.u2 = u2
        self.x_min = x_min
        self.x_max = x_max
        self.segments = []  # List to store trained segment instances
        # Store other parameters for segment creation
        self.N_t = N_t
        self.N_x = N_x
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
        self.lambda_tv = lambda_tv
        self.init_lrs = init_lrs
        self.grade_epochs = grade_epochs
        self.decay_rates = decay_rates
        self.init_lr_stage2 = init_lr_stage2
        self.epoch_stage2 = epoch_stage2
        self.decay_rate_stage2 = decay_rate_stage2
        self.f_adaptive_weight = f_adaptive_weight
        self.f_opt_kh = f_opt_kh
        self.res = res
        self.res_visc = res_visc
        self.use_loss_fn_elem_int = use_loss_fn_elem_int
        self.total_grade = total_grade
        self.k = k
        self.use_a = use_a
        self.M_t = M_t
        self.M_x = M_x
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
                               "sampling_method": sampling_method
                              })

    def train_seq(self, save_model=True):
        """
        Train all segments sequentially, copying the previous segment's parameters to the next.

        Args:
            save_model (bool): Whether to save model parameters.
        """

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

            # Create a new MinMaxOptProb instance for this segment
            segment = MinMaxOptProb(
                t_min=t_min, t_max=t_max, x_min=self.x_min, x_max=self.x_max, N_t=self.N_t, N_x=self.N_x, 
                N_bc=self.N_bc, N_ic=self.N_ic, hidden_dims_list=self.hidden_dims_list, seed=self.seed, 
                sigma=self.sigma, N_perturb=self.N_perturb, k_h_rf_times=self.k_h_rf_times, u0=current_u0, u1=self.u1, u2=self.u2, project=self.project,
                gamma=self.gamma, alpha=self.alpha, N_test_gamma=self.N_test_gamma, adjust_gamma=self.adjust_gamma,
                L1reg=self.L1reg, sampling_method=self.sampling_method, notes=self.notes, lambda_tv=self.lambda_tv,
                init_lrs=self.init_lrs, grade_epochs=self.grade_epochs, decay_rates=self.decay_rates, 
                init_lr_stage2=self.init_lr_stage2, epoch_stage2=self.epoch_stage2, decay_rate_stage2=self.decay_rate_stage2,
                f_adaptive_weight=self.f_adaptive_weight, f_opt_kh=self.f_opt_kh, res=self.res, res_visc=self.res_visc, 
                use_loss_fn_elem_int=self.use_loss_fn_elem_int, total_grade=self.total_grade, k=self.k, use_a=self.use_a, seq_train=True, set_wandb=self.set_wandb
            )

            # Plot before training
            print(f"\nPlotting before training segment [{t_min:.4f}, {t_max:.4f}]")
            plot_u_theta(
                model=segment.model,
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
                model=segment.model,
                M_t=self.M_t,
                M_x=self.M_x,
                t_min=t_min,
                t_max=t_max,
                x_min=self.x_min,
                x_max=self.x_max,
                upload_fig=True
            )

            # Store the segment and update previous_params with the trained parameters
            self.segments.append(segment)

        # After training all segments, plot the combined output
        print("\nPlotting combined output across all segments from t=0 to total_t_max")
        self.plot_combined_u_theta(M_t_per_segment=self.M_t, M_x=self.M_x, upload_fig=True)
        wandb.finish()


    def save_model(self, model_path_prefix):
        """
        Save the model parameters for all segments.

        Args:
            model_path_prefix (str): Prefix for the model file paths.
        """
        for i, segment in enumerate(self.segments):
            model_path = f"{model_path_prefix}_segment_{i}.pkl"
            segment.save_model(model_path)
            print(f"Saved segment {i} model to {model_path}")

    def load_model(self, model_path_prefix):
        """
        Load the model parameters for all segments.

        Args:
            model_path_prefix (str): Prefix for the model file paths.
        """
        if not self.segments:
            raise ValueError("Segments must be trained or initialized before loading models.")
        for i in range(self.num_segments):
            model_path = f"{model_path_prefix}_segment_{i}.pkl"
            segment = self.segments[i]
            segment.load_model(model_path)
            print(f"Loaded segment {i} model from {model_path}")

    def plot_combined_u_theta(self, M_t_per_segment=1000, M_x=2000, upload_fig=False):
        """
        Plot the combined output of all trained segments over the full time domain [0, total_t_max].

        Args:
            M_t_per_segment (int): Number of grid points in t per segment.
            M_x (int): Number of grid points in x.
        """
        # Generate time grids for each segment based on self.seq_T
        t_grids = []
        for i in range(self.num_segments):
            t_min = self.seq_T[i]
            t_max = self.seq_T[i + 1]
            t_segment = jnp.linspace(t_min, t_max, M_t_per_segment+1)[:-1]
            t_grids.append(t_segment)

        # Concatenate all segment time grids into a full time grid
        t_grid_full = jnp.concatenate(t_grids)
        # Create the spatial grid
        x_grid = jnp.linspace(self.x_min, self.x_max, M_x)

        # Create a meshgrid for t and x
        t, x = jnp.meshgrid(t_grid_full, x_grid, indexing='ij')  # Shapes: (total_M_t, M_x)

        # Flatten the grid for evaluation
        tx_points_full = jnp.stack([t.ravel(), x.ravel()], axis=1)  # Shape: (total_M_t * M_x, 2)

        # Determine which segment each time point belongs to
        segment_indices = jnp.digitize(tx_points_full[:, 0], self.seq_T) - 1
        segment_indices = jnp.clip(segment_indices, 0, self.num_segments - 1)

        # Collect outputs from each segment
        u_values_full = []
        for i in range(self.num_segments):
            mask = (segment_indices == i)
            tx_points_segment = tx_points_full[mask]
            if tx_points_segment.size > 0:
                segment = self.segments[i]
                u_values_segment = segment.model.forward(tx_points_segment)
                u_values_full.append(u_values_segment)

        # Combine all segment outputs into a single array
        u_values_full = jnp.concatenate(u_values_full)
        u_values_full = u_values_full.reshape(len(t_grid_full), M_x)

        # Create the plot
        t_range = self.total_t_max - self.total_t_min
        fig = plt.figure(figsize=(10, 4 + 4 * t_range))
        gs = gridspec.GridSpec(2, 1, height_ratios=[t_range, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # 2D color mesh plot
        pcm = ax1.pcolormesh(x_grid, t_grid_full, u_values_full, shading='nearest', cmap='viridis')
        fig.colorbar(pcm, ax=ax1, label='$u_\\theta(t, x)$')
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        ax1.set_title(f'Combined Neural Network $u_\\theta$ over [{self.total_t_min}, {self.total_t_max}]')

        # 1D plot at t = t_max
        # Get the last segment
        segment = self.segments[-1]
        t_jnp = jnp.full_like(x_grid, self.total_t_max)
        tx_points_max = jnp.stack([t_jnp, x_grid], axis=1)  # Shape: (M_x, 2)
        u_at_max = segment.model.forward(tx_points_max)
        ax2.plot(x_grid, u_at_max)
        ax2.set_xlabel('x')
        ax2.set_ylabel('$u_\\theta(t_{max}, x)$')
        ax2.set_title(f'Neural Network Function at t = {self.total_t_max:.2f}')

        # Adjust layout and display the main plot
        plt.tight_layout()
        # Upload the figure to wandb
        if upload_fig:
            wandb.log({f"u_theta_tmin{t_min:.1f}_tmax_{t_max:.1f}": wandb.Image(fig)})

        plt.show()