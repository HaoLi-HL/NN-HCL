import jax
import jax.numpy as jnp
import optax
from datetime import datetime
import os
import pickle
from minmax_opt_class import MinMaxOptProb
from my_helpers import plot_u_theta, plot_loss_history
import matplotlib.pyplot as plt




class SequentialMinMaxOptProb:
    def __init__(self, num_segments, total_t_min, total_t_max, N_t=10, N_x=10, lr_theta=0.01, hidden_dims=(64, 64, 64), 
                 seed=0, sigma=0.1, N_perturb=5000, N_bc=100, N_ic=100, x_min=0, x_max=1, k_h_rf_times=1, 
                 u0=None, u1=None, u2=None, lambda_bc_static=0.9, alpha_p_coeffs=0, f_adaptive=1000, 
                 nu=0.001, res_visc=None, res=None, if_pre_train=False, M_t=500, M_x=500):
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
        self.num_segments = num_segments
        self.total_t_min = total_t_min
        self.total_t_max = total_t_max
        self.t_segment_length = (total_t_max-total_t_min) / num_segments
        self.u0 = u0
        self.u1 = u1
        self.u2 = u2
        self.x_min = x_min
        self.x_max = x_max
        self.segments = []  # List to store trained segment instances
        # Store other parameters for segment creation
        self.N_t = N_t
        self.N_x = N_x
        self.lr_theta = lr_theta
        self.hidden_dims = hidden_dims
        self.seed = seed
        self.sigma = sigma
        self.N_perturb = N_perturb
        self.N_bc = N_bc
        self.N_ic = N_ic
        self.k_h_rf_times = k_h_rf_times
        # self.lambda_bc_static = lambda_bc_static
        # self.alpha_p_coeffs = alpha_p_coeffs
        # self.f_adaptive = f_adaptive
        self.nu = nu
        self.res_visc = res_visc
        self.res = res
        self.if_pre_train = if_pre_train
        self.M_t = M_t
        self.M_x = M_x


    def train_sequential(self, pre_train_num_iterations=0, train_num_iterations=0, save_loss=True, save_model=True):
        """
        Train all segments sequentially, copying the previous segment's parameters to the next.

        Args:
            pre_train_num_iterations (int): Number of pre-training iterations per segment.
            train_num_iterations (int): Number of training iterations per segment.
            save_loss (bool): Whether to save loss history.
            save_model (bool): Whether to save model parameters.
        """
        previous_params = None
        for i in range(self.num_segments):
            # Define time interval for the current segment
            t_min = i * self.t_segment_length + self.total_t_min
            t_max = (i + 1) * self.t_segment_length + self.total_t_min

            # Set initial condition: use u0 for first segment, else use previous segment's final state
            if i == 0:
                current_u0 = self.u0
            else:
                prev_segment = self.segments[i - 1]
                current_u0 = lambda x: prev_segment.model(
                    prev_segment.params,
                    jnp.stack([jnp.ones_like(x) * prev_segment.pwp.t_max, x], axis=1)
                )

            # Create a new MinMaxOptProb instance for this segment
            segment = MinMaxOptProb(
                N_t=self.N_t, N_x=self.N_x, lr_theta=self.lr_theta, hidden_dims=self.hidden_dims,
                seed=self.seed, sigma=self.sigma, N_perturb=self.N_perturb, N_bc=self.N_bc, N_ic=self.N_ic,
                t_min=t_min, t_max=t_max, x_min=self.x_min, x_max=self.x_max, k_h_rf_times=self.k_h_rf_times,
                u0=current_u0, u1=self.u1, u2=self.u2, nu=self.nu, res_visc=self.res_visc, res=self.res
                )

            # If not the first segment, copy the previous segment's parameters
            # if previous_params is not None:
            #     segment.set_params(previous_params)

            # Plot before training
            print(f"\nPlotting before training segment {i + 1}/{self.num_segments} (t = [{t_min:.4f}, {t_max:.4f}])")
            plot_u_theta(
                model=segment.model,
                params=segment.params,
                M_t=self.M_t,
                M_x=self.M_x,
                t_min=t_min,
                t_max=t_max,
                x_min=self.x_min,
                x_max=self.x_max
            )

            # Pre-train the segment
            if pre_train_num_iterations != 0:
                print(f"\nPre-training segment {i + 1}/{self.num_segments} (t = [{t_min:.4f}, {t_max:.4f}])")
                segment.pre_train_process(pre_train_num_iterations, save_loss=save_loss, save_model=save_model)
                plot_loss_history(segment.pre_loss_int_history, segment.pre_loss_bc_history, 'Pre-train Loss History')

            # Train the segment
            if train_num_iterations != 0:
                print(f"\nTraining segment {i + 1}/{self.num_segments} (t = [{t_min:.4f}, {t_max:.4f}])")
                segment.train_process(train_num_iterations, save_loss=save_loss, save_model=save_model)
                plot_loss_history(segment.loss_int_history, segment.loss_bc_history, 'Train Loss History')

            # Plot after training
            print(f"\nPlotting after training segment {i + 1}/{self.num_segments} (t = [{t_min:.4f}, {t_max:.4f}])")
            plot_u_theta(
                model=segment.model,
                params=segment.params,
                M_t=self.M_t,
                M_x=self.M_x,
                t_min=t_min,
                t_max=t_max,
                x_min=self.x_min,
                x_max=self.x_max
            )
            


            # Store the segment and update previous_params with the trained parameters
            self.segments.append(segment)
            # previous_params = segment.params
            # previous_lb = segment.model.lb
            # previous_ub = segment.model.ub

        # After training all segments, plot the combined output
        print("\nPlotting combined output across all segments from t=0 to total_t_max")
        self.plot_combined_u_theta(M_t_per_segment=self.M_t, M_x=self.M_x)


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
    
    def plot_combined_u_theta(self, M_t_per_segment=500, M_x=500):
        """
        Plot the combined output of all trained segments over the full time domain [0, total_t_max].

        Args:
            M_t_per_segment (int): Number of grid points in t per segment.
            M_x (int): Number of grid points in x.
        """
        # Total time points across all segments
        total_M_t = self.num_segments * M_t_per_segment

        # Create full grids for t and x
        t_grid_full = jnp.linspace(0, self.total_t_max, total_M_t)
        x_grid = jnp.linspace(self.x_min, self.x_max, M_x)
        t, x = jnp.meshgrid(t_grid_full, x_grid, indexing='ij')  # Shapes: (total_M_t, M_x)

        # Flatten the grid for evaluation
        tx_points_full = jnp.stack([t.ravel(), x.ravel()], axis=1)  # Shape: (total_M_t * M_x, 2)

        # Determine which segment each time point belongs to
        segment_indices = (tx_points_full[:, 0] / self.t_segment_length).astype(int)
        segment_indices = jnp.clip(segment_indices, 0, self.num_segments - 1)

        # Collect outputs from each segment
        u_values_full = []
        for i in range(self.num_segments):
            mask = (segment_indices == i)
            tx_points_segment = tx_points_full[mask]
            if tx_points_segment.size > 0:
                segment = self.segments[i]
                u_values_segment = segment.model(segment.params, tx_points_segment)
                u_values_full.append(u_values_segment)

        # Combine all segment outputs into a single array
        u_values_full = jnp.concatenate(u_values_full)
        u_values_full = u_values_full.reshape(total_M_t, M_x)

        # Create the plot
        height = int(self.total_t_max * 4)
        width = int((self.x_max - self.x_min) * 4 + 1)
        plt.figure(figsize=(width, height))
        plt.pcolormesh(x_grid, t_grid_full, u_values_full, shading='nearest', cmap='viridis')
        plt.colorbar(label='$u_\\theta(t, x)$')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Combined Neural Network Function $u_\\theta$ over [0, total_t_max]')
        plt.show()
