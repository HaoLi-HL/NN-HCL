import jax
import jax.numpy as jnp
import numpy as np
import optax
from datetime import datetime
import os
import pickle
from minmax_opt_class import MinMaxOptProb
from my_helpers import plot_u_theta, plot_loss_history
import matplotlib.pyplot as plt




class SequentialMinMaxOptProb:
    def __init__(self, num_segments, total_t_min, total_t_max, seq_T=None, N_t=10, N_x=10, lr_theta=0.01, hidden_dims=(64, 64, 64), 
                 seed=0, sigma=0.1, N_perturb=5000, N_bc=100, N_ic=100, x_min=0, x_max=1, k_h_rf_times=1, 
                 u0=None, u1=None, u2=None, lambda_bc_static=0.9, alpha_p_coeffs=0, f_adaptive=1000, 
                 nu=0.001, res_visc=None, res=None, if_pre_train=False, M_t=500, M_x=500, use_loss_elem_int=True):
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
        
        self.total_t_min = total_t_min
        self.total_t_max = total_t_max
        self.seq_T = np.array(seq_T)
        if self.seq_T is None:
            self.num_segments = num_segments
            self.t_segment_length = (total_t_max-total_t_min) / num_segments
            self.seq_T = np.array([self.total_t_min + i * self.t_segment_length for i in range(self.num_segments+1)])
        else:
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
        self.use_loss_elem_int = use_loss_elem_int


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
            t_min = self.seq_T[i]
            t_max = self.seq_T[i+1]

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
                u0=current_u0, u1=self.u1, u2=self.u2, nu=self.nu, res_visc=self.res_visc, res=self.res,
                use_loss_elem_int=self.use_loss_elem_int,
                )

            # If not the first segment, copy the previous segment's parameters
            # if previous_params is not None:
            #     segment.set_params(previous_params)

            # Plot before training
            print(f"\nPlotting before training segment [{t_min:.4f}, {t_max:.4f}]")
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
                print(f"\nPre-training segment [{t_min:.4f}, {t_max:.4f}]")
                segment.pre_train_process(pre_train_num_iterations, save_loss=save_loss, save_model=save_model)
                plot_loss_history(segment.pre_loss_int_history, segment.pre_loss_bc_history, 'Pre-train Loss History')

            # Train the segment
            if train_num_iterations != 0:
                print(f"\nTraining segment [{t_min:.4f}, {t_max:.4f}]")
                segment.train_process(train_num_iterations, save_loss=save_loss, save_model=save_model)
                plot_loss_history(segment.loss_int_history, segment.loss_bc_history, 'Train Loss History')

            # Plot after training
            print(f"\nPlotting after training segment [{t_min:.4f}, {t_max:.4f}]")
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
        self.plot_solution_at_time(self.total_t_max)


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
        # Generate time grids for each segment based on self.seq_T
        t_grids = []
        for i in range(self.num_segments):
            t_start = self.seq_T[i]
            t_end = self.seq_T[i + 1]
            t_segment = jnp.linspace(t_start, t_end, M_t_per_segment+1)[:-1]
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
                u_values_segment = segment.model(segment.params, tx_points_segment)
                u_values_full.append(u_values_segment)

        # Combine all segment outputs into a single array
        u_values_full = jnp.concatenate(u_values_full)
        u_values_full = u_values_full.reshape(len(t_grid_full), M_x)

        # Create the plot
        height = (self.total_t_max - self.total_t_min) * 3
        width = (self.x_max - self.x_min) * 3 + 1
        plt.figure(figsize=(width, height))
        plt.pcolormesh(x_grid, t_grid_full, u_values_full, shading='nearest', cmap='viridis')
        plt.colorbar(label='$u_\\theta(t, x)$')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(f'Combined Neural Network Function $u_\\theta$ over [{self.total_t_min}, {self.total_t_max}]')
        plt.show()

    def plot_solution_at_time(self, T, M_x=100):
        """
        Plot the solution u_theta at a specific time T.

        Args:
            T (float): The time at which to plot the solution.
            M_x (int): Number of grid points in the x-direction (default: 100).
        """
        # Find the segment that contains time T
        segment_index = None
        for i in range(len(self.seq_T) - 1):
            if self.seq_T[i] <= T < self.seq_T[i + 1]:
                segment_index = i
                break
        if segment_index is None:
            if T == self.seq_T[-1]:  # Allow T to be exactly at the end
                segment_index = len(self.seq_T) - 2
            else:
                raise ValueError(f"Time T={T} is outside the defined segments [{self.seq_T[0]}, {self.seq_T[-1]}]")

        # Get the segment
        segment = self.segments[segment_index]

        # Generate spatial grid and input points
        x_grid = jnp.linspace(self.x_min, self.x_max, M_x)
        t_fixed = jnp.full_like(x_grid, T)  # Array of T repeated for each x
        tx_points = jnp.stack([t_fixed, x_grid], axis=1)  # Shape: (M_x, 2)

        # Evaluate the neural network at these points
        u_values = segment.model(segment.params, tx_points)  # Shape: (M_x,)

        # Plot the solution
        plt.figure(figsize=(8, 6))
        plt.plot(x_grid, u_values, label=f'$u_\\theta$ at t={T}')
        plt.xlabel('x')
        plt.ylabel('$u_\\theta$')
        plt.title(f'Solution at Time T={T}')
        plt.legend()
        plt.show()