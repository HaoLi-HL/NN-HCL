import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import linen as nn
from NN_class import MLP
from jax import vmap
import wandb


def plot_loss_history(loss_int, loss_bc, title):
    plt.figure()
    plt.plot(loss_int, label='Loss Interior')
    plt.plot(loss_bc, label='Loss Boundary')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.title(title)
    plt.show()

import matplotlib.gridspec as gridspec
# Function to plot u_theta
def plot_u_theta(model, params=None, M_t=1000, M_x=2000, t_min=0.0, t_max=1.0, x_min=0.0, x_max=1.0, ts=None, upload_fig=False):
    """
    Plot the neural network function u_theta over a specified domain [t_min, t_max] x [x_min, x_max],
    and include a 1D plot of u_theta at t = t_max. Additionally, plot separate 1D snapshots for each timestamp in ts if provided.

    Args:
        model: Neural network u_theta.
        params: Parameters (pytree) of the neural network.
        M_t: Number of grid points in the t-direction (default: 1000).
        M_x: Number of grid points in the x-direction (default: 1000).
        t_min: Minimum value for t (default: 0.0).
        t_max: Maximum value for t (default: 1.0).
        x_min: Minimum value for x (default: 0.0).
        x_max: Maximum value for x (default: 1.0).
        ts: List or array of timestamps for which to plot individual 1D snapshots (default: None).
    """
    if ts is None:
        ts = []

    # Create a grid for t and x over the specified ranges
    t_grid = jnp.linspace(t_min, t_max, M_t)
    x_grid = jnp.linspace(x_min, x_max, M_x)
    t, x = jnp.meshgrid(t_grid, x_grid, indexing='ij')  # Shapes: (M_t, M_x)

    # Flatten the grid for batch evaluation
    tx_points = jnp.stack([t.ravel(), x.ravel()], axis=1)  # Shape: (M_t * M_x, 2)

    # Evaluate u_theta on the grid points
    if params is None:
        u_values = model.forward(tx_points)
    else:
        u_values = model(params, tx_points)  # Shape: (M_t * M_x, 1)
    u_values = u_values.reshape(M_t, M_x)  # Reshape to (M_t, M_x)

    # Create figure with two subplots for the main plot
    # Create figure with two subplots for the main plot, with size proportional to (t_max - t_min)
    t_range = t_max - t_min
    fig = plt.figure(figsize=(10, 4 + 4 * t_range))
    gs = gridspec.GridSpec(2, 1, height_ratios=[t_range, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # 2D color mesh plot
    pcm = ax1.pcolormesh(x_grid, t_grid, u_values, shading='nearest', cmap='viridis')
    fig.colorbar(pcm, ax=ax1, label='$u_\\theta(t, x)$')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_title('Neural Network Function $u_\\theta$')

    # 1D plot at t = t_max
    ax2.plot(x_grid, u_values[-1, :])
    ax2.set_xlabel('x')
    ax2.set_ylabel('$u_\\theta(t_{max}, x)$')
    ax2.set_title(f'Neural Network Function at t = {t_max:2f}')

    # Adjust layout and display the main plot
    plt.tight_layout()
    # Upload the main figure to wandb
    if upload_fig:
        wandb.log({f"u_theta_tmin{t_min:.1f}_tmax_{t_max:.1f}": wandb.Image(fig)})
    plt.show()

    # Plot separate figures for each timestamp in ts
    for t_val in ts:
        # Find the closest index in t_grid
        idx = jnp.argmin(jnp.abs(t_grid - t_val))
        # Create a new figure for the snapshot
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(x_grid, u_values[idx, :])
        ax.set_xlabel('x')
        ax.set_ylabel('$u_\\theta(t, x)$')
        ax.set_title(f'Neural Network Function at t = {t_val:.2f} (closest t = {t_grid[idx]:.2f})')
        plt.tight_layout()
        plt.show()

def plot_kh(pwp, M_factor=10):
    """
    Plot the piecewise linear polynomial k_h over the domain [t_min, t_max] x [x_min, x_max]
    on a triangular mesh, using JAX for efficiency.

    Args:
        pwp: PiecewiseLinearPolynomials instance with N_t, N_x, t_min, t_max, x_min, x_max, and coeffs.
        M_factor: Factor to refine the plotting grid (default: 10).
    """
    # Retrieve ranges and grid parameters from pwp
    t_min, t_max = pwp.t_min, pwp.t_max
    x_min, x_max = pwp.x_min, pwp.x_max
    N_t, N_x = pwp.N_t, pwp.N_x
    tau = (t_max - t_min) / N_t  # Step size for t
    hx = (x_max - x_min) / N_x   # Step size for x

    # Create fine grid using JAX arrays
    M_t, M_x = N_t * M_factor, N_x * M_factor
    t_grid = jnp.linspace(t_min, t_max, M_t)
    x_grid = jnp.linspace(x_min, x_max, M_x)
    t, x = jnp.meshgrid(t_grid, x_grid, indexing='ij')  # Shapes: (M_t, M_x)

    # Flatten the grid for vectorized operations
    tx_points = jnp.stack([t.ravel(), x.ravel()], axis=1)  # Shape: (M_t * M_x, 2)

    # Function to find the element index for a single point
    def find_element(tx):
        t_val, x_val = tx
        # Find the rectangle containing the point
        i = jnp.floor((t_val - t_min) / tau).astype(int)
        j = jnp.floor((x_val - x_min) / hx).astype(int)
        i = jnp.clip(i, 0, N_t - 1)
        j = jnp.clip(j, 0, N_x - 1)

        # Get vertices of the two triangles in this rectangle
        A = jnp.array([t_min + i * tau, x_min + j * hx])
        B = jnp.array([t_min + (i + 1) * tau, x_min + j * hx])
        C = jnp.array([t_min + i * tau, x_min + (j + 1) * hx])
        D = jnp.array([t_min + (i + 1) * tau, x_min + (j + 1) * hx])

        # Check which triangle contains the point
        in_triangle_1 = point_in_triangle(tx, A, B, C)
        in_triangle_2 = point_in_triangle(tx, B, D, C)
        global_idx = jnp.where(
            in_triangle_1,
            2 * (i * N_x + j),        # First triangle
            jnp.where(
                in_triangle_2,
                2 * (i * N_x + j) + 1,  # Second triangle
                -1                      # Not in any triangle
            )
        )
        return global_idx

    # Vectorize the find_element function over all points
    element_indices = vmap(find_element)(tx_points)  # Shape: (M_t * M_x,)

    # Evaluate k_h at all points using their corresponding element indices
    kh_values = pwp.evaluate(tx_points, element_indices)  # Shape: (M_t * M_x,)
    kh_values = kh_values.reshape(M_t, M_x)  # Reshape to (M_t, M_x)

    # Plot using pcolormesh
    height = int((t_max - t_min)*4)
    width = int((x_max - x_min)*4 + 1)
    plt.figure(figsize=(width, height))
    plt.pcolormesh(x_grid, t_grid, kh_values, shading='nearest', cmap='viridis')
    plt.colorbar(label='$k_h(t, x)$')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Piecewise Linear Polynomial $k_h$ on Triangular Mesh')
    plt.show()

def point_in_triangle(P, A, B, C):
    """
    Check if point P is inside triangle ABC using barycentric coordinates.

    Args:
        P: Point to check, shape (2,).
        A, B, C: Triangle vertices, each shape (2,).

    Returns:
        Boolean: True if P is inside the triangle, False otherwise.
    """
    def sign(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    d1 = sign(P, A, B)
    d2 = sign(P, B, C)
    d3 = sign(P, C, A)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)  # True if all signs are the same or zero


def plot_numerical_solution(file_path='data/burgers_sin_ref.npz', ts=None):
    """
    Plot the numerical solution u over the time and space domain from the saved .npz file,
    including a 2D color mesh plot and a 1D plot at the maximum time. Additionally, plot
    separate 1D snapshots for each timestamp in ts if provided.

    Args:
        file_path: Path to the .npz file containing 'times', 'x', and 'u' (default: 'data/burgers_sin_ref.npz').
        ts: List or array of timestamps for which to plot individual 1D snapshots (default: None).
    """
    if ts is None:
        ts = []

    # Load the data from the .npz file
    data = np.load(file_path)
    times = data['times']
    x = data['x']
    u = data['u']  # Shape: (n_times, n_x)

    # Create figure with two subplots for the main plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 2D color mesh plot
    pcm = ax1.pcolormesh(x, times, u, shading='nearest', cmap='viridis')
    fig.colorbar(pcm, ax=ax1, label='$u(t, x)$')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_title('Numerical Solution $u$')

    # 1D plot at t = max(times)
    ax2.plot(x, u[-1, :])
    ax2.set_xlabel('x')
    ax2.set_ylabel('$u(t_{max}, x)$')
    ax2.set_title(f'Numerical Solution at t = {times[-1]:.2f}')

    # Adjust layout and display the main plot
    plt.tight_layout()
    plt.show()

    # Plot separate figures for each timestamp in ts
    for t_val in ts:
        # Find the closest index in times
        idx = np.argmin(np.abs(times - t_val))
        # Create a new figure for the snapshot
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(x, u[idx, :])
        ax.set_xlabel('x')
        ax.set_ylabel('$u(t, x)$')
        ax.set_title(f'Numerical Solution at t = {t_val:.2f} (closest t = {times[idx]:.2f})')
        plt.tight_layout()
        plt.show()


def compute_errors(model, u_ref=None, file_path='data/burgers_sin_ref.npz', params=None, T=None, x_min=-1.0, x_max=1.0, M_x=601):
    """
    Compute the relative error at time T and the space-time relative error between
    the neural network approximation u* and the reference solution u.

    Args:
        model: Neural network u_theta.
        u_ref: Reference function u_ref(t, x) where t is scalar and x is array (default: None).
        file_path: Path to the .npz file containing 'times', 'x', and 'u' (default: 'data/burgers_sin_ref.npz').
        params: Parameters (pytree) of the neural network (default: None).
        T: The final time for the error at T. If None, uses the maximum time in the data or t_max (default: None).
        x_min: Minimum x value (default: -1.0).
        x_max: Maximum x value (default: 1.0).
        M_x: Number of x grid points (default: 601).

    Returns:
        err_T: Relative error at time T.
        err_spacetime: Space-time relative error.
    """
    # Load numerical solution data or use reference function
    if u_ref is None:
        data = np.load(file_path)
        times = data['times']
        x = data['x']
        u_num = data['u']  # Shape: (n_times, n_x)
    else:
        times = np.linspace(0, T, M_x) if T is not None else np.linspace(0, 1.0, M_x)
        x = np.linspace(x_min, x_max, M_x)
        u_num = np.array([u_ref(t, x) for t in times])  # Shape: (n_times, n_x)

    if T is None:
        T = times[-1]

    # Create grid for evaluation (matching the numerical grid)
    t_grid, x_grid = np.meshgrid(times, x, indexing='ij')  # Shapes: (n_t, n_x)
    tx_points = np.stack([t_grid.ravel(), x_grid.ravel()], axis=1)  # Shape: (n_t * n_x, 2)

    # Evaluate the neural network on the grid points
    if params is None:
        u_star_values = model.forward(tx_points)
    else:
        u_star_values = model(params, tx_points)  # Assuming shape: (n_points, 1) or (n_points,)
    u_star_values = np.asarray(u_star_values).squeeze()  # Ensure it's 1D array
    u_star = u_star_values.reshape(len(times), len(x))  # Shape: (n_t, n_x)

    # Compute relative error at t = T
    idx_T = np.argmin(np.abs(times - T))
    u_T = u_num[idx_T, :]
    u_star_T = u_star[idx_T, :]
    num_T = np.trapezoid(np.abs(u_star_T - u_T), x)
    den_T = np.trapezoid(np.abs(u_T), x)
    err_T = num_T / den_T if den_T != 0 else np.inf

    # Compute space-time relative error
    abs_diff = np.abs(u_star - u_num)  # Shape: (n_t, n_x)
    abs_u = np.abs(u_num)

    # Integrate over x for each t
    int_x_abs_diff = np.trapezoid(abs_diff, x, axis=1)  # Shape: (n_t,)
    int_x_abs_u = np.trapezoid(abs_u, x, axis=1)

    # Integrate over t
    spacetime_num = np.trapezoid(int_x_abs_diff, times)
    spacetime_den = np.trapezoid(int_x_abs_u, times)
    err_spacetime = spacetime_num / spacetime_den if spacetime_den != 0 else np.inf

    # Print the errors
    print(f"Relative error at T: {err_T}")
    print(f"Space-time relative error: {err_spacetime}")

    return err_T, err_spacetime


def compute_errors_combined_u_theta(seq_model, u_ref=None, file_path='data/burgers_sin_ref.npz', T=None, x_min=-1.0, x_max=1.0, M_x=601):
    """
    Compute the relative error at time T and the space-time relative error between
    the neural network approximation u* and the reference solution u.

    Args:
        seq_model: Instance of SeqMinMaxOptProb representing the combined neural networks.
        u_ref: Reference function u_ref(t, x) where t is scalar and x is array (default: None).
        file_path: Path to the .npz file containing 'times', 'x', and 'u' (default: 'data/burgers_sin_ref.npz').
        T: The final time for the error at T. If None, uses the maximum time in the data or seq_model.total_t_max (default: None).
        x_min: Minimum x value (default: -1.0).
        x_max: Maximum x value (default: 1.0).
        M_x: Number of x grid points (default: 601).

    Returns:
        err_T: Relative error at time T.
        err_spacetime: Space-time relative error.
    """
    # Load numerical solution data or use reference function
    if u_ref is None:
        data = np.load(file_path)
        times = data['times']
        x = data['x']
        u_num = data['u']  # Shape: (n_times, n_x)
    else:
        T = seq_model.total_t_max if T is None else T
        times = np.linspace(seq_model.total_t_min, T, M_x)
        x = np.linspace(x_min, x_max, M_x)
        u_num = np.array([u_ref(t, x) for t in times])  # Shape: (n_times, n_x)

    if T is None:
        T = times[-1]

    # Create grid for evaluation (matching the numerical grid)
    t_grid, x_grid = np.meshgrid(times, x, indexing='ij')  # Shapes: (n_t, n_x)
    tx_points = np.stack([t_grid.ravel(), x_grid.ravel()], axis=1)  # Shape: (n_t * n_x, 2)

    # Determine which segment each time point belongs to
    segment_indices = jnp.digitize(tx_points[:, 0], seq_model.seq_T, right=True) - 1
    segment_indices = jnp.clip(segment_indices, 0, seq_model.num_segments - 1)

    # Collect outputs from each segment
    u_star_values = []
    for i in range(seq_model.num_segments):
        mask = (segment_indices == i)
        tx_points_segment = tx_points[mask]
        if tx_points_segment.size > 0:
            segment = seq_model.segments[i]
            u_values_segment = segment.model.forward(tx_points_segment)
            u_star_values.append(u_values_segment)

    # Combine all segment outputs into a single array
    u_star_values = jnp.concatenate(u_star_values)
    u_star_values = np.asarray(u_star_values).squeeze()  # Ensure it's 1D array
    u_star = u_star_values.reshape(len(times), len(x))  # Shape: (n_t, n_x)

    # Compute relative error at t = T
    idx_T = np.argmin(np.abs(times - T))
    u_T = u_num[idx_T, :]
    u_star_T = u_star[idx_T, :]
    num_T = np.trapezoid(np.abs(u_star_T - u_T), x)
    den_T = np.trapezoid(np.abs(u_T), x)
    err_T = num_T / den_T if den_T != 0 else np.inf

    # Compute space-time relative error
    abs_diff = np.abs(u_star - u_num)  # Shape: (n_t, n_x)
    abs_u = np.abs(u_num)

    # Integrate over x for each t
    int_x_abs_diff = np.trapezoid(abs_diff, x, axis=1)  # Shape: (n_t,)
    int_x_abs_u = np.trapezoid(abs_u, x, axis=1)

    # Integrate over t
    spacetime_num = np.trapezoid(int_x_abs_diff, times)
    spacetime_den = np.trapezoid(int_x_abs_u, times)
    err_spacetime = spacetime_num / spacetime_den if spacetime_den != 0 else np.inf

    # Print the errors
    print(f"Relative error at T: {err_T}")
    print(f"Space-time relative error: {err_spacetime}")

    return err_T, err_spacetime


def plot_comparison_seperate(model, u_ref=None, file_path='data/burgers_sin_ref.npz', 
                             params=None, ts=None, x_min=-1.0, x_max=1.0, M_x=601):
    """
    Plot both the neural network solution and the reference solution for each timestamp in ts,
    with both solutions overlaid on the same axes in subplots (one subplot per timestamp).

    Args:
        model: Neural network u_theta.
        u_ref: Reference function u_ref(t, x) where t is scalar and x is array (default: None).
        file_path: Path to the .npz file containing 'times', 'x', and 'u' (default: 'data/burgers_sin_ref.npz').
        params: Parameters (pytree) of the neural network (default: None).
        ts: List or array of timestamps for which to plot the solutions (default: None).
        x_min: Minimum x value (default: -1.0).
        x_max: Maximum x value (default: 1.0).
        M_x: Number of x grid points (default: 601).
    """
    if ts is None:
        ts = []
    if not ts:
        print("No timestamps provided. Nothing to plot.")
        return

    # Load numerical solution data or use reference function
    if u_ref is None:
        data = np.load(file_path)
        times = data['times']
        x = data['x']
        u_num = data['u']  # Shape: (n_times, n_x)
    else:
        x = np.linspace(x_min, x_max, M_x)
        times = np.array(ts)
        u_num = np.array([u_ref(t, x) for t in times])  # Shape: (len(ts), n_x)

    # Create subplots: one for each timestamp
    fig, axs = plt.subplots(len(ts), 1, figsize=(10, 4 * len(ts)), squeeze=False)
    axs = axs.flatten()  # Ensure axs is iterable even for len(ts)=1

    for i, t_val in enumerate(ts):
        # Reference solution: find closest time index or use exact t_val
        idx = np.argmin(np.abs(times - t_val)) if u_ref is None else i
        u_ref_t = u_num[idx, :]

        # Neural network solution: evaluate at exact t_val on the same x grid
        x_jnp = jnp.asarray(x)
        t_jnp = jnp.full_like(x_jnp, t_val)
        tx_points = jnp.stack([t_jnp, x_jnp], axis=1)  # Shape: (n_x, 2)

        if params is None:
            u_nn_values = model.forward(tx_points)
        else:
            u_nn_values = model(params, tx_points)
        u_nn_t = np.asarray(u_nn_values).squeeze()  # Shape: (n_x,)

        # Plot both on the same axes
        ax = axs[i]
        ax.plot(x, u_ref_t, label='Ref. solution', linewidth=2)
        ax.plot(x, u_nn_t, label='NN solution', linestyle='--', linewidth=2)
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title(f'Solutions at t = {t_val:.2f}' + (f' (numerical closest t = {times[idx]:.2f})' if u_ref is None else ''))

    plt.tight_layout()
    plt.show()


def plot_comparison_together(model, u_ref=None, file_path='data/burgers_sin_ref.npz', 
                             params=None, ts=None, save_path='data/sine_wave_plot_together.eps', title=None, x_min=-1.0, x_max=1.0, M_x=601):
    """
    Plot both the neural network solution (solid line) and the reference solution (dashed line)
    for each timestamp in ts, with all comparisons overlaid on the same single figure. For each
    time, the pair of lines (numerical and neural) share the same color, progressing from light
    to dark as time increases.

    Args:
        model: Neural network u_theta.
        u_ref: Reference function u_ref(t, x) where t is scalar and x is array (default: None).
        file_path: Path to the .npz file containing 'times', 'x', and 'u' (default: 'data/burgers_sin_ref.npz').
        params: Parameters (pytree) of the neural network (default: None).
        ts: List or array of timestamps for which to plot the solutions (default: None).
        save_path: Path to save the .eps file (default: 'data/sine_wave_plot_together.eps').
        title: Title of the plot (default: None).
        x_min: Minimum x value (default: -1.0).
        x_max: Maximum x value (default: 1.0).
        M_x: Number of x grid points (default: 601).
    """
    if ts is None:
        ts = []
    if not ts:
        print("No timestamps provided. Nothing to plot.")
        return

    # Load numerical solution data or use reference function
    if u_ref is None:
        data = np.load(file_path)
        times = data['times']
        x = data['x']
        u_num = data['u']  # Shape: (n_times, n_x)
    else:
        x = np.linspace(x_min, x_max, M_x)
        times = np.array(ts)
        u_num = np.array([u_ref(t, x) for t in times])  # Shape: (len(ts), n_x)

    # Create a single figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get colormap (light to dark blue)
    cmap = plt.cm.Blues

    # Normalize times for color mapping
    ts_array = np.array(ts)
    min_t = ts_array.min()
    max_t = ts_array.max()
    if min_t == max_t:
        norms = np.full(len(ts), 0.5)  # Medium color if all times are the same
    else:
        norms = 0.6 * (ts_array - min_t) / (max_t - min_t) + 0.4

    for i, t_val in enumerate(ts):
        norm = norms[i]
        color = cmap(norm)

        # Reference solution: find closest time index or use exact t_val
        idx = np.argmin(np.abs(times - t_val)) if u_ref is None else i
        u_ref_t = u_num[idx, :]

        # Neural network solution: evaluate at exact t_val on the same x grid
        x_jnp = jnp.asarray(x)
        t_jnp = jnp.full_like(x_jnp, t_val)
        tx_points = jnp.stack([t_jnp, x_jnp], axis=1)  # Shape: (n_x, 2)

        if params is None:
            u_nn_values = model.forward(tx_points)
        else:
            u_nn_values = model(params, tx_points)
        u_nn_t = np.asarray(u_nn_values).squeeze()  # Shape: (n_x,)

        # Plot both on the same axes with same color
        ax.plot(x, u_ref_t, linestyle='--', color=color, linewidth=3, label=f'Ref. solution t={t_val:.2f}')
        ax.plot(x, u_nn_t, linestyle='-', color=color, linewidth=2, label=f'NN solution t={t_val:.2f}')

    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, format='eps', bbox_inches='tight')
    plt.show()

def plot_comparison_seperate_combined_u_theta(seq_model, u_ref=None, file_path='data/burgers_sin_ref.npz', ts=None, x_min=-1.0, x_max=1.0, M_x=601):
    """
    Plot both the neural network solution and the reference solution for each timestamp in ts,
    with both solutions overlaid on the same axes in subplots (one subplot per timestamp).

    Args:
        seq_model: Instance of SeqMinMaxOptProb representing the combined neural networks.
        u_ref: Reference function u_ref(t, x) where t is scalar and x is array (default: None).
        file_path: Path to the .npz file containing 'times', 'x', and 'u' (default: 'data/burgers_sin_ref.npz').
        ts: List or array of timestamps for which to plot the solutions (default: None).
        x_min: Minimum x value (default: -1.0).
        x_max: Maximum x value (default: 1.0).
        M_x: Number of x grid points (default: 601).
    """
    if ts is None:
        ts = []
    if not ts:
        print("No timestamps provided. Nothing to plot.")
        return

    # Load numerical solution data or use reference function
    if u_ref is None:
        data = np.load(file_path)
        times = data['times']
        x = data['x']
        u_num = data['u']  # Shape: (n_times, n_x)
    else:
        x = np.linspace(x_min, x_max, M_x)
        times = np.array(ts)
        u_num = np.array([u_ref(t, x) for t in times])  # Shape: (len(ts), n_x)

    # Create subplots: one for each timestamp
    fig, axs = plt.subplots(len(ts), 1, figsize=(10, 4 * len(ts)), squeeze=False)
    axs = axs.flatten()  # Ensure axs is iterable even for len(ts)=1

    for i, t_val in enumerate(ts):
        # Reference solution: find closest time index or use exact t_val
        idx = np.argmin(np.abs(times - t_val)) if u_ref is None else i
        u_ref_t = u_num[idx, :]

        # Find the segment that contains time t_val
        segment_index = None
        for j in range(len(seq_model.seq_T) - 1):
            if seq_model.seq_T[j] <= t_val <= seq_model.seq_T[j + 1]:
                segment_index = j
                break
        if segment_index is None:
            if t_val == seq_model.seq_T[-1]:  # Allow t_val to be exactly at the end
                segment_index = len(seq_model.seq_T) - 2
            else:
                raise ValueError(f"Time t_val={t_val} is outside the defined segments [{seq_model.seq_T[0]}, {seq_model.seq_T[-1]}]")

        # Get the segment
        segment = seq_model.segments[segment_index]

        # Neural network solution: evaluate at exact t_val on the same x grid
        x_jnp = jnp.asarray(x)
        t_jnp = jnp.full_like(x_jnp, t_val)
        tx_points = jnp.stack([t_jnp, x_jnp], axis=1)  # Shape: (n_x, 2)

        u_nn_values = segment.model.forward(tx_points)
        u_nn_t = np.asarray(u_nn_values).squeeze()  # Shape: (n_x,)

        # Plot both on the same axes
        ax = axs[i]
        ax.plot(x, u_ref_t, label='Ref. solution', linewidth=2)
        ax.plot(x, u_nn_t, label='NN solution', linestyle='--', linewidth=2)
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title(f'Solutions at t = {t_val:.2f}' + (f' (numerical closest t = {times[idx]:.2f})' if u_ref is None else ''))

    plt.tight_layout()
    plt.show()

