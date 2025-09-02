import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import linen as nn
from NN_class import MLP
from jax import vmap
import wandb
from os.path import splitext
import matplotlib.gridspec as gridspec
from matplotlib import cm
from PwpTri_class import PiecewiseLinearPolynomialsTri
from PwpRect_class import PiecewiseLinearPolynomialsRect
from PwpHex_class import PiecewiseLinearPolynomialsHex
from PwpTetra_class import PiecewiseLinearPolynomialsTetra


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

def plot_u_theta(prob, params=None, M_t=1000, M_t_per_segment=1000, M_x=2000, t_min=None, t_max=None, x_min=None, x_max=None, y_min=None, y_max=None, M_y=2000, 
                 ts=None, upload_fig=False, save_path=None, fontsize=16, u_ref=None, file_path=None):
    """
    Plot the neural network function u_theta over a specified domain [t_min, t_max] x [x_min, x_max],
    and include a 1D plot of u_theta at t = t_max. Additionally, plot separate 1D snapshots for each timestamp in ts if provided.
    The function handles both MinMaxOptProb (single segment) and SeqMinMaxOptProb (multiple segments).
    If y_min and y_max are provided, assumes 3D input (t,x,y), skips the full domain plot and t_max plot, and plots 2D slices u(t,x,y) at each ts.
    Args:
        prob: The optimization problem object (MinMaxOptProb or SeqMinMaxOptProb).
        params: Parameters (pytree) of the neural network (default: None, uses prob's internal if available).
        M_t: Number of grid points in the t-direction for single segment (default: 1000).
        M_t_per_segment: Number of grid points in t per segment for sequential (default: 1000).
        M_x: Number of grid points in the x-direction (default: 2000).
        t_min: Minimum value for t (default: prob.total_t_min or 0.0).
        t_max: Maximum value for t (default: prob.total_t_max or 1.0).
        x_min: Minimum value for x (default: prob.x_min or 0.0).
        x_max: Maximum value for x (default: prob.x_max or 1.0).
        y_min: Minimum value for y in 3D (default: None).
        y_max: Maximum value for y in 3D (default: None).
        M_y: Number of grid points in the y-direction for 3D (default: 2000).
        ts: List or array of timestamps for which to plot individual 1D snapshots (default: None).
        upload_fig: Whether to upload the figure to wandb (default: False).
        save_path: Path to save the figure (default: None, does not save if None).
        fontsize: Font size for axis labels, ticks, title, and annotations (default: 12).
        u_ref: Reference function u_ref(t, x, y) for 3D comparison, where t is scalar, x and y are 2D arrays (default: None).
        file_path: Path to .npz file with 'times', 'x', 'y', 'u' for reference (default: None).
    """
    if ts is None:
        ts = []
    # Determine if it's a sequential problem
    is_sequential = hasattr(prob, 'num_segments') and hasattr(prob, 'segments') and hasattr(prob, 'seq_T')
    # Set defaults for t_min, t_max, x_min, x_max if not provided
    if t_min is None:
        t_min = getattr(prob, 'total_t_min', 0.0)
    if t_max is None:
        t_max = getattr(prob, 'total_t_max', 1.0)
    if x_min is None:
        x_min = getattr(prob, 'x_min', 0.0)
    if x_max is None:
        x_max = getattr(prob, 'x_max', 1.0)
    # Determine if 3D
    is_3d = (y_min is not None) and (y_max is not None)
    # Define grids
    x_grid = jnp.linspace(x_min, x_max, M_x)
    if is_3d:
        y_grid = jnp.linspace(y_min, y_max, M_y)
    # Define num_segments and bins for digitize
    num_segments = prob.num_segments if is_sequential else 1
    time_bins = prob.seq_T if is_sequential else jnp.array([t_min, t_max + 1e-6])
    if is_3d and file_path is not None:
        data = np.load(file_path)
        times = data['times']
        x_ref = data['x']
        y_ref = data['y']
        u_num = data['u']
        # Assume grids match; otherwise, interpolation would be needed
    if not is_3d:
        if is_sequential:
            # Sequential case
            seq_T = prob.seq_T
            # Generate time grids for each segment
            t_grids = []
            for i in range(num_segments):
                seg_t_min = seq_T[i]
                seg_t_max = seq_T[i + 1]
                t_segment = jnp.linspace(seg_t_min, seg_t_max, M_t_per_segment + 1)[:-1] # Exclude endpoint to avoid overlap
                t_grids.append(t_segment)
            # Concatenate all segment time grids into a full time grid
            t_grid_full = jnp.concatenate(t_grids)
            # Create the spatial grid
            # Create a meshgrid for t and x
            t, x = jnp.meshgrid(t_grid_full, x_grid, indexing='ij') # Shapes: (total_M_t, M_x)
            # Flatten the grid for evaluation
            tx_points_full = jnp.stack([t.ravel(), x.ravel()], axis=1) # Shape: (total_M_t * M_x, 2)
            # Determine which segment each time point belongs to
            segment_indices = jnp.digitize(tx_points_full[:, 0], seq_T) - 1
            segment_indices = jnp.clip(segment_indices, 0, num_segments - 1)
            # Collect outputs from each segment
            u_values_full_list = []
            for i in range(num_segments):
                mask = (segment_indices == i)
                tx_points_segment = tx_points_full[mask]
                if tx_points_segment.size > 0:
                    segment = prob.segments[i]
                    if params is None:
                        u_values_segment = segment.model.forward(tx_points_segment)
                    else:
                        u_values_segment = segment.model(params, tx_points_segment) if hasattr(segment.model, '__call__') else segment.model.forward(tx_points_segment)
                    u_values_full_list.append(u_values_segment)
            # Combine all segment outputs into a single array
            u_values_full = jnp.concatenate(u_values_full_list).reshape(len(t_grid_full), M_x)
            t_grid = t_grid_full
            u_values = u_values_full
            # For 1D at t_max: use the last segment
            last_segment = prob.segments[-1]
            t_jnp = jnp.full_like(x_grid, t_max)
            tx_points_max = jnp.stack([t_jnp, x_grid], axis=1)
            if params is None:
                u_at_max = last_segment.model.forward(tx_points_max)
            else:
                u_at_max = last_segment.model(params, tx_points_max) if hasattr(last_segment.model, '__call__') else last_segment.model.forward(tx_points_max)
        else:
            # Single segment case
            t_grid = jnp.linspace(t_min, t_max, M_t)
            t, x = jnp.meshgrid(t_grid, x_grid, indexing='ij') # Shapes: (M_t, M_x)
            # Flatten the grid for batch evaluation
            tx_points = jnp.stack([t.ravel(), x.ravel()], axis=1) # Shape: (M_t * M_x, 2)
            # Evaluate u_theta on the grid points
            if params is None:
                u_values = prob.model.forward(tx_points)
            else:
                u_values = prob.model(params, tx_points) if hasattr(prob.model, '__call__') else prob.model.forward(tx_points)
            u_values = u_values.reshape(len(t_grid), M_x) # Reshape to (M_t, M_x)
            u_at_max = u_values[-1, :]
        # Calculate t_range for figure sizes
        t_range = t_max - t_min
        # First figure: 2D color mesh plot
        fig1 = plt.figure(figsize=(10, 5 * t_range if t_range > 0 else 4))
        ax1 = fig1.add_subplot(111)
        pcm = ax1.pcolormesh(x_grid, t_grid, u_values, shading='nearest', cmap='viridis')
        pcm.set_rasterized(True) # Rasterize the mesh to reduce EPS size
        cbar = fig1.colorbar(pcm, ax=ax1)
        # cbar.set_label('$u_\\theta(t, x)$', fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        ax1.set_xlabel('x', fontsize=fontsize)
        ax1.set_ylabel('t', fontsize=fontsize)
        ax1.tick_params(labelsize=fontsize)
        # ax1.set_title('Neural Network Function $u_\\theta$' if not is_sequential else f'Combined Neural Network $u_\\theta$ over [{t_min}, {t_max}]', fontsize=fontsize)
        ax1.set_yticks(np.linspace(t_min, t_max, int((t_max-t_min)//0.25)+1))
        plt.tight_layout()
        # Upload and save the 2D figure
        if upload_fig:
            wandb.log({f"u_theta_2d_tmin{t_min:.1f}_tmax_{t_max:.1f}": wandb.Image(fig1)})
        if save_path:
            base, ext = splitext(save_path)
            plt.savefig(save_path, dpi=100)
        plt.show()
        # Second figure: 1D plot at t = t_max
        fig2 = plt.figure(figsize=(10, 4))
        ax2 = fig2.add_subplot(111)
        ax2.plot(x_grid, u_at_max)
        ax2.set_xlabel('x', fontsize=fontsize)
        # ax2.set_ylabel('$u_\\theta(t_{max}, x)$', fontsize=fontsize)
        ax2.tick_params(labelsize=fontsize)
        ax2.set_title(f'Neural Network Function at t = {t_max:.2f}', fontsize=fontsize)
        plt.tight_layout()
        # Upload and save the 1D figure
        if upload_fig:
            wandb.log({f"u_theta_1d_tmax_{t_max:.1f}": wandb.Image(fig2)})
        # if save_path:
        # base, ext = splitext(save_path)
        # plt.savefig(f"{base}_t{t_max:.2f}{ext}")
        plt.show()
    # Plot separate figures for each timestamp in ts
    for t_val in ts:
        # Find which segment t_val belongs to
        seg_idx = jnp.digitize(t_val, time_bins) - 1
        seg_idx = int(jnp.clip(seg_idx, 0, num_segments - 1))
        if is_sequential:
            segment = prob.segments[seg_idx]
        else:
            segment = prob
        # Build points
        if is_3d:
            x_mesh, y_mesh = jnp.meshgrid(x_grid, y_grid, indexing='ij')
            x_flat = x_mesh.ravel()
            y_flat = y_mesh.ravel()
            t_flat = jnp.full_like(x_flat, t_val)
            points_t = jnp.stack([t_flat, x_flat, y_flat], axis=1)
        else:
            t_jnp = jnp.full_like(x_grid, t_val)
            points_t = jnp.stack([t_jnp, x_grid], axis=1)
        # Evaluate
        if params is None:
            u_at_t = segment.model.forward(points_t)
        else:
            u_at_t = segment.model(params, points_t) if hasattr(segment.model, '__call__') else segment.model.forward(points_t)
        u_at_t = jnp.asarray(u_at_t).squeeze()
        # Reshape if 3D
        if is_3d:
            u_at_t = u_at_t.reshape(x_mesh.shape)
        # Create figure
        if is_3d:
            x_range = x_max - x_min
            y_range = y_max - y_min
            fig_width = 5 * x_range + 1.5 
            fig_height = 5 * y_range
            has_ref = (u_ref is not None) or (file_path is not None)
            if has_ref:
                fig, axs = plt.subplots(1, 2, figsize=(2 * fig_width, fig_height))
                ax_nn = axs[0]
                ax_ref = axs[1]
            else:
                fig, ax_nn = plt.subplots(1, 1, figsize=(fig_width, fig_height))
                ax_ref = None
            # Plot NN
            u_at_t_np = np.array(u_at_t)
            if has_ref:
                if u_ref is not None:
                    xx, yy = np.meshgrid(np.array(x_grid), np.array(y_grid), indexing='ij')
                    u_ref_at_t = u_ref(t_val*jnp.ones_like(xx), xx, yy)
                else:
                    idx = np.argmin(np.abs(times - t_val))
                    u_ref_at_t = u_num[idx]
                min_u = min(float(np.min(u_at_t_np)), float(np.min(u_ref_at_t)))
                max_u = max(float(np.max(u_at_t_np)), float(np.max(u_ref_at_t)))
                pcm_nn = ax_nn.pcolormesh(x_grid, y_grid, u_at_t_np.T, shading='nearest', cmap='viridis', vmin=min_u, vmax=max_u)
                # Plot ref
                pcm_ref = ax_ref.pcolormesh(x_grid, y_grid, u_ref_at_t.T, shading='nearest', cmap='viridis', vmin=min_u, vmax=max_u)
                pcm_ref.set_rasterized(True)
                cbar_ref = fig.colorbar(pcm_ref, ax=ax_ref)
                cbar_ref.ax.tick_params(labelsize=fontsize)
                ax_ref.set_xlabel('x', fontsize=fontsize)
                ax_ref.set_ylabel('y', fontsize=fontsize)
                ax_ref.tick_params(labelsize=fontsize)
                ax_ref.set_title(f'Ref at t = {t_val:.2f}', fontsize=fontsize)
            else:
                pcm_nn = ax_nn.pcolormesh(x_grid, y_grid, u_at_t_np.T, shading='nearest', cmap='viridis')
            pcm_nn.set_rasterized(True)
            cbar_nn = fig.colorbar(pcm_nn, ax=ax_nn)
            cbar_nn.ax.tick_params(labelsize=fontsize)
            ax_nn.set_xlabel('x', fontsize=fontsize)
            ax_nn.set_ylabel('y', fontsize=fontsize)
            ax_nn.tick_params(labelsize=fontsize)
            ax_nn.set_title(f'NN at t = {t_val:.2f}', fontsize=fontsize)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.plot(x_grid, u_at_t)
            ax.set_xlabel('x', fontsize=fontsize)
            ax.set_ylabel('$u_\\theta(t, x)$', fontsize=fontsize)
            ax.tick_params(labelsize=fontsize)
            ax.set_title(f'Neural Network Function at t = {t_val:.2f}', fontsize=fontsize)
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
    kh_values = np.asarray(kh_values)

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
    fig.colorbar(pcm, ax=ax1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_title('Numerical sol. $u$')

    # 1D plot at t = max(times)
    ax2.plot(x, u[-1, :])
    ax2.set_xlabel('x')
    ax2.set_ylabel('$u(t_{max}, x)$')
    ax2.set_title(f'Numerical sol. at t = {times[-1]:.2f}')

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
        ax.set_title(f'Numerical sol. at t = {t_val:.2f}')
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
        x = np.linspace(seq_model.x_min, seq_model.x_max, M_x)
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
    print(f"Error at T: {num_T}, Relative error at T: {err_T}")
    print(f"Space-time error: {spacetime_num}, Space-time relative error: {err_spacetime}")

    return err_T, err_spacetime

def plot_comparison_together(prob, u_ref=None, file_path='data/burgers_sin_ref.npz', 
                             params=None, ts=None, save_path='data/sine_wave_plot_together.eps', title=None, x_min=-1.0, x_max=1.0, M_x=601, tol=1e-6, fontsize=16):
    """
    Plot both the neural network solution (solid line) and the reference solution (dashed line)
    for each timestamp in ts, with all comparisons overlaid on the same single figure. For each
    time, the pair of lines (numerical and neural) share the same color, progressing from light
    to dark as time increases.

    Args:
        prob: Instance of MinMaxOptProb or SeqMinMaxOptProb representing the neural network(s).
        u_ref: Reference function u_ref(t, x) where t is scalar and x is array (default: None).
        file_path: Path to the .npz file containing 'times', 'x', and 'u' (default: 'data/burgers_sin_ref.npz').
        params: Parameters (pytree) of the neural network (default: None).
        ts: List or array of timestamps for which to plot the solutions (default: None).
        save_path: Path to save the .eps file (default: 'data/sine_wave_plot_together.eps').
        title: Title of the plot (default: None).
        x_min: Minimum x value (default: -1.0).
        x_max: Maximum x value (default: 1.0).
        M_x: Number of x grid points (default: 601).
        tol: Tolerance for floating point comparison in sequential case (default: 1e-6).
        fontsize: Font size for axis labels, ticks, title, and legend (default: 12).
    """
    if ts is None:
        ts = []
    if not ts:
        print("No timestamps provided. Nothing to plot.")
        return

    # Determine if it's a sequential problem
    is_sequential = hasattr(prob, 'num_segments') and hasattr(prob, 'segments') and hasattr(prob, 'seq_T')

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

    # Get colormap (distinct colors)
    cmap = plt.cm.tab10
    norms = np.linspace(0, 1, len(ts))

    # List of markers for reference lines
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd']

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

        if is_sequential:
            # Find the segment that contains time t_val
            segment_index = None
            for j in range(len(prob.seq_T) - 1):
                if prob.seq_T[j] <= t_val <= prob.seq_T[j + 1] + tol:  # Allow a small tolerance for floating point errors
                    segment_index = j
                    break
            if segment_index is None:
                if jnp.abs(t_val - prob.seq_T[-1]) < tol:  # Allow t_val to be exactly at the end
                    segment_index = len(prob.seq_T) - 2
                else:
                    raise ValueError(f"Time t_val={t_val} is outside the defined segments [{prob.seq_T[0]}, {prob.seq_T[-1]}]")
            # Get the segment
            segment = prob.segments[segment_index]
            u_nn_values = segment.model.forward(tx_points)
        else:
            if params is None:
                u_nn_values = prob.model.forward(tx_points)
            else:
                u_nn_values = prob.model(params, tx_points)

        u_nn_t = np.asarray(u_nn_values).squeeze()  # Shape: (n_x,)

        # Plot both on the same axes with same color
        ax.plot(x, u_ref_t, linestyle='--', color=color, linewidth=2, marker=markers[i % len(markers)], markevery=30, label=f'Ref. sol. t={t_val:.2f}')
        ax.plot(x, u_nn_t, linestyle='-', color=color, linewidth=2, label=f'NN sol. t={t_val:.2f}')

    # Set font sizes
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_xlabel('x', fontsize=fontsize)
    ax.set_ylabel('u', fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, format='eps', bbox_inches='tight')
    plt.show()


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
    fig, axs = plt.subplots(len(ts), 1, figsize=(10, 5 * len(ts)), squeeze=False)
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
        ax.plot(x, u_ref_t, label='Ref. sol.', linestyle='--', linewidth=2)
        ax.plot(x, u_nn_t, label='NN sol.', linewidth=2)
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title(f'Solutions at t = {t_val:.2f}')

    plt.tight_layout()
    plt.show()

def plot_comparison_seperate_combined_u_theta(seq_model, u_ref=None, file_path='data/burgers_sin_ref.npz', ts=None, x_min=-1.0, x_max=1.0, M_x=601, save_eps=False, tol=1e-6):
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
        save_eps: Whether to save each figure as .eps file (default: False).
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
        x = np.linspace(seq_model.x_min, seq_model.x_max, M_x)
        times = np.array(ts)
        u_num = np.array([u_ref(t, x) for t in times])  # Shape: (len(ts), n_x)

    for i, t_val in enumerate(ts):
        # Create a new figure for each timestamp
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        # Reference solution: find closest time index or use exact t_val
        idx = np.argmin(np.abs(times - t_val)) if u_ref is None else i
        u_ref_t = u_num[idx, :]

        # Find the segment that contains time t_val
        segment_index = None
        for j in range(len(seq_model.seq_T) - 1):
            if seq_model.seq_T[j] <= t_val <= seq_model.seq_T[j + 1] + tol:  # Allow a small tolerance for floating point errors
                segment_index = j
                break
        if segment_index is None:
            if jnp.abs(t_val - seq_model.seq_T[-1]) < tol:  # Allow t_val to be exactly at the end
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
        ax.plot(x, u_ref_t, label='Ref. sol.', linestyle='--', linewidth=2)
        ax.plot(x, u_nn_t, label='NN sol.', linewidth=2)
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title(f'Solutions at t = {t_val:.2f}')

        plt.tight_layout()
        if save_eps:
            plt.savefig(f"plot_t_{t_val:.2f}.eps", format='eps', bbox_inches='tight')
        plt.show()

# def plot_u_theta_mesh(model, params=None, M_t=1000, M_x=2000, t_min=0.0, t_max=0.5, x_min=-1.0, x_max=1.0, ts=None, upload_fig=False, pwp=None, vals=None):
#     """
#     Plot the neural network function u_theta over a specified domain [t_min, t_max] x [x_min, x_max].
#     Optionally overlay the mesh from PiecewiseLinearPolynomials and denote vals on the meshes.
#     The 1D plot at t = t_max is removed, and only the 2D plot is shown.

#     Args:
#         model: Neural network u_theta.
#         params: Parameters (pytree) of the neural network.
#         M_t: Number of grid points in the t-direction (default: 1000).
#         M_x: Number of grid points in the x-direction (default: 2000).
#         t_min: Minimum value for t (default: 0.0).
#         t_max: Maximum value for t (default: 0.5).
#         x_min: Minimum value for x (default: -1.0).
#         x_max: Maximum value for x (default: 1.0).
#         ts: List or array of timestamps for which to plot individual 1D snapshots (default: None).
#         upload_fig: Whether to upload the figure to wandb (default: False).
#         pwp: Instance of PiecewiseLinearPolynomials(Tri/Rect) to plot the mesh (default: None).
#         vals: Values to denote on each mesh (default: None).
#     """
#     if ts is None:
#         ts = []

#     # Create a grid for t and x over the specified ranges
#     t_grid = np.linspace(t_min, t_max, M_t)
#     x_grid = np.linspace(x_min, x_max, M_x)
#     t, x = np.meshgrid(t_grid, x_grid, indexing='ij')  # Shapes: (M_t, M_x)

#     # Flatten the grid for batch evaluation
#     tx_points = np.stack([t.ravel(), x.ravel()], axis=1)  # Shape: (M_t * M_x, 2)

#     # Evaluate u_theta on the grid points
#     if params is None:
#         u_values = model.forward(tx_points)
#     else:
#         u_values = model.forward_with_params(params, tx_points) if hasattr(model, 'forward_with_params') else model(params, tx_points)
#     u_values = u_values.reshape(M_t, M_x)  # Reshape to (M_t, M_x)

#     # Create larger figure for the 2D plot
#     t_range = t_max - t_min
#     fig = plt.figure(figsize=(20, 8 + 8 * t_range))
#     ax1 = fig.add_subplot(111)

#     # 2D color mesh plot
#     pcm = ax1.pcolormesh(x_grid, t_grid, u_values, shading='nearest', cmap='viridis')
#     fig.colorbar(pcm, ax=ax1)
#     ax1.set_xlabel('x')
#     ax1.set_ylabel('t')
#     ax1.set_title('Neural Network Function $u_\\theta$')

#     # Overlay the mesh and denote vals if pwp and vals are provided
#     if pwp is not None:
#         if isinstance(pwp, PiecewiseLinearPolynomialsTri):
#             triangles = pwp.partitions
#             for idx, tri in enumerate(triangles):
#                 # Plot the three edges of the triangle
#                 ax1.plot([tri[0][1], tri[1][1]], [tri[0][0], tri[1][0]], color='black', alpha=0.5, linewidth=0.5)
#                 ax1.plot([tri[1][1], tri[2][1]], [tri[1][0], tri[2][0]], color='black', alpha=0.5, linewidth=0.5)
#                 ax1.plot([tri[2][1], tri[0][1]], [tri[2][0], tri[0][0]], color='black', alpha=0.5, linewidth=0.5)
#                 center = np.mean(tri, axis=0)
#                 if vals is not None:
#                     # Denote avg_val at the center
#                     ax1.text(center[1], center[0], f'{vals[idx]:.2f}', fontsize=6, color='black', ha='center', va='center')

#         elif isinstance(pwp, PiecewiseLinearPolynomialsRect):
#             rectangles = pwp.partitions
#             for idx, rect in enumerate(rectangles):
#                 # Plot the four edges of the rectangle
#                 ax1.plot([rect[0][1], rect[1][1]], [rect[0][0], rect[1][0]], color='black', alpha=0.5, linewidth=0.5)  # Bottom
#                 ax1.plot([rect[0][1], rect[2][1]], [rect[0][0], rect[2][0]], color='black', alpha=0.5, linewidth=0.5)  # Left
#                 ax1.plot([rect[2][1], rect[3][1]], [rect[2][0], rect[3][0]], color='black', alpha=0.5, linewidth=0.5)  # Top
#                 ax1.plot([rect[3][1], rect[1][1]], [rect[3][0], rect[1][0]], color='black', alpha=0.5, linewidth=0.5)  # Right
#                 center = np.mean(rect, axis=0)
#                 if vals is not None:
#                     # Denote avg_val at the center
#                     ax1.text(center[1], center[0], f'{vals[idx]:.2f}', fontsize=6, color='black', ha='center', va='center')

#     # Adjust layout and display the main plot
#     plt.tight_layout()
#     # Upload the main figure to wandb
#     if upload_fig:
#         wandb.log({f"u_theta_tmin{t_min:.1f}_tmax_{t_max:.1f}": wandb.Image(fig)})
#     plt.show()

#     # Plot separate figures for each timestamp in ts
#     for t_val in ts:
#         # Find the closest index in t_grid
#         idx = np.argmin(np.abs(t_grid - t_val))
#         # Create a new figure for the snapshot
#         fig, ax = plt.subplots(1, 1, figsize=(10, 4))
#         ax.plot(x_grid, u_values[idx, :])
#         ax.set_xlabel('x')
#         ax.set_ylabel('$u_\\theta(t, x)$')
#         ax.set_title(f'Neural Network Function at t = {t_val:.2f} (closest t = {t_grid[idx]:.2f})')
#         plt.tight_layout()
#         plt.show()

def plot_u_theta_mesh(model, params=None, M_t=1000, M_x=2000, t_min=0.0, t_max=0.5, x_min=-1.0, x_max=1.0, y_min=None, y_max=None, M_y=None, ts=None, upload_fig=False, pwp=None, vals=None, u_ref=None):
    """
    Plot the neural network function u_theta over a specified domain [t_min, t_max] x [x_min, x_max].
    Optionally overlay the mesh from PiecewiseLinearPolynomials and denote vals on the meshes.
    The 1D plot at t = t_max is removed, and only the 2D plot is shown.

    Args:
        model: Neural network u_theta.
        params: Parameters (pytree) of the neural network.
        M_t: Number of grid points in the t-direction (default: 1000).
        M_x: Number of grid points in the x-direction (default: 2000).
        t_min: Minimum value for t (default: 0.0).
        t_max: Maximum value for t (default: 0.5).
        x_min: Minimum value for x (default: -1.0).
        x_max: Maximum value for x (default: 1.0).
        y_min: Minimum value for y (for 3D cases, default: None).
        y_max: Maximum value for y (for 3D cases, default: None).
        M_y: Number of grid points in the y-direction (for 3D cases, default: None, falls back to M_x).
        ts: List or array of timestamps for which to plot individual 1D snapshots (default: None).
        upload_fig: Whether to upload the figure to wandb (default: False).
        pwp: Instance of PiecewiseLinearPolynomials(Tri/Rect/Hex/Tetra) to plot the mesh (default: None).
        vals: Values to denote on each mesh (default: None).
        u_ref: Callable reference solution function u_ref(t, x, y) (default: None). If provided for input_dim=3, plots it side-by-side with u_theta at t=t_max, including mesh overlay.
    """
    if ts is None:
        ts = []

    input_dim = model.input_dim if hasattr(model, 'input_dim') else 2
    if M_y is None:
        M_y = M_x

    if input_dim == 2:
        # Create a grid for t and x over the specified ranges
        t_grid = np.linspace(t_min, t_max, M_t)
        x_grid = np.linspace(x_min, x_max, M_x)
        t, x = np.meshgrid(t_grid, x_grid, indexing='ij')  # Shapes: (M_t, M_x)

        # Flatten the grid for batch evaluation
        tx_points = np.stack([t.ravel(), x.ravel()], axis=1)  # Shape: (M_t * M_x, 2)

        # Evaluate u_theta on the grid points
        if params is None:
            u_values = model.forward(tx_points)
        else:
            u_values = model.forward_with_params(params, tx_points) if hasattr(model, 'forward_with_params') else model(params, tx_points)
        u_values = u_values.reshape(M_t, M_x)  # Reshape to (M_t, M_x)

        # Create larger figure for the 2D plot
        t_range = t_max - t_min
        fig = plt.figure(figsize=(20, 8 + 8 * t_range))
        ax1 = fig.add_subplot(111)

        # 2D color mesh plot
        pcm = ax1.pcolormesh(x_grid, t_grid, u_values, shading='nearest', cmap='viridis')
        fig.colorbar(pcm, ax=ax1)
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        ax1.set_title('Neural Network Function $u_\\theta$')

        # Overlay the mesh and denote vals if pwp and vals are provided
        if pwp is not None:
            if isinstance(pwp, PiecewiseLinearPolynomialsTri):
                triangles = pwp.partitions
                for idx, tri in enumerate(triangles):
                    # Plot the three edges of the triangle
                    ax1.plot([tri[0][1], tri[1][1]], [tri[0][0], tri[1][0]], color='black', alpha=0.5, linewidth=0.5)
                    ax1.plot([tri[1][1], tri[2][1]], [tri[1][0], tri[2][0]], color='black', alpha=0.5, linewidth=0.5)
                    ax1.plot([tri[2][1], tri[0][1]], [tri[2][0], tri[0][0]], color='black', alpha=0.5, linewidth=0.5)
                    center = np.mean(tri, axis=0)
                    if vals is not None:
                        # Denote avg_val at the center
                        # ax1.text(center[1], center[0], f'{vals[idx]:.1e}', fontsize=6, color='black', ha='center', va='center')
                        ax1.text(center[1], center[0], f'{vals[idx]: .0e}', fontsize=4, color='black', ha='center', va='center', clip_on=True, bbox=dict(facecolor='white', alpha=0.5, pad=1))

            elif isinstance(pwp, PiecewiseLinearPolynomialsRect):
                rectangles = pwp.partitions
                for idx, rect in enumerate(rectangles):
                    # Plot the four edges of the rectangle
                    ax1.plot([rect[0][1], rect[1][1]], [rect[0][0], rect[1][0]], color='black', alpha=0.5, linewidth=0.5)  # Bottom
                    ax1.plot([rect[0][1], rect[2][1]], [rect[0][0], rect[2][0]], color='black', alpha=0.5, linewidth=0.5)  # Left
                    ax1.plot([rect[2][1], rect[3][1]], [rect[2][0], rect[3][0]], color='black', alpha=0.5, linewidth=0.5)  # Top
                    ax1.plot([rect[3][1], rect[1][1]], [rect[3][0], rect[1][0]], color='black', alpha=0.5, linewidth=0.5)  # Right
                    center = np.mean(rect, axis=0)
                    if vals is not None:
                        # Denote avg_val at the center
                        # ax1.text(center[1], center[0], f'{vals[idx]:.1e}', fontsize=6, color='black', ha='center', va='center')
                        ax1.text(center[1], center[0], f'{vals[idx]: .0e}', fontsize=4, color='black', ha='center', va='center', clip_on=True, bbox=dict(facecolor='white', alpha=0.5, pad=1))

        # Adjust layout and display the main plot
        plt.tight_layout()
        # Upload the main figure to wandb
        if upload_fig:
            wandb.log({f"u_theta_tmin{t_min:.1f}_tmax_{t_max:.1f}": wandb.Image(fig)})
        plt.show()

    elif input_dim == 3:
        if y_min is None or y_max is None:
            raise ValueError("For input_dim=3, y_min and y_max must be provided.")

        # 2D plot at t = t_max over (x, y)
        x_grid = np.linspace(x_min, x_max, M_x)
        y_grid = np.linspace(y_min, y_max, M_y)
        x, y = np.meshgrid(x_grid, y_grid)  # Shapes: (M_x, M_y)
        
        t = t_max * np.ones_like(x)
        txy_points = np.stack([t.ravel(), x.ravel(), y.ravel()], axis=1)  # Shape: (M_x * M_y, 3)
        
        # Evaluate u_theta on the grid points
        if params is None:
            u_values = model.forward(txy_points)
        else:
            u_values = model.forward_with_params(params, txy_points) if hasattr(model, 'forward_with_params') else model(params, txy_points)
        u_values = u_values.reshape(M_y, M_x)  # Reshape to (M_x, M_y)
        
        # Determine figure size based on whether u_ref is provided
        fig_width = 20 if u_ref is not None else 10
        fig = plt.figure(figsize=(fig_width, 8))
        
        # Plot u_theta
        ax1 = fig.add_subplot(121) if u_ref is not None else fig.add_subplot(111)
        
        # 2D color mesh plot for u_theta
        pcm1 = ax1.pcolormesh(x_grid, y_grid, u_values, shading='nearest', cmap='viridis')
        fig.colorbar(pcm1, ax=ax1)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'Neural Network Function $u_\\theta$ at t = {t_max:.2f}')
        
        # Overlay the mesh and denote vals for elements sharing a face with t = t_max
        
        if pwp is not None:
            partitions = pwp.partitions
            # Find elements that share a face with t = t_max (i.e., max t == t_max)
            max_t_per_elem = np.max(partitions[:, :, 0], axis=1)
            surface_mask = np.isclose(max_t_per_elem, t_max, atol=1e-6)
            surface_elems = partitions[surface_mask]
            surface_indices = np.where(surface_mask)[0]
            
            if isinstance(pwp, PiecewiseLinearPolynomialsHex):
                
                for i, hex_elem in enumerate(surface_elems):
                    # Find vertices at t = t_max (top face)
                    top_mask = np.isclose(hex_elem[:, 0], t_max, atol=1e-6)
                    top_verts = hex_elem[top_mask]
                    if len(top_verts) == 4:  # Expected for hex top face
                        x_top = top_verts[:, 1]
                        y_top = top_verts[:, 2]
                        # Order for closed rectangle: assuming vertices [lowx lowy, lowx highy, highx lowy, highx highy]
                        # Indices based on binary order: 0: lowx lowy, 1: lowx highy, 2: highx lowy, 3: highx highy
                        indices = np.array([0, 1, 3, 2])
                        ax1.plot(np.append(x_top[indices], x_top[indices[0]]), np.append(y_top[indices], y_top[indices[0]]), 
                                 color='black', alpha=0.5, linewidth=0.5)
                        
                        center_x = np.mean(x_top)
                        center_y = np.mean(y_top)
                        if vals is not None:
                            # ax1.text(center_x, center_y, f'{vals[surface_indices[i]]:.1e}', fontsize=6, color='black', ha='center', va='center')
                            ax1.text(center_x, center_y, f'{vals[surface_indices[i]]: .0e}', fontsize=5, color='black', ha='center', va='center', clip_on=True, bbox=dict(facecolor='white', alpha=0.5, pad=1))
            
            elif isinstance(pwp, PiecewiseLinearPolynomialsTetra):
                for i, tet in enumerate(surface_elems):
                    # Find vertices at t = t_max
                    top_mask = np.isclose(tet[:, 0], t_max, atol=1e-6)
                    top_verts = tet[top_mask]
                    if len(top_verts) == 3:  # Triangle face
                        x_top = top_verts[:, 1]
                        y_top = top_verts[:, 2]
                        ax1.plot(np.append(x_top, x_top[0]), np.append(y_top, y_top[0]), 
                                 color='black', alpha=0.5, linewidth=0.5)
                        
                        center_x = np.mean(x_top)
                        center_y = np.mean(y_top)
                        if vals is not None:
                            # ax1.text(center_x, center_y, f'{vals[surface_indices[i]]:.1e}', fontsize=6, color='black', ha='center', va='center')
                            ax1.text(center_x, center_y, f'{vals[surface_indices[i]]: .0e}', fontsize=5, color='black', ha='center', va='center', clip_on=True, bbox=dict(facecolor='white', alpha=0.5, pad=1))
            else:
                print("No such class!!!")
        
        # If u_ref is provided, plot it side-by-side
        if u_ref is not None:
            # Evaluate u_ref on the same grid points (assuming u_ref takes (t, x, y))
            u_ref_values = u_ref(txy_points[:, 0], txy_points[:, 1], txy_points[:, 2])
            u_ref_values = u_ref_values.reshape(M_y, M_x)
            
            ax2 = fig.add_subplot(122)
            
            # 2D color mesh plot for u_ref
            pcm2 = ax2.pcolormesh(x_grid, y_grid, u_ref_values, shading='nearest', cmap='viridis')
            fig.colorbar(pcm2, ax=ax2)
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_title(f'Reference Solution $u_ref$ at t = {t_max:.2f}')
            
            # Overlay the same mesh on u_ref plot
            if pwp is not None:
                if isinstance(pwp, PiecewiseLinearPolynomialsHex):
                    
                    for i, hex_elem in enumerate(surface_elems):
                        # Find vertices at t = t_max (top face)
                        top_mask = np.isclose(hex_elem[:, 0], t_max, atol=1e-6)
                        top_verts = hex_elem[top_mask]
                        if len(top_verts) == 4:  # Expected for hex top face
                            x_top = top_verts[:, 1]
                            y_top = top_verts[:, 2]
                            # Order for closed rectangle: assuming vertices [lowx lowy, lowx highy, highx lowy, highx highy]
                            # Indices based on binary order: 0: lowx lowy, 1: lowx highy, 2: highx lowy, 3: highx highy
                            indices = np.array([0, 1, 3, 2])
                            ax2.plot(np.append(x_top[indices], x_top[indices[0]]), np.append(y_top[indices], y_top[indices[0]]), 
                                     color='black', alpha=0.5, linewidth=0.5)
                            
                            center_x = np.mean(x_top)
                            center_y = np.mean(y_top)
                            # if vals is not None:
                            #     ax2.text(center_x, center_y, f'{vals[surface_indices[i]]:.1e}', fontsize=6, color='black', ha='center', va='center')
                
                elif isinstance(pwp, PiecewiseLinearPolynomialsTetra):
                    for i, tet in enumerate(surface_elems):
                        # Find vertices at t = t_max
                        top_mask = np.isclose(tet[:, 0], t_max, atol=1e-6)
                        top_verts = tet[top_mask]
                        if len(top_verts) == 3:  # Triangle face
                            x_top = top_verts[:, 1]
                            y_top = top_verts[:, 2]
                            ax2.plot(np.append(x_top, x_top[0]), np.append(y_top, y_top[0]), 
                                     color='black', alpha=0.5, linewidth=0.5)
                            
                            center_x = np.mean(x_top)
                            center_y = np.mean(y_top)
                            if vals is not None:
                                ax2.text(center_x, center_y, f'{vals[surface_indices[i]]:.1e}', fontsize=6, color='black', ha='center', va='center')
                else:
                    print("No such class!!!")

        plt.tight_layout()
        # Upload the figure to wandb
        if upload_fig:
            wandb.log({f"u_theta_at_tmax_{t_max:.1f}": wandb.Image(fig)})
        plt.show()

    # Plot separate figures for each timestamp in ts (adapted for dimension)
    for t_val in ts:
        if input_dim == 2:
            # Find the closest index in t_grid
            idx = np.argmin(np.abs(t_grid - t_val))
            # Create a new figure for the snapshot
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.plot(x_grid, u_values[idx, :])
            ax.set_xlabel('x')
            ax.set_ylabel('$u_\\theta(t, x)$')
            ax.set_title(f'Neural Network Function at t = {t_val:.2f} (closest t = {t_grid[idx]:.2f})')
            plt.tight_layout()
            plt.show()
        elif input_dim == 3:
            # Similar to t_max plot but without mesh/vals (or adapt if needed)
            x, y = np.meshgrid(x_grid, y_grid)
            t = t_val * np.ones_like(x)
            txy_points = np.stack([t.ravel(), x.ravel(), y.ravel()], axis=1)
            u_values_ts = model.forward(txy_points).reshape(M_y, M_x)
            fig_ts, ax_ts = plt.subplots(1, 1, figsize=(10, 8))
            pcm_ts = ax_ts.pcolormesh(x_grid, y_grid, u_values_ts, shading='nearest', cmap='viridis')
            fig_ts.colorbar(pcm_ts, ax=ax_ts)
            ax_ts.set_xlabel('x')
            ax_ts.set_ylabel('y')
            ax_ts.set_title(f'Neural Network Function $u_\\theta$ at t = {t_val:.2f}')
            plt.tight_layout()
            plt.show()

def plot_u_2D(ts, u_given, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=1000, ny=1000):
    """
    Plot the function u_given at given times ts over the domain [x_min, x_max] x [y_min, y_max].

    Args:
        ts (list or np.array): List of times at which to plot the solution.
        x_min (float): Minimum x value (default: 0.0).
        x_max (float): Maximum x value (default: 1.0).
        y_min (float): Minimum y value (default: 0.0).
        y_max (float): Maximum y value (default: 1.0).
        nx (int): Number of grid points in x-direction (default: 1000).
        ny (int): Number of grid points in y-direction (default: 1000).
    """
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    num_times = len(ts)
    nrows = int(np.ceil(np.sqrt(num_times)))
    ncols = int(np.ceil(num_times / nrows))
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols + 1.5, 5 * nrows), squeeze=False)
    
    for i, t in enumerate(ts):
        row = i // ncols
        col = i % ncols
        ax = axs[row, col]
        
        u = u_given(t * np.ones_like(X), X, Y)
        
        pcm = ax.contourf(X, Y, u, levels=50, cmap='viridis')
        fig.colorbar(pcm, ax=ax, label='u(t, x, y)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Exact Solution at t = {t:.2f}')
    
    # Hide unused subplots
    for i in range(num_times, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axs[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()