import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import linen as nn
from NN_class import MLP

# Function to plot u_theta
def plot_u_theta(model, params, M_t=100, M_x=100, t_min=0.0, t_max=1.0, x_min=0.0, x_max=1.0):
    """
    Plot the neural network function u_theta over a specified domain [t_min, t_max] x [x_min, x_max].

    Args:
        model: Flax model instance representing u_theta.
        params: Parameters (pytree) of the neural network.
        M_t: Number of grid points in the t-direction (default: 100).
        M_x: Number of grid points in the x-direction (default: 100).
        t_min: Minimum value for t (default: 0.0).
        t_max: Maximum value for t (default: 1.0).
        x_min: Minimum value for x (default: 0.0).
        x_max: Maximum value for x (default: 1.0).
    """
    # Create a grid for t and x over the specified ranges
    t_grid = jnp.linspace(t_min, t_max, M_t)
    x_grid = jnp.linspace(x_min, x_max, M_x)
    t, x = jnp.meshgrid(t_grid, x_grid, indexing='ij')  # Shapes: (M_t, M_x)

    # Flatten the grid for batch evaluation
    tx_points = jnp.stack([t.ravel(), x.ravel()], axis=1)  # Shape: (M_t * M_x, 2)

    # Evaluate u_theta on the grid points
    u_values = model.apply({'params': params}, tx_points)  # Shape: (M_t * M_x, 1)
    u_values = u_values.reshape(M_t, M_x)  # Reshape to (M_t, M_x)

    # Create the plot
    height = int((t_max - t_min)*4)
    width = int((x_max - x_min)*4 + 1)
    plt.figure(figsize=(width, height))
    plt.pcolormesh(x_grid, t_grid, u_values, shading='nearest', cmap='viridis')
    plt.colorbar(label='$u_\\theta(t, x)$')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Neural Network Function $u_\\theta$')
    plt.show()



def plot_kh(pwp, M_factor=10):
    """
    Plot the piecewise linear polynomial k_h over [0,1]^2 on a triangular mesh.

    Args:
        pwp: Object with N_t, N_x, and coeffs for the piecewise polynomial.
        M_factor: Factor to refine the plotting grid (default: 10).
    """
    N_t, N_x = pwp.N_t, pwp.N_x
    tau, hx = 1.0/N_t, 1.0/N_x  # Grid sizes

    # Create fine grid
    M_t, M_x = N_t * M_factor, N_x * M_factor
    t_grid = np.linspace(0, 1, M_t)
    x_grid = np.linspace(0, 1, M_x)
    t, x = np.meshgrid(t_grid, x_grid, indexing='ij')

    # Function to check if a point is inside a given triangle
    def point_in_triangle(P, A, B, C):
        def sign(p1, p2, p3):
            return (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0])
        d1 = sign(P, A, B)
        d2 = sign(P, B, C)
        d3 = sign(P, C, A)
        has_neg = d1 < 0 or d2 < 0 or d3 < 0
        has_pos = d1 > 0 or d2 > 0 or d3 > 0
        return not has_neg or not has_pos  # All same sign or zero

    # Assign each grid point to its corresponding triangle element
    element_indices = np.full(t.size, -1, dtype=int)
    for idx in range(t.size):
        tp, xp = t.ravel()[idx], x.ravel()[idx]
        i = np.floor(tp / tau).astype(int)
        j = np.floor(xp / hx).astype(int)
        i = np.clip(i, 0, N_t-1)
        j = np.clip(j, 0, N_x-1)

        # Get vertices of the two triangles in this rectangle
        A = [i*tau, j*hx]
        B = [(i+1)*tau, j*hx]
        C = [i*tau, (j+1)*hx]
        D = [(i+1)*tau, (j+1)*hx]

        # First triangle: A, B, C
        if point_in_triangle([tp, xp], A, B, C):
            global_idx = 2*(i*N_x + j)  # First triangle in rectangle (i,j)
            element_indices[idx] = global_idx
        # Second triangle: B, D, C
        elif point_in_triangle([tp, xp], B, D, C):
            global_idx = 2*(i*N_x + j) + 1  # Second triangle in rectangle (i,j)
            element_indices[idx] = global_idx
        else:
            print(f"Point not in any triangle: t={tp}, x={xp}")
            element_indices[idx] = -1  # Handle points not found (should be rare)

    # Reshape element_indices to match grid shape
    element_indices = element_indices.reshape(t.shape)

    # Evaluate k_h at grid points using their corresponding element indices
    tx_points = np.stack([t.ravel(), x.ravel()], axis=1)
    kh_values = pwp.evaluate(tx_points, np.ravel(element_indices))
    kh_values = kh_values.reshape(t.shape)

    # Plot using pcolormesh
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(x_grid, t_grid, kh_values, shading='nearest', cmap='viridis')
    plt.colorbar(label='$k_h(t, x)$')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Piecewise Linear Polynomial $k_h$ on Triangular Mesh')
    plt.show()