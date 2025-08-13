import jax.numpy as jnp
from jax import vmap
import numpy as np
from scipy import sparse

import flax.linen as nn
import optax.tree_utils as otu
from NN_class import MLP
# import mesh2D as mesh2D


class PiecewiseLinearPolynomialsRect:
    def __init__(self, N_t, N_x, t_min=0.0, t_max=1.0, x_min=-1.0, x_max=1.0, gauss_order=3, bdry_order=3):
        """
        Initialize piecewise linear polynomials on a rectangular mesh with custom ranges.

        Args:
            N_t: Number of divisions along the t-axis.
            N_x: Number of divisions along the x-axis.
            t_min: Minimum value for t (default: 0.0).
            t_max: Maximum value for t (default: 1.0).
            x_min: Minimum value for x (default: 0.0).
            x_max: Maximum value for x (default: 1.0).
            gauss_order: Order of Gauss quadrature (number of points per direction, default: 2).
            bdry_order: Order of boundary quadrature (number of points per edge, default: 3).
        """
        self.N_t = N_t
        self.N_x = N_x
        self.t_min = t_min
        self.t_max = t_max
        self.x_min = x_min
        self.x_max = x_max
        self.gauss_order = gauss_order
        self.bdry_order = bdry_order
        self.N_elements = N_t * N_x  # Number of rectangles
        self.partitions = self._generate_rectangles()
        self.coeffs = jnp.zeros((self.N_elements, 3))  # Coefficients a, b, c for a + b*t + c*x per rectangle
        self.quad_points, self.quad_weights, self.element_indices, self.quad_points_parti, self.quad_weights_parti = self._setup_quadrature()
        self.bdry_points_parti, self.bdry_weights_parti = self.get_boundary_quadrature()
        self.normals = self.compute_normals()

    def _generate_rectangles(self):
        """Generate the rectangles for the mesh with custom ranges."""
        tau = (self.t_max - self.t_min) / self.N_t  # Step size along t
        hx = (self.x_max - self.x_min) / self.N_x   # Step size along x
        rectangles = []
        for i in range(self.N_t):
            for j in range(self.N_x):
                rectangles.append(jnp.array([
                    [self.t_min + i * tau, self.x_min + j * hx],       # Bottom-left
                    [self.t_min + (i + 1) * tau, self.x_min + j * hx], # Top-left
                    [self.t_min + (i + 1) * tau, self.x_min + (j + 1) * hx], # Top-right
                    [self.t_min + i * tau, self.x_min + (j + 1) * hx]  # Bottom-right
                ]))
        return jnp.array(rectangles)

    def _setup_quadrature(self):
        """
        Setup Gauss quadrature points and weights for rectangular mesh.

        Returns:
            quad_points: Quadrature points, shape (N_quad, 2).
            quad_weights: Quadrature weights, shape (N_quad,).
            element_indices: Indices of rectangles, shape (N_quad,).
            quad_points_parti: Quadrature points per rectangle, shape (N_elements, Nq, 2).
            quad_weights_parti: Quadrature weights per rectangle, shape (N_elements, Nq).
        """
        Nq = self.gauss_order ** 2  # Number of quadrature points per rectangle

        # Gauss-Legendre nodes and weights in [-1, 1]
        nodes, weights = np.polynomial.legendre.leggauss(self.gauss_order)
        nodes_jnp = jnp.array(nodes)
        weights_jnp = jnp.array(weights)

        # Tensor product for 2D
        t_nodes, x_nodes = jnp.meshgrid(nodes_jnp, nodes_jnp)
        t_weights, x_weights = jnp.meshgrid(weights_jnp, weights_jnp)
        quad_weights_2d = (t_weights * x_weights).ravel()

        # Function to map [-1,1]x[-1,1] to rectangle [t_low, t_high]x[x_low, x_high]
        def map_to_rect(rect):
            t_low, x_low = rect[0]
            t_high, x_high = rect[2]
            t_scale = (t_high - t_low) / 2
            x_scale = (x_high - x_low) / 2
            t_center = (t_high + t_low) / 2
            x_center = (x_high + x_low) / 2
            t_points = t_center + t_nodes.ravel() * t_scale
            x_points = x_center + x_nodes.ravel() * x_scale
            points = jnp.stack([t_points, x_points], axis=1)
            weights = quad_weights_2d * (t_scale * x_scale)
            return points, weights

        # Vectorize over rectangles
        quad_points_rect, quad_weights_rect = vmap(map_to_rect)(self.partitions)

        # Flatten for global quadrature
        quad_points = quad_points_rect.reshape(-1, 2)
        quad_weights = quad_weights_rect.reshape(-1)
        element_indices = jnp.repeat(jnp.arange(self.N_elements), Nq)

        return quad_points, quad_weights, element_indices, quad_points_rect, quad_weights_rect

    def get_boundary_quadrature(self):
        """
        Generate Gauss-Legendre quadrature points and weights for the boundary of each rectangle.

        Returns:
            bdry_points_rect: Quadrature points for the boundaries, shape (N_elements, 4 * gauss_order, 2).
            bdry_weights_rect: Quadrature weights for the boundaries, shape (N_elements, 4 * gauss_order).
        """
        n = self.bdry_order  # Number of quadrature points per edge

        # Gauss-Legendre nodes and weights in [-1, 1]
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes_jnp = jnp.array(nodes)
        weights_jnp = jnp.array(weights)

        # Function to compute quadrature for a single edge [v1, v2]
        def edge_quadrature(v1, v2):
            L = jnp.linalg.norm(v2 - v1) / 2  # Half-length for scaling
            direction = (v2 - v1) / (2 * L)  # Unit direction scaled
            center = (v1 + v2) / 2
            points = center + nodes_jnp[:, None] * L * direction
            weights_scaled = weights_jnp * L
            return points, weights_scaled

        # Vectorize the edge quadrature function over multiple edges
        edge_quadrature_vmap = vmap(edge_quadrature, in_axes=(0, 0))

        # Compute quadrature for each set of edges for all rectangles
        points_bottom, weights_bottom = edge_quadrature_vmap(self.partitions[:, 0], self.partitions[:, 3])  # Bottom: 0-3
        points_top, weights_top = edge_quadrature_vmap(self.partitions[:, 1], self.partitions[:, 2])  # Top: 1-2
        points_left, weights_left = edge_quadrature_vmap(self.partitions[:, 0], self.partitions[:, 1])  # Left: 0-1
        points_right, weights_right = edge_quadrature_vmap(self.partitions[:, 3], self.partitions[:, 2])  # Right: 3-2

        # Concatenate points and weights for all four edges per rectangle
        bdry_points_rect = jnp.concatenate([points_bottom, points_top, points_left, points_right], axis=1)  # Shape: (N_elements, 4*n, 2)
        bdry_weights_rect = jnp.concatenate([weights_bottom, weights_top, weights_left, weights_right], axis=1)  # Shape: (N_elements, 4*n)

        return bdry_points_rect, bdry_weights_rect

    def compute_normals(self):
        """Compute outward unit normals for each edge of each rectangle."""
        # Edges order: bottom, top, left, right
        # Normals: bottom (0, -1), top (0, 1), left (-1, 0), right (1, 0)
        normals = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]])  # Fixed for all rectangles
        return jnp.tile(normals, (self.N_elements, 1, 1))  # Shape: (N_elements, 4, 2)