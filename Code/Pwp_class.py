import jax.numpy as jnp
from jax import vmap, pmap
import numpy as np
from scipy import sparse

import flax.linen as nn
import optax.tree_utils as otu
from NN_class import MLP
# import mesh2D as mesh2D


class PiecewiseLinearPolynomials:
    def __init__(self, N_t, N_x, t_min=0.0, t_max=1.0, x_min=0.0, x_max=1.0, times=1):
        """
        Initialize piecewise linear polynomials on a triangular mesh with custom ranges.

        Args:
            N_t: Number of divisions along the t-axis.
            N_x: Number of divisions along the x-axis.
            t_min: Minimum value for t (default: 0.0).
            t_max: Maximum value for t (default: 1.0).
            x_min: Minimum value for x (default: 0.0).
            x_max: Maximum value for x (default: 1.0).
            times: Number of refinement levels for quadrature (default: 1).
        """
        self.N_t = N_t
        self.N_x = N_x
        self.t_min = t_min
        self.t_max = t_max
        self.x_min = x_min
        self.x_max = x_max
        self.times = times
        self.N_elements = 2 * N_t * N_x  # Two triangles per rectangle
        self.triangles = self._generate_triangles()
        self.coeffs = jnp.zeros((self.N_elements, 3))  # Coefficients for each triangle
        self.quad_points, self.quad_weights, self.element_indices, self.quad_points_tri, self.quad_weights_tri = self._setup_quadrature()

    def _generate_triangles(self):
        """Generate the original triangles for the mesh with custom ranges."""
        tau = (self.t_max - self.t_min) / self.N_t  # Step size along t
        hx = (self.x_max - self.x_min) / self.N_x   # Step size along x
        triangles = []
        for i in range(self.N_t):
            for j in range(self.N_x):
                # Bottom-left triangle
                triangles.append(jnp.array([
                    [self.t_min + i * tau, self.x_min + j * hx],
                    [self.t_min + (i + 1) * tau, self.x_min + j * hx],
                    [self.t_min + i * tau, self.x_min + (j + 1) * hx]
                ]))
                # Top-right triangle
                triangles.append(jnp.array([
                    [self.t_min + (i + 1) * tau, self.x_min + j * hx],
                    [self.t_min + (i + 1) * tau, self.x_min + (j + 1) * hx],
                    [self.t_min + i * tau, self.x_min + (j + 1) * hx]
                ]))
        return jnp.array(triangles)

    def _setup_quadrature(self):
        """
        Setup quadrature points and weights using a refined triangular mesh with
        the three-point trapezoidal rule, optimized with JAX vectorization.

        Returns:
            quad_points: Quadrature points, shape (N_quad, 2).
            quad_weights: Quadrature weights, shape (N_quad,).
            element_indices: Indices of original triangles, shape (N_quad,).
            quad_points_tri: Quadrature points per triangle, shape (N_elements, Nq, 2).
            quad_weights_tri: Quadrature weights per triangle, shape (N_elements, Nq).
        """
        Nq = (4 ** self.times) * 3  # Number of quadrature points per triangle

        # Define a function to refine and compute quadrature for a single triangle
        def refine_and_setup(tri):
            refined_triangles = self._refine_triangle(tri, self.times)
            points = []
            weights = []
            for r_tri in refined_triangles:
                v1, v2, v3 = r_tri
                area = self._triangle_area(v1, v2, v3)
                w = area / 3.0
                for v in [v1, v2, v3]:
                    points.append(v)
                    weights.append(w)
            return jnp.array(points), jnp.array(weights)

        # Vectorize the refinement and quadrature setup across all triangles
        quad_points_tri, quad_weights_tri = vmap(refine_and_setup)(self.triangles)

        # Flatten for global quadrature
        quad_points = quad_points_tri.reshape(-1, 2)
        quad_weights = quad_weights_tri.reshape(-1)
        element_indices = jnp.repeat(jnp.arange(self.N_elements), Nq)

        return quad_points, quad_weights, element_indices, quad_points_tri, quad_weights_tri

    def _refine_triangle(self, tri, times):
        """
        Refine a triangle into 4^times smaller triangles by connecting midpoints.

        Args:
            tri: Array of three vertices, shape (3, 2).
            times: Number of refinement levels.

        Returns:
            Array of refined triangles, shape (4^times, 3, 2).
        """
        if times == 0:
            return tri[None, :]
        else:
            m12 = (tri[0] + tri[1]) / 2
            m13 = (tri[0] + tri[2]) / 2
            m23 = (tri[1] + tri[2]) / 2
            new_tris = jnp.array([
                [tri[0], m12, m13],
                [m12, tri[1], m23],
                [m13, m23, tri[2]],
                [m12, m23, m13]
            ])
            # Recursively refine each new triangle using vmap
            refined = vmap(lambda t: self._refine_triangle(t, times - 1))(new_tris)
            return refined.reshape(-1, 3, 2)

    def _triangle_area(self, v1, v2, v3):
        """
        Compute the area of a triangle given its vertices using the shoelace formula.

        Args:
            v1, v2, v3: Arrays representing the vertices, shape (2,).

        Returns:
            Area of the triangle.
        """
        return 0.5 * jnp.abs((v2[0] - v1[0]) * (v3[1] - v1[1]) - (v2[1] - v1[1]) * (v3[0] - v1[0]))

    def evaluate(self, tx_points, element_indices):
        """Evaluate k_h at given points with corresponding element indices."""
        return (self.coeffs[element_indices, 0] +
                self.coeffs[element_indices, 1] * tx_points[:, 0] +
                self.coeffs[element_indices, 2] * tx_points[:, 1])