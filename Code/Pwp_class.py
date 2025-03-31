import jax
import jax.numpy as jnp
import numpy as np
import jax.numpy as jnp
import jax
from scipy import sparse

import flax.linen as nn
import optax.tree_utils as otu
from NN_class import MLP
# import mesh2D as mesh2D

class PiecewiseLinearPolynomials:
    def __init__(self, N_t, N_x, times=1):
        """
        Initialize piecewise linear polynomials on a triangular mesh.

        Args:
            N_t: Number of divisions along the t-axis.
            N_x: Number of divisions along the x-axis.
            times: Number of refinement levels for quadrature (default is 1).
        """
        self.N_t = N_t
        self.N_x = N_x
        self.times = times
        self.N_elements = 2 * N_t * N_x  # Two triangles per rectangle
        self.triangles = self._generate_triangles()
        self.coeffs = jnp.zeros((self.N_elements, 3))  # Coefficients for each triangle
        self.quad_points, self.quad_weights, self.element_indices, self.quad_points_tri, self.quad_weights_tri = self._setup_quadrature()

    def _generate_triangles(self):
        """Generate the original triangles for the mesh."""
        tau = 1.0 / self.N_t
        hx = 1.0 / self.N_x
        triangles = []
        for i in range(self.N_t):
            for j in range(self.N_x):
                # Bottom-left triangle
                triangles.append(jnp.array([
                    [i * tau, j * hx],
                    [(i + 1) * tau, j * hx],
                    [i * tau, (j + 1) * hx]
                ]))
                # Top-right triangle
                triangles.append(jnp.array([
                    [(i + 1) * tau, j * hx],
                    [(i + 1) * tau, (j + 1) * hx],
                    [i * tau, (j + 1) * hx]
                ]))
        return jnp.array(triangles)
    
    def _setup_quadrature(self):
        """
        Setup quadrature points and weights using a refined triangular mesh with 
        the three-point trapezoidal rule.

        Returns:
            quad_points: Quadrature points, shape (N_quad, 2).
            quad_weights: Quadrature weights, shape (N_quad,).
            element_indices: Indices of original triangles, shape (N_quad,).
        """
        N_t, N_x, times = self.N_t, self.N_x, self.times
        tau = 1.0 / N_t  # Step size along t
        hx = 1.0 / N_x   # Step size along x
        
        # Refine each triangle 'times' times
        triangles = self.triangles
        refined_triangles = []

        # Step 3: Apply three-point trapezoidal rule on each refined triangle
        element_indices = []
        # Nq = number of quadrature points = 4^times * 3
        Nq = (4 ** times) * 3
        quad_points_tri = jnp.zeros((self.N_elements, Nq, 2))
        quad_weights_tri = jnp.zeros((self.N_elements, Nq))

        for idx, tri in enumerate(triangles):
            refined_triangles = self._refine_triangle(tri, times)
            points = []
            weights = []
            for r_tri in refined_triangles:
                v1, v2, v3 = r_tri
                # Compute area of the refined triangle
                area = self._triangle_area(v1, v2, v3)
                # Weight for each vertex in the three-point trapezoidal rule
                w = area / 3.0
                # Add the three vertices as quadrature points
                for v in [v1, v2, v3]:
                    points.append(v)
                    weights.append(w)
                    element_indices.append(idx)
            # Store in arrays
            quad_points_tri = quad_points_tri.at[idx].set(jnp.array(points))
            quad_weights_tri = quad_weights_tri.at[idx].set(jnp.array(weights))

        quad_points = quad_points_tri.reshape((self.N_elements*Nq, -1))          # Shape: (N_quad, 2)
        quad_weights = quad_weights_tri.reshape((self.N_elements*Nq, -1))          # Shape: (N_quad,)
        element_indices = jnp.array(element_indices)   # Shape: (N_quad,)
        return quad_points, quad_weights, element_indices, quad_points_tri, quad_weights_tri

    def _refine_triangle(self, tri, times):
        """
        Refine a triangle into 4^times smaller triangles by connecting midpoints.

        Args:
            tri: List of three vertices [(t1,x1), (t2,x2), (t3,x3)].
            times: Number of refinement levels.

        Returns:
            List of refined triangles.
        """
        if times == 0:
            return [tri]
        else:
            # Compute midpoints of the edges
            m12 = ((tri[0][0] + tri[1][0]) / 2, (tri[0][1] + tri[1][1]) / 2)  # Midpoint of v1-v2
            m13 = ((tri[0][0] + tri[2][0]) / 2, (tri[0][1] + tri[2][1]) / 2)  # Midpoint of v1-v3
            m23 = ((tri[1][0] + tri[2][0]) / 2, (tri[1][1] + tri[2][1]) / 2)  # Midpoint of v2-v3
            # Form four new triangles
            new_tris = [
                [tri[0], m12, m13],   # Triangle with vertex v1
                [m12, tri[1], m23],   # Triangle with vertex v2
                [m13, m23, tri[2]],   # Triangle with vertex v3
                [m12, m23, m13]       # Central triangle
            ]
            # Recursively refine each new triangle
            refined = []
            for new_tri in new_tris:
                refined.extend(self._refine_triangle(new_tri, times - 1))
            return refined

    def _triangle_area(self, v1, v2, v3):
        """
        Compute the area of a triangle given its vertices using the shoelace formula.

        Args:
            v1, v2, v3: Tuples (t, x) representing the vertices.

        Returns:
            Area of the triangle.
        """
        t1, x1 = v1
        t2, x2 = v2
        t3, x3 = v3
        return 0.5 * abs((t2 - t1) * (x3 - x1) - (t3 - t1) * (x2 - x1))

    def evaluate(self, tx_points, element_indices):
        """Evaluate k_h at given points with corresponding element indices."""
        return (self.coeffs[element_indices, 0] +
                self.coeffs[element_indices, 1] * tx_points[:, 0] +
                self.coeffs[element_indices, 2] * tx_points[:, 1])
        