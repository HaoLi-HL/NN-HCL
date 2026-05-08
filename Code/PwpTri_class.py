import jax.numpy as jnp
from jax import lax, vmap, pmap
import numpy as np
from scipy import sparse

import flax.linen as nn
import optax.tree_utils as otu
from NN_class import MLP
# import mesh2D as mesh2D


class PiecewiseLinearPolynomialsTri:
    def __init__(self, N_t, N_x, t_min=0.0, t_max=1.0, x_min=0.0, x_max=1.0, rf_times=1, bdry_rf_times=1):
        """
        Initialize a triangular mesh with custom ranges.

        Args:
            N_t: Number of divisions along the t-axis.
            N_x: Number of divisions along the x-axis.
            t_min: Minimum value for t (default: 0.0).
            t_max: Maximum value for t (default: 1.0).
            x_min: Minimum value for x (default: 0.0).
            x_max: Maximum value for x (default: 1.0).
            rf_times: Number of refinement levels for quadrature (default: 1).
        """
        self.N_t = N_t
        self.N_x = N_x
        self.t_min = t_min
        self.t_max = t_max
        self.x_min = x_min
        self.x_max = x_max
        self.ht = (self.t_max - self.t_min) / self.N_t
        self.hx = (self.x_max - self.x_min) / self.N_x
        self.rf_times = rf_times
        self.bdry_rf_times = bdry_rf_times
        self.N_elements = 2 * N_t * N_x  # Two triangles per rectangle
        self.partitions = self._generate_partitions()
        self.p_dof = 3
        self.coeffs = jnp.zeros((self.N_elements, self.p_dof))  # Coefficients for each triangle
        self.quad_points, self.quad_weights, self.quad_element_indices, self.quad_points_parti, self.quad_weights_parti = self._setup_quadrature()
        self.Nq_level0 = self.Nq
        self.Nq_by_level = (self.Nq_level0, self._num_quad_points_for_refinement(self.rf_times + 1))
        self.quad_levels = jnp.zeros(self.N_elements, dtype=jnp.int32)
        self.eta_osc = jnp.zeros(self.N_elements)
        self.eta_quad = jnp.zeros(self.N_elements)
        self.eta_grad = jnp.zeros(self.N_elements)
        self.element_loss_weights = jnp.ones(self.N_elements)
        self.level0_element_indices = jnp.arange(self.N_elements)
        self.level1_element_indices = jnp.array([], dtype=jnp.int32)
        self.quad_points_parti_level1 = jnp.zeros((0, self.Nq_by_level[1], 2), dtype=self.partitions.dtype)
        self.quad_weights_parti_level1 = jnp.zeros((0, self.Nq_by_level[1]), dtype=self.quad_weights_parti.dtype)
        self.level1_cache_positions = -jnp.ones(self.N_elements, dtype=jnp.int32)
        self.bdry_points_parti, self.bdry_weights_parti = self.get_boundary_quadrature()
        self.normals = self.compute_normals()

    def _num_quad_points_for_refinement(self, rf_times):
        return (4 ** rf_times) * 3

    def _generate_partitions(self):
        """Generate the original triangles for the mesh with custom ranges."""
        triangles = []
        for i in range(self.N_t):
            for j in range(self.N_x):
                # Bottom-left triangle
                triangles.append(jnp.array([
                    [self.t_min + i * self.ht, self.x_min + j * self.hx],
                    [self.t_min + (i + 1) * self.ht, self.x_min + j * self.hx],
                    [self.t_min + i * self.ht, self.x_min + (j + 1) * self.hx]
                ]))
                # Top-right triangle
                triangles.append(jnp.array([
                    [self.t_min + (i + 1) * self.ht, self.x_min + j * self.hx],
                    [self.t_min + (i + 1) * self.ht, self.x_min + (j + 1) * self.hx],
                    [self.t_min + i * self.ht, self.x_min + (j + 1) * self.hx]
                ]))
        return jnp.array(triangles)

    def _setup_quadrature(self):
        """
        Setup quadrature points and weights using a refined triangular mesh with
        the three-point trapezoidal rule, optimized with JAX vectorization.

        Returns:
            quad_points: Quadrature points, shape (N_quad, 2).
            quad_weights: Quadrature weights, shape (N_quad,).
            quad_element_indices: Indices of original triangles, shape (N_quad,).
            quad_points_tri: Quadrature points per triangle, shape (N_elements, self.Nq, 2).
            quad_weights_tri: Quadrature weights per triangle, shape (N_elements, self.Nq).
        """
        self.Nq = (4 ** self.rf_times) * 3  # Number of quadrature points per triangle

        # Define a function to refine and compute quadrature for a single triangle
        def refine_and_setup(tri):
            refined_triangles = self._refine_triangle(tri, self.rf_times)
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
        quad_points_tri, quad_weights_tri = vmap(refine_and_setup)(self.partitions)

        # Flatten for global quadrature
        quad_points = quad_points_tri.reshape(-1, 2)
        quad_weights = quad_weights_tri.reshape(-1)
        quad_element_indices = jnp.repeat(jnp.arange(self.N_elements), self.Nq)

        return quad_points, quad_weights, quad_element_indices, quad_points_tri, quad_weights_tri

    def _setup_quadrature_for_elements(self, element_indices, rf_times):
        element_indices = jnp.asarray(element_indices, dtype=jnp.int32)

        def refine_and_setup(tri):
            refined_triangles = self._refine_triangle(tri, rf_times)
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

        return vmap(refine_and_setup)(self.partitions[element_indices])

    def get_quadrature_by_level(self, level, element_indices=None):
        """
        Return physical quadrature points and weights for a level.

        Level 0 uses the existing full per-element arrays. Level 1 is generated
        only for requested elements; the selected refined level-1 group is
        cached separately after each adaptive update.
        """
        if element_indices is None:
            if level == 0:
                return self.Nq_level0, self.quad_points_parti, self.quad_weights_parti
            if level == 1:
                return self.Nq_by_level[1], self.quad_points_parti_level1, self.quad_weights_parti_level1
            raise ValueError("level must be 0 or 1.")

        element_indices = jnp.asarray(element_indices, dtype=jnp.int32)
        if level == 0:
            return self.Nq_by_level[0], self.quad_points_parti[element_indices], self.quad_weights_parti[element_indices]
        if level == 1:
            points, weights = self._setup_quadrature_for_elements(element_indices, self.rf_times + 1)
            return self.Nq_by_level[1], points, weights
        raise ValueError("level must be 0 or 1.")

    def get_quadrature_group_by_level(self, level):
        """
        Return the current element group and stored quadrature arrays for one level.

        This is the adaptive-loss path. Level 0 gathers from the stored full
        level-0 bank. Level 1 returns the cached selected refined bank.
        """
        if level == 0:
            element_indices = self.level0_element_indices
            return element_indices, self.quad_points_parti[element_indices], self.quad_weights_parti[element_indices]
        if level == 1:
            return self.level1_element_indices, self.quad_points_parti_level1, self.quad_weights_parti_level1
        raise ValueError("level must be 0 or 1.")

    def reset_adaptive_quadrature(self):
        self.quad_levels = jnp.zeros(self.N_elements, dtype=jnp.int32)
        self.eta_osc = jnp.zeros(self.N_elements)
        self.eta_quad = jnp.zeros(self.N_elements)
        self.eta_grad = jnp.zeros(self.N_elements)
        self.element_loss_weights = jnp.ones(self.N_elements)
        self.level0_element_indices = jnp.arange(self.N_elements)
        self.level1_element_indices = jnp.array([], dtype=jnp.int32)
        self.quad_points_parti_level1 = jnp.zeros((0, self.Nq_by_level[1], 2), dtype=self.partitions.dtype)
        self.quad_weights_parti_level1 = jnp.zeros((0, self.Nq_by_level[1]), dtype=self.quad_weights_parti.dtype)
        self.level1_cache_positions = -jnp.ones(self.N_elements, dtype=jnp.int32)

    def update_adaptive_quadrature_by_indicator(self, indicator, top_fraction=None, top_count=None, indicator_name=None):
        """
        Select level-1 elements from a scalar elementwise indicator.

        The selected elements get level-1 quadrature and double element loss
        weight. Level-1 physical quadrature is cached only for selected
        elements.
        """
        if top_count is None:
            if top_fraction is None:
                top_fraction = 2.0 / self.N_x
            top_count = int(np.ceil(float(top_fraction) * self.N_elements))
        top_count = max(0, min(self.N_elements, int(top_count)))

        indicator = jnp.asarray(indicator).reshape((self.N_elements,))
        if top_count == 0:
            level1_element_indices = jnp.array([], dtype=jnp.int32)
        else:
            _, level1_element_indices = lax.top_k(indicator, top_count)

        quad_levels = jnp.zeros(self.N_elements, dtype=jnp.int32)
        quad_levels = quad_levels.at[level1_element_indices].set(1)
        level0_element_indices = jnp.where(quad_levels == 0)[0]
        element_loss_weights = jnp.ones(self.N_elements)
        element_loss_weights = element_loss_weights.at[level1_element_indices].set(2)
        if top_count == 0:
            quad_points_parti_level1 = jnp.zeros((0, self.Nq_by_level[1], 2), dtype=self.partitions.dtype)
            quad_weights_parti_level1 = jnp.zeros((0, self.Nq_by_level[1]), dtype=self.quad_weights_parti.dtype)
        else:
            _, quad_points_parti_level1, quad_weights_parti_level1 = self.get_quadrature_by_level(1, level1_element_indices)
        level1_cache_positions = -jnp.ones(self.N_elements, dtype=jnp.int32)
        level1_cache_positions = level1_cache_positions.at[level1_element_indices].set(jnp.arange(top_count, dtype=jnp.int32))

        self.quad_levels = quad_levels
        if indicator_name == 'quad':
            self.eta_quad = indicator
        elif indicator_name == 'grad':
            self.eta_grad = indicator
        else:
            self.eta_osc = indicator
        self.element_loss_weights = element_loss_weights
        self.level0_element_indices = level0_element_indices
        self.level1_element_indices = level1_element_indices
        self.quad_points_parti_level1 = quad_points_parti_level1
        self.quad_weights_parti_level1 = quad_weights_parti_level1
        self.level1_cache_positions = level1_cache_positions
        return indicator, level1_element_indices

    def _pad_level0_quadrature_subset(self, element_indices):
        points0 = self.quad_points_parti[element_indices]
        weights0 = self.quad_weights_parti[element_indices]
        pad_size = self.Nq_by_level[1] - self.Nq_by_level[0]
        if pad_size <= 0:
            return points0, weights0
        pad_points = jnp.repeat(points0[:, :1, :], pad_size, axis=1)
        pad_weights = jnp.zeros((points0.shape[0], pad_size), dtype=weights0.dtype)
        return jnp.concatenate([points0, pad_points], axis=1), jnp.concatenate([weights0, pad_weights], axis=1)

    def get_adaptive_quadrature_for_elements(self, element_indices):
        """
        Return fixed-shape adaptive quadrature for selected elements only.

        This is intended for sampled batches whose level mix is not known
        statically. Non-refined elements use level-0 quadrature padded by
        zero-weight points; refined elements use level-1 quadrature.
        """
        element_indices = jnp.asarray(element_indices, dtype=jnp.int32)
        levels = self.quad_levels[element_indices]
        points0, weights0 = self._pad_level0_quadrature_subset(element_indices)
        if self.level1_element_indices.size == 0:
            return points0, weights0
        level1_positions = jnp.maximum(self.level1_cache_positions[element_indices], 0)
        points1 = self.quad_points_parti_level1[level1_positions]
        weights1 = self.quad_weights_parti_level1[level1_positions]
        level_mask_points = levels[:, None, None].astype(bool)
        level_mask_weights = levels[:, None].astype(bool)
        return jnp.where(level_mask_points, points1, points0), jnp.where(level_mask_weights, weights1, weights0)

    def _refine_triangle(self, tri, rf_times):
        """
        Refine a triangle into 4^rf_times smaller triangles by connecting midpoints.

        Args:
            tri: Array of three vertices, shape (3, 2).
            rf_times: Number of refinement levels.

        Returns:
            Array of refined triangles, shape (4^rf_times, 3, 2).
        """
        if rf_times == 0:
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
            refined = vmap(lambda t: self._refine_triangle(t, rf_times - 1))(new_tris)
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

    def evaluate(self, tx_points, quad_element_indices):
        """Evaluate k_h at given points with corresponding element indices."""
        return (self.coeffs[quad_element_indices, 0] +
                self.coeffs[quad_element_indices, 1] * tx_points[:, 0] +
                self.coeffs[quad_element_indices, 2] * tx_points[:, 1])

    def get_boundary_quadrature(self):
        """
        Generate composite trapezoid rule quadrature points and weights for the boundary of each triangle.

        Returns:
            bdry_points_tri: Quadrature points for the boundaries, shape (N_elements, 3*(2**bdry_rf_times + 1), 2).
            bdry_weights_tri: Quadrature weights for the boundaries, shape (N_elements, 3*(2**bdry_rf_times + 1)).
        """
        bdry_rf_times = self.bdry_rf_times
        n = 2 ** bdry_rf_times  # Number of subintervals per edge
        def _edge_quadrature(v1, v2):
            """
            Compute quadrature points and weights for a single edge using the composite trapezoid rule.

            Args:
                v1: Starting vertex of the edge, shape (2,).
                v2: Ending vertex of the edge, shape (2,).

            Returns:
                points: Quadrature points along the edge, shape (n+1, 2).
                weights: Quadrature weights for the points, shape (n+1,).
            """
            L = jnp.linalg.norm(v2 - v1)  # Edge length
            t = jnp.linspace(0, 1, n + 1)  # Parameter values from 0 to 1
            points = v1 + t[:, None] * (v2 - v1)  # Linearly spaced points along the edge
            weights = jnp.full((n + 1,), L / n)  # Interior weights
            weights = weights.at[0].set(L / (2 * n))  # Endpoint weight (start)
            weights = weights.at[-1].set(L / (2 * n))  # Endpoint weight (end)
            return points, weights

        # Vectorize the edge quadrature function over multiple edges
        edge_quadrature_vmap = vmap(_edge_quadrature, in_axes=(0, 0))

        # Compute quadrature for each set of edges for all triangles
        points1, weights1 = edge_quadrature_vmap(self.partitions[:, 0], self.partitions[:, 1])  # Edge 0-1
        points2, weights2 = edge_quadrature_vmap(self.partitions[:, 1], self.partitions[:, 2])  # Edge 1-2
        points3, weights3 = edge_quadrature_vmap(self.partitions[:, 2], self.partitions[:, 0])  # Edge 2-0

        # Concatenate points and weights for all three edges per triangle
        bdry_points_tri = jnp.concatenate([points1, points2, points3], axis=1)  # Shape: (N_elements, 3*(n+1), 2)
        bdry_weights_tri = jnp.concatenate([weights1, weights2, weights3], axis=1)  # Shape: (N_elements, 3*(n+1))

        return bdry_points_tri, bdry_weights_tri

    
    def compute_normals(self):
        """Compute outward unit normals for each edge of each triangle."""
        triangles = self.partitions
        v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        # Edge 0-1
        tangent01 = v1 - v0
        normal01 = jnp.stack([-tangent01[:, 1], tangent01[:, 0]], axis=1) / jnp.linalg.norm(tangent01, axis=1, keepdims=True)
        # Edge 1-2
        tangent12 = v2 - v1
        normal12 = jnp.stack([-tangent12[:, 1], tangent12[:, 0]], axis=1) / jnp.linalg.norm(tangent12, axis=1, keepdims=True)
        # Edge 2-0
        tangent20 = v0 - v2
        normal20 = jnp.stack([-tangent20[:, 1], tangent20[:, 0]], axis=1) / jnp.linalg.norm(tangent20, axis=1, keepdims=True)
        return jnp.stack([normal01, normal12, normal20], axis=1)  # Shape: (N_elements, 3, 2)
