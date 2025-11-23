import jax.numpy as jnp
from jax import vmap
import numpy as np

class PiecewiseLinearPolynomialsRect:
    def __init__(self, N_t, N_x, t_min=0.0, t_max=1.0, x_min=0.0, x_max=1.0, rf_times=3, bdry_rf_times=3, dg_type='Q1'):
        """
        Initialize a rectangular mesh with custom ranges.

        Args:
            N_t: Number of divisions along the t-axis.
            N_x: Number of divisions along the x-axis.
            t_min: Minimum value for t (default: 0.0).
            t_max: Maximum value for t (default: 1.0).
            x_min: Minimum value for x (default: 0.0).
            x_max: Maximum value for x (default: 1.0).
            rf_times: Refinement level for composite quadrature. 
            bdry_rf_times: Order of boundary quadrature (number of points per edge, default: 3).
        """
        self.N_t = N_t
        self.N_x = N_x
        self.t_min = t_min
        self.t_max = t_max
        self.x_min = x_min
        self.x_max = x_max
        self.rf_times = rf_times
        self.bdry_rf_times = bdry_rf_times
        self.N_elements = N_t * N_x  # Number of rectangles
        self.partitions = self._generate_rectangles()
        if dg_type == 'Q1':
            self.p_dof = 4
        elif dg_type == 'P1':
            self.p_dof = 3
        self.coeffs = jnp.zeros((self.N_elements, self.p_dof))  # Coefficients a, b, c for a + b*t + c*x per rectangle
        self.quad_points, self.quad_weights, self.quad_element_indices, self.quad_points_parti, self.quad_weights_parti = self._setup_quadrature()
        self.bdry_points_parti, self.bdry_weights_parti = self.get_boundary_quadrature()
        self.normals = self.compute_normals()

    def _generate_rectangles(self):
        """Generate the rectangles for the mesh with custom ranges."""
        self.ht = (self.t_max - self.t_min) / self.N_t  # Step size along t
        self.hx = (self.x_max - self.x_min) / self.N_x   # Step size along x

        t_starts = jnp.linspace(self.t_min, self.t_max - self.ht, self.N_t)
        x_starts = jnp.linspace(self.x_min, self.x_max - self.hx, self.N_x)
        tl, xl = jnp.meshgrid(t_starts, x_starts)
        # Define the 4 vertices for each hexahedron using binary offsets
        offsets_bin = jnp.array([[int(b) for b in f'{i:02b}'] for i in range(4)], dtype=jnp.float32)  # (4, 2) binary 00 to 11
        deltas = jnp.array([self.ht, self.hx])
        offsets = offsets_bin * deltas  # (4,2)
        lowers = jnp.stack([tl.ravel(), xl.ravel()], axis=1)[:, None, :]  # (N_elements, 1, 2)
        partitions = lowers + offsets[None, :, :]  # (N_elements, 4, 2)
        return partitions

    def _setup_quadrature(self):
        """
        Setup Gauss quadrature points and weights for rectangular mesh.

        Returns:
            quad_points: Quadrature points, shape (N_quad, 2).
            quad_weights: Quadrature weights, shape (N_quad,).
            quad_element_indices: Indices of rectangles, shape (N_quad,).
            quad_points_parti: Quadrature points per rectangle, shape (N_elements, self.Nq, 2).
            quad_weights_parti: Quadrature weights per rectangle, shape (N_elements, self.Nq).
        """
        
        # Composite quadrature with refinement level k = self.rf_times
        k = self.rf_times
        num_intervals = 2 ** k
        num_points = num_intervals + 1
        self.Nq = num_points ** 2

        def composite_quad(rect):
            t_low, x_low = rect[0]
            t_high, x_high = rect[3]
            dt = (t_high - t_low) / num_intervals
            dx = (x_high - x_low) / num_intervals
            t_grid = jnp.linspace(t_low, t_high, num_points)
            x_grid = jnp.linspace(x_low, x_high, num_points)
            tt, xx = jnp.meshgrid(t_grid, x_grid, indexing='ij')
            points = jnp.stack([tt.ravel(), xx.ravel()], axis=1)
            wt = jnp.full(num_points, dt)
            wt = wt.at[0].set(dt / 2)
            wt = wt.at[-1].set(dt / 2)
            wx = jnp.full(num_points, dx)
            wx = wx.at[0].set(dx / 2)
            wx = wx.at[-1].set(dx / 2)
            weights_2d = wt[:, None] * wx[None, :]
            weights = weights_2d.ravel()
            return points, weights

        quad_points_parti, quad_weights_parti = vmap(composite_quad)(self.partitions)

        # Flatten for global quadrature
        quad_points = quad_points_parti.reshape(-1, 2)
        quad_weights = quad_weights_parti.reshape(-1)
        quad_element_indices = jnp.repeat(jnp.arange(self.N_elements), self.Nq)

        return quad_points, quad_weights, quad_element_indices, quad_points_parti, quad_weights_parti

    def get_boundary_quadrature(self):
        """
        Generate composite trapezoidal quadrature points and weights for the boundary of each rectangle.

        Returns:
            bdry_points_parti: Quadrature points per rectangle per edge, shape (N_elements, 4, Nq_edge, 2).
            bdry_weights_parti: Quadrature weights per rectangle per edge, shape (N_elements, 4, Nq_edge).
        """
        k = self.bdry_rf_times
        num_intervals = 2 ** k
        num_points = num_intervals + 1
        Nq_edge = num_points

        edges_info = jnp.array([
            [0, 0, 1],  # low t: fixed dim 0 (t) low, vary dim 1 (x)
            [0, 1, 1],  # high t: fixed dim 0 high, vary dim 1
            [1, 0, 0],  # low x: fixed dim 1 (x) low, vary dim 0 (t)
            [1, 1, 0],  # high x: fixed dim 1 high, vary dim 0
        ])

        def edge_quad(rect, info):
            fixed_dim = info[0].astype(jnp.int32)
            fixed_sign = info[1].astype(jnp.int32)
            vary_dim = info[2].astype(jnp.int32)
            lows = rect[0]
            highs = rect[3]
            fixed_val = jnp.where(fixed_sign == 0, lows[fixed_dim], highs[fixed_dim])
            interval = (highs[vary_dim] - lows[vary_dim]) / num_intervals
            grid = lows[vary_dim] + jnp.arange(num_points) * interval
            points = jnp.full((Nq_edge, 2), fixed_val)
            points = points.at[:, vary_dim].set(grid)
            wb = jnp.full(num_points, interval)
            wb = wb.at[0].set(interval / 2)
            wb = wb.at[-1].set(interval / 2)
            weights = wb
            return points, weights

        def compute_for_info(info):
            return vmap(lambda rect: edge_quad(rect, info))(self.partitions)

        points_parti, weights_parti = vmap(compute_for_info)(edges_info)
        bdry_points_parti = jnp.transpose(points_parti, (1, 0, 2, 3))
        bdry_weights_parti = jnp.transpose(weights_parti, (1, 0, 2))
        return bdry_points_parti, bdry_weights_parti

    def compute_normals(self):
        """
        Compute outward unit normals for each edge of each rectangle.

        Returns:
            normals: Outward unit normals, shape (N_elements, 4, 2).
        """
        edge_vertex_indices = jnp.array([
            [0, 1],  # low t (bottom)
            [2, 3],  # high t (top)
            [0, 2],  # low x (left)
            [1, 3],  # high x (right)
        ])

        def compute_normal(rect, edge_idx):
            verts_idx = edge_vertex_indices[edge_idx]
            points = rect[verts_idx]
            u = points[1] - points[0]
            n = jnp.array([-u[1], u[0]])  # Rotate 90 degrees counterclockwise
            n = n / jnp.linalg.norm(n + 1e-12)  # Avoid division by zero
            elem_center = jnp.mean(rect, axis=0)
            edge_center = jnp.mean(points, axis=0)
            vec = edge_center - elem_center
            # If dot < 0, flip to make outward
            n = jnp.where(jnp.dot(n, vec) < 0, -n, n)
            return n

        vmap_edge = vmap(compute_normal, in_axes=(None, 0))
        # Apply over elements
        vmap_elem = vmap(lambda rect: vmap_edge(rect, jnp.arange(4)))
        return vmap_elem(self.partitions)