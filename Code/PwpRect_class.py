import jax.numpy as jnp
from jax import vmap
import numpy as np

class PiecewiseLinearPolynomialsRect:
    def __init__(self, N_t, N_x, t_min=0.0, t_max=1.0, x_min=-1.0, x_max=1.0, rf_times=3, bdry_order=3, dg_type='Q1'):
        """
        Initialize a rectangular mesh with custom ranges.

        Args:
            N_t: Number of divisions along the t-axis.
            N_x: Number of divisions along the x-axis.
            t_min: Minimum value for t (default: 0.0).
            t_max: Maximum value for t (default: 1.0).
            x_min: Minimum value for x (default: -1.0).
            x_max: Maximum value for x (default: 1.0).
            rf_times: Refinement level for composite quadrature. 
            bdry_order: Order of boundary quadrature (number of points per edge, default: 3).
        """
        self.N_t = N_t
        self.N_x = N_x
        self.t_min = t_min
        self.t_max = t_max
        self.x_min = x_min
        self.x_max = x_max
        self.rf_times = rf_times
        self.bdry_order = bdry_order
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
        Setup Gauss quadrature points and weights for the boundaries of the rectangular mesh.

        Returns:
            bdry_points_parti: Boundary quadrature points per rectangle, shape (N_elements, 4 * bdry_order, 2).
            bdry_weights_parti: Boundary quadrature weights per rectangle, shape (N_elements, 4 * bdry_order).
        """
        # Gauss-Legendre nodes and weights in [-1, 1]
        nodes, weights = np.polynomial.legendre.leggauss(self.bdry_order)
        nodes_jnp = jnp.array(nodes)
        weights_jnp = jnp.array(weights)

        def boundary_quad(rect):
            t_low, x_low = rect[0]
            t_high, x_high = rect[3]
            points_list = []
            weights_list = []

            # Bottom edge: t = t_low, x varies
            scale_x = (x_high - x_low) / 2
            center_x = (x_high + x_low) / 2
            x_points = center_x + nodes_jnp * scale_x
            t_points = jnp.full(self.bdry_order, t_low)
            points = jnp.stack([t_points, x_points], axis=1)
            points_list.append(points)
            w_x = weights_jnp * scale_x
            weights_list.append(w_x)

            # Top edge: t = t_high, x varies
            x_points = center_x + nodes_jnp * scale_x
            t_points = jnp.full(self.bdry_order, t_high)
            points = jnp.stack([t_points, x_points], axis=1)
            points_list.append(points)
            weights_list.append(w_x)

            # Left edge: x = x_low, t varies
            scale_t = (t_high - t_low) / 2
            center_t = (t_high + t_low) / 2
            t_points = center_t + nodes_jnp * scale_t
            x_points = jnp.full(self.bdry_order, x_low)
            points = jnp.stack([t_points, x_points], axis=1)
            points_list.append(points)
            w_t = weights_jnp * scale_t
            weights_list.append(w_t)

            # Right edge: x = x_high, t varies
            t_points = center_t + nodes_jnp * scale_t
            x_points = jnp.full(self.bdry_order, x_high)
            points = jnp.stack([t_points, x_points], axis=1)
            points_list.append(points)
            weights_list.append(w_t)

            all_points = jnp.vstack(points_list)
            all_weights = jnp.concatenate(weights_list)
            return all_points, all_weights

        bdry_points_parti, bdry_weights_parti = vmap(boundary_quad)(self.partitions)
        return bdry_points_parti, bdry_weights_parti

    def compute_normals(self):
        """
        Compute outward normal vectors for each boundary quadrature point.

        Returns:
            normals: Normal vectors, shape (N_elements, 4 * bdry_order, 2).
        """
        N_bd_per_face = self.bdry_order
        normals_list = []

        # Bottom: (-1, 0)
        normals_bottom = jnp.tile(jnp.array([-1.0, 0.0]), (N_bd_per_face, 1))
        normals_list.append(normals_bottom)

        # Top: (1, 0)
        normals_top = jnp.tile(jnp.array([1.0, 0.0]), (N_bd_per_face, 1))
        normals_list.append(normals_top)

        # Left: (0, -1)
        normals_left = jnp.tile(jnp.array([0.0, -1.0]), (N_bd_per_face, 1))
        normals_list.append(normals_left)

        # Right: (0, 1)
        normals_right = jnp.tile(jnp.array([0.0, 1.0]), (N_bd_per_face, 1))
        normals_list.append(normals_right)

        normals_per_element = jnp.vstack(normals_list)
        normals = jnp.tile(normals_per_element[None, :, :], (self.N_elements, 1, 1))
        return normals