import jax.numpy as jnp
from jax import vmap
import numpy as np

class PiecewiseLinearPolynomialsHex:
    def __init__(self, N_t, N_x, N_y, t_min=0.0, t_max=1.0, x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, rf_times=2, bdry_rf_times=3, dg_type='Q1'):
        """
        Initialize a hexahedral mesh with custom ranges.

        Args:
            N_t: Number of divisions along the t-axis.
            N_x: Number of divisions along the x-axis.
            N_y: Number of divisions along the y-axis.
            t_min: Minimum value for t (default: 0.0).
            t_max: Maximum value for t (default: 1.0).
            x_min: Minimum value for x (default: -1.0).
            x_max: Maximum value for x (default: 1.0).
            y_min: Minimum value for y (default: -1.0).
            y_max: Maximum value for y (default: 1.0).
            rf_times: Refinement level for composite quadrature. 
            bdry_rf_times: Refinement level for boundary quadrature.
        """
        self.N_t = N_t
        self.N_x = N_x
        self.N_y = N_y
        self.t_min = t_min
        self.t_max = t_max
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.rf_times = rf_times
        self.bdry_rf_times = bdry_rf_times
        self.N_elements = N_t * N_x * N_y  # Number of hexahedra
        self.partitions = self._generate_hexahedra()
        if dg_type == 'Q1':
            self.p_dof = 8
        elif dg_type == 'P1':
            self.p_dof = 4
        self.coeffs = jnp.zeros((self.N_elements, self.p_dof))
        self.quad_points, self.quad_weights, self.quad_element_indices, self.quad_points_parti, self.quad_weights_parti = self._setup_quadrature()
        self.bdry_points_parti, self.bdry_weights_parti = self.get_boundary_quadrature()
        self.normals = self.compute_normals()

    def _generate_hexahedra(self):
        """Generate the hexahedra for the mesh with custom ranges."""
        self.ht = (self.t_max - self.t_min) / self.N_t
        self.hx = (self.x_max - self.x_min) / self.N_x
        self.hy = (self.y_max - self.y_min) / self.N_y
        t_starts = jnp.linspace(self.t_min, self.t_max - self.ht, self.N_t)
        x_starts = jnp.linspace(self.x_min, self.x_max - self.hx, self.N_x)
        y_starts = jnp.linspace(self.y_min, self.y_max - self.hy, self.N_y)
        tl, xl, yl = jnp.meshgrid(t_starts, x_starts, y_starts, indexing='ij')
        # Define the 8 vertices for each hexahedron using binary offsets
        offsets_bin = jnp.array([[int(b) for b in f'{i:03b}'] for i in range(8)], dtype=jnp.float32)  # (8,3) binary 000 to 111
        deltas = jnp.array([self.ht, self.hx, self.hy])
        offsets = offsets_bin * deltas  # (8,3)
        lowers = jnp.stack([tl.ravel(), xl.ravel(), yl.ravel()], axis=1)[:, None, :]  # (N_elements, 1, 3)
        partitions = lowers + offsets[None, :, :]  # (N_elements, 8, 3)
        return partitions
        
    def _setup_quadrature(self):
        """
        Setup composite quadrature points and weights for hexahedral mesh.

        Returns:
            quad_points: Quadrature points, shape (N_quad, 3).
            quad_weights: Quadrature weights, shape (N_quad,).
            quad_element_indices: Indices of hexahedra, shape (N_quad,).
            quad_points_parti: Quadrature points per hexahedron, shape (N_elements, self.Nq, 3).
            quad_weights_parti: Quadrature weights per hexahedron, shape (N_elements, self.Nq).
        """
        # Composite quadrature with refinement level k = self.rf_times
        k = self.rf_times
        num_intervals = 2 ** k
        num_points = num_intervals + 1
        self.Nq = num_points ** 3

        def composite_quad(hex):
            t_low, x_low, y_low = hex[0]
            t_high, x_high, y_high = hex[-1]
            dt = (t_high - t_low) / num_intervals
            dx = (x_high - x_low) / num_intervals
            dy = (y_high - y_low) / num_intervals
            t_grid = jnp.linspace(t_low, t_high, num_points)
            x_grid = jnp.linspace(x_low, x_high, num_points)
            y_grid = jnp.linspace(y_low, y_high, num_points)
            tt, xx, yy = jnp.meshgrid(t_grid, x_grid, y_grid, indexing='ij')
            points = jnp.stack([tt.ravel(), xx.ravel(), yy.ravel()], axis=1)
            wt = jnp.full(num_points, dt)
            wt = wt.at[0].set(dt / 2)
            wt = wt.at[-1].set(dt / 2)
            wx = jnp.full(num_points, dx)
            wx = wx.at[0].set(dx / 2)
            wx = wx.at[-1].set(dx / 2)
            wy = jnp.full(num_points, dy)
            wy = wy.at[0].set(dy / 2)
            wy = wy.at[-1].set(dy / 2)
            weights_3d = wt[:, None, None] * wx[None, :, None] * wy[None, None, :]
            weights = weights_3d.ravel()
            return points, weights

        # Vectorize over hexahedra
        quad_points_parti, quad_weights_parti = vmap(composite_quad)(self.partitions)

        # Flatten for global quadrature
        quad_points = quad_points_parti.reshape(-1, 3)
        quad_weights = quad_weights_parti.reshape(-1)
        quad_element_indices = jnp.repeat(jnp.arange(self.N_elements), self.Nq)

        return quad_points, quad_weights, quad_element_indices, quad_points_parti, quad_weights_parti

    def get_boundary_quadrature(self):
        """
        Generate composite trapezoidal quadrature points and weights for the boundary of each hexahedron.

        Returns:
            bdry_points_parti: Quadrature points per hexahedron per face, shape (N_elements, 6, Nq_face, 3).
            bdry_weights_parti: Quadrature weights per hexahedron per face, shape (N_elements, 6, Nq_face).
        """
        k = self.bdry_rf_times
        num_intervals = 2 ** k
        num_points = num_intervals + 1
        Nq_face = num_points ** 2

        faces_info = jnp.array([
            [0, 0, 1, 2],  # Face 0: fixed_dim=0 (t), fixed_sign=0 (low t), vary_dim1=1 (x), vary_dim2=2 (y)
            [0, 1, 1, 2],  # Face 1: fixed_dim=0 (t), fixed_sign=1 (high t), vary_dim1=1 (x), vary_dim2=2 (y)
            [1, 0, 0, 2],  # Face 2: fixed_dim=1 (x), fixed_sign=0 (low x), vary_dim1=0 (t), vary_dim2=2 (y)
            [1, 1, 0, 2],  # Face 3: fixed_dim=1 (x), fixed_sign=1 (high x), vary_dim1=0 (t), vary_dim2=2 (y)
            [2, 0, 0, 1],  # Face 4: fixed_dim=2 (y), fixed_sign=0 (low y), vary_dim1=0 (t), vary_dim2=1 (x)
            [2, 1, 0, 1],  # Face 5: fixed_dim=2 (y), fixed_sign=1 (high y), vary_dim1=0 (t), vary_dim2=1 (x)
        ])

        def face_quad(hex, info):
            fixed_dim = info[0].astype(jnp.int32)
            fixed_sign = info[1].astype(jnp.int32)
            vary_dims = info[2:4].astype(jnp.int32)
            lows = hex[0]
            highs = hex[-1]
            fixed_val = jnp.where(fixed_sign == 0, lows[fixed_dim], highs[fixed_dim])
            intervals = (highs[vary_dims] - lows[vary_dims]) / num_intervals
            grids = [lows[d] + jnp.arange(num_points) * intervals[j] for j, d in enumerate(vary_dims)]
            w_base = []
            for j in range(2):
                wb = jnp.full(num_points, intervals[j])
                wb = wb.at[0].set(intervals[j] / 2)
                wb = wb.at[-1].set(intervals[j] / 2)
                w_base.append(wb)
            gg1, gg2 = jnp.meshgrid(grids[0], grids[1], indexing='ij')
            points = jnp.full((Nq_face, 3), fixed_val)
            points = points.at[:, vary_dims[0]].set(gg1.ravel())
            points = points.at[:, vary_dims[1]].set(gg2.ravel())
            weights_2d = w_base[0][:, None] * w_base[1][None, :]
            weights = weights_2d.ravel()
            return points, weights

        def compute_for_info(info):
            return vmap(lambda hex: face_quad(hex, info))(self.partitions)

        points_parti, weights_parti = vmap(compute_for_info)(faces_info)
        bdry_points_parti = jnp.transpose(points_parti, (1, 0, 2, 3))
        bdry_weights_parti = jnp.transpose(weights_parti, (1, 0, 2))
        return bdry_points_parti, bdry_weights_parti

    def compute_normals(self):
        """
        Compute outward unit normals for each face of each hexahedron.

        Returns:
            normals: Outward unit normals, shape (N_elements, 6, 3).
        """
        face_vertex_indices = jnp.array([
            [0, 1, 2, 3],  # low t
            [4, 5, 6, 7],  # high t
            [0, 1, 4, 5],  # low x
            [2, 3, 6, 7],  # high x
            [0, 2, 4, 6],  # low y
            [1, 3, 5, 7],  # high y
        ])

        def compute_normal(hex, face_idx):
            verts_idx = face_vertex_indices[face_idx]
            points = hex[verts_idx]
            u = points[1] - points[0]
            v = points[2] - points[0]
            n = jnp.cross(u, v)
            n = n / jnp.linalg.norm(n + 1e-12)  # Avoid division by zero
            elem_center = (hex[0] + hex[-1]) / 2
            face_center = jnp.mean(points, axis=0)
            vec = face_center - elem_center
            # If dot < 0, flip to make outward
            n = jnp.where(jnp.dot(n, vec) < 0, -n, n)
            return n

        vmap_face = vmap(compute_normal, in_axes=(None, 0))
        vmap_elem = vmap(lambda hex: vmap_face(hex, jnp.arange(6)))
        return vmap_elem(self.partitions)