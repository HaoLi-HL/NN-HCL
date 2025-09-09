import jax.numpy as jnp
from jax import vmap
import numpy as np

class PiecewiseLinearPolynomialsTetra:
    def __init__(self, N_t, N_x, N_y, t_min=0.0, t_max=1.0, x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, quad_order=2, bdry_order=3, dg_type='P1'):
        """
        Initialize a tetrahedral mesh with custom ranges by dividing each hexahedron into 6 tetrahedrons.

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
            quad_order: Order of Gauss quadrature (1, 2, or 3 for degree 1, 2, 3).
            bdry_order: Order of boundary quadrature (number of points per face direction, default: 3).
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
        self.quad_order = quad_order
        self.bdry_order = bdry_order
        self.ht = (self.t_max - self.t_min) / self.N_t
        self.hx = (self.x_max - self.x_min) / self.N_x
        self.hy = (self.y_max - self.y_min) / self.N_y
        self.vol_per_tetra = self.ht * self.hx * self.hy / 6.0  # Volume of each tetrahedron
        self.hex_partitions = self._generate_hexahedra()
        self.N_hex = N_t * N_x * N_y
        self.N_elements = self.N_hex * 6  # 6 tetrahedrons per hexahedron
        self.partitions = self._generate_tetrahedra()
        if dg_type == 'P1':
            self.p_dof = 4
        else:
            raise ValueError("Only 'P1' dg_type supported for tetrahedral elements.")
        self.coeffs = jnp.zeros((self.N_elements, self.p_dof))
        self.quad_points, self.quad_weights, self.quad_element_indices, self.quad_points_parti, self.quad_weights_parti = self._setup_quadrature()
        # self.bdry_points_parti, self.bdry_weights_parti = self.get_boundary_quadrature()
        # self.normals = self.compute_normals()
        

    def _generate_hexahedra(self):
        """Generate the hexahedra for the mesh with custom ranges."""
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

    def _generate_tetrahedra(self):
        """Generate the tetrahedra by dividing each hexahedron into 6 tetrahedrons."""
        tetra_indices = jnp.array([[0,2,3,7], [0,3,1,7], [0,1,5,7], [0,5,4,7], [0,4,6,7], [0,6,2,7]])
        tiled_indices = jnp.repeat(tetra_indices[None, :, :], self.N_hex, axis=0)  # (N_hex, 6, 4)

        def select_tetra(hex_p, indices):
            return hex_p[indices]

        tet_part = vmap(select_tetra)(self.hex_partitions, tiled_indices)  # (N_hex, 6, 4, 3)
        return tet_part.reshape(self.N_elements, 4, 3)

    def _setup_quadrature(self):
        """
        Setup quadrature points and weights for tetrahedral mesh using barycentric coordinates.

        Returns:
            quad_points: Quadrature points, shape (N_quad, 3).
            quad_weights: Quadrature weights, shape (N_quad,).
            quad_element_indices: Indices of tetrahedra, shape (N_quad,).
            quad_points_parti: Quadrature points per tetrahedron, shape (N_elements, self.Nq, 3).
            quad_weights_parti: Quadrature weights per tetrahedron, shape (N_elements, self.Nq).
        """
        if self.quad_order == 1:
            # self.Nq = 1
            # bary_points = jnp.array([[0.25, 0.25, 0.25, 0.25]])
            # norm_weights = jnp.array([1.0])
            self.Nq = 4
            bary_points = jnp.eye(4)
            norm_weights = jnp.array([0.25, 0.25, 0.25, 0.25])
        elif self.quad_order == 2:
            self.Nq = 4
            a = (1.0 + jnp.sqrt(1.8)) / 4.0
            b = (1.0 - a) / 3.0
            bary_points = jnp.array([
                [a, b, b, b],
                [b, a, b, b],
                [b, b, a, b],
                [b, b, b, a]
            ])
            norm_weights = jnp.array([0.25, 0.25, 0.25, 0.25])
        elif self.quad_order == 3:
            self.Nq = 8
            alpha = 0.0585540713266
            beta = 0.263112829241
            # Optimized weights to minimize moment errors for better accuracy
            w1_per = 0.07534533344104577
            w2_per = 0.17465466655895423
            def get_orbit(p):
                large = 1.0 - 3.0 * p
                small = p
                orbit = []
                for i in range(4):
                    bary = jnp.full(4, small)
                    bary = bary.at[i].set(large)
                    orbit.append(bary)
                return jnp.stack(orbit)
            orbit1 = get_orbit(alpha)
            orbit2 = get_orbit(beta)
            bary_points = jnp.vstack([orbit1, orbit2])
            norm_weights = jnp.array([w1_per] * 4 + [w2_per] * 4)
        elif self.quad_order == 'composite':
            self.Nq = 10
            bary_points = jnp.array([
                [1.0, 0.0, 0.0, 0.0],  # V0
                [0.0, 1.0, 0.0, 0.0],  # V1
                [0.0, 0.0, 1.0, 0.0],  # V2
                [0.0, 0.0, 0.0, 1.0],  # V3
                [0.5, 0.5, 0.0, 0.0],  # m_V0V1
                [0.0, 0.0, 0.5, 0.5],  # m_V2V3
                [0.5, 0.0, 0.5, 0.0],  # m_V0V2
                [0.5, 0.0, 0.0, 0.5],  # m_V0V3
                [0.0, 0.5, 0.5, 0.0],  # m_V1V2
                [0.0, 0.5, 0.0, 0.5],  # m_V1V3
            ])
            norm_weights = jnp.array([
                1.0/32, 1.0/32, 1.0/32, 1.0/32,  # vertices
                6.0/32, 6.0/32,  # poles
                4.0/32, 4.0/32, 4.0/32, 4.0/32   # equatorial
            ])
        else:
            raise ValueError("quad_order must be 1, 2, 3 or composite.")

        def map_to_tetra(tet):
            # tet shape (4, 3), bary_points (self.Nq, 4)
            points = jnp.einsum('qi,ij->qj', bary_points, tet)
            return points

        quad_points_parti = vmap(map_to_tetra)(self.partitions)  # (N_elements, self.Nq, 3)

        quad_weights_parti = jnp.full((self.N_elements, self.Nq), self.vol_per_tetra) * norm_weights[None, :]

        # Flatten for global quadrature
        quad_points = quad_points_parti.reshape(-1, 3)
        quad_weights = quad_weights_parti.reshape(-1)
        quad_element_indices = jnp.repeat(jnp.arange(self.N_elements), self.Nq)

        return quad_points, quad_weights, quad_element_indices, quad_points_parti, quad_weights_parti