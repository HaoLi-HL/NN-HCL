import jax.numpy as jnp
from jax import vmap
import numpy as np

class PiecewiseLinearPolynomialsHex:
    def __init__(self, N_t, N_x, N_y, t_min=0.0, t_max=1.0, x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, gauss_order=3, bdry_order=3):
        """
        Initialize piecewise linear polynomials on a hexahedral mesh with custom ranges.

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
            gauss_order: Order of Gauss quadrature (number of points per direction, default: 3).
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
        self.gauss_order = gauss_order
        self.bdry_order = bdry_order
        self.N_elements = N_t * N_x * N_y  # Number of hexahedra
        self.partitions = self._generate_hexahedra()
        self.coeffs = jnp.zeros((self.N_elements, 4))  # Coefficients a + b*t + c*x + d*y per hexahedron
        self.quad_points, self.quad_weights, self.element_indices, self.quad_points_parti, self.quad_weights_parti = self._setup_quadrature()
        # self.bdry_points_parti, self.bdry_weights_parti = self.get_boundary_quadrature()
        # self.normals = self.compute_normals()

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
        Setup Gauss quadrature points and weights for hexahedral mesh.

        Returns:
            quad_points: Quadrature points, shape (N_quad, 3).
            quad_weights: Quadrature weights, shape (N_quad,).
            element_indices: Indices of hexahedra, shape (N_quad,).
            quad_points_parti: Quadrature points per hexahedron, shape (N_elements, Nq, 3).
            quad_weights_parti: Quadrature weights per hexahedron, shape (N_elements, Nq).
        """
        Nq = self.gauss_order ** 3  # Number of quadrature points per hexahedron

        # Gauss-Legendre nodes and weights in [-1, 1]
        nodes, weights = np.polynomial.legendre.leggauss(self.gauss_order)
        nodes_jnp = jnp.array(nodes)
        weights_jnp = jnp.array(weights)

        # Tensor product for 3D
        t_nodes, x_nodes, y_nodes = jnp.meshgrid(nodes_jnp, nodes_jnp, nodes_jnp)
        t_weights, x_weights, y_weights = jnp.meshgrid(weights_jnp, weights_jnp, weights_jnp)
        quad_weights_3d = (t_weights * x_weights * y_weights).ravel()

        # Function to map [-1,1]^3 to hexahedron [t_low, t_high]x[x_low, x_high]x[y_low, y_high]
        def map_to_hex(hex):
            t_low, x_low, y_low = hex[0]
            t_high, x_high, y_high = hex[-1]
            t_scale = (t_high - t_low) / 2
            x_scale = (x_high - x_low) / 2
            y_scale = (y_high - y_low) / 2
            t_center = (t_high + t_low) / 2
            x_center = (x_high + x_low) / 2
            y_center = (y_high + y_low) / 2
            t_points = t_center + t_nodes.ravel() * t_scale
            x_points = x_center + x_nodes.ravel() * x_scale
            y_points = y_center + y_nodes.ravel() * y_scale
            points = jnp.stack([t_points, x_points, y_points], axis=1)
            weights = quad_weights_3d * (t_scale * x_scale * y_scale)
            return points, weights

        # Vectorize over hexahedra
        quad_points_parti, quad_weights_parti = vmap(map_to_hex)(self.partitions)

        # Flatten for global quadrature
        quad_points = quad_points_parti.reshape(-1, 3)
        quad_weights = quad_weights_parti.reshape(-1)
        element_indices = jnp.repeat(jnp.arange(self.N_elements), Nq)

        return quad_points, quad_weights, element_indices, quad_points_parti, quad_weights_parti
