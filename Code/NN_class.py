import jax.numpy as jnp
import flax.linen as nn
import jax.random as random
import numpy as np

# class MLP(nn.Module):
#     """MLP with trainable scaling parameter a and flexible activation function."""
#     hidden_dims: list = (64, 64)  # List of hidden layer dimensions
#     activation: callable = nn.tanh  # Default activation is tanh to match net_u_tx
#     lb: jnp.ndarray = None  # Lower bound for input normalization
#     ub: jnp.ndarray = None  # Upper bound for input normalization
    
#     def setup(self):
#         self.a = self.param('a', lambda key: jnp.sqrt(0.01))
#         # Assign unique names to each Dense layer in the list
#         self.layers = [nn.Dense(dim, name=f'dense_{i}') for i, dim in enumerate(self.hidden_dims)]
#         self.output_layer = nn.Dense(1, name='output')

#     def __call__(self, x):
#         if self.lb is not None and self.ub is not None:
#             x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
#         else:
#             raise ValueError("Bounds lb and ub must be provided for normalization.")
#         for layer in self.layers:
#             z = layer(x)
#             x = self.activation(10.0 * self.a * z)
#         x = self.output_layer(x)
#         return x.squeeze()


class MLP:
    """Simple MLP for approximating u_theta(t, x) without Flax."""
    def __init__(self, hidden_dims=(64, 64), activation=jnp.tanh, lb=None, ub=None, seed=0):
        """
        Initialize the MLP with custom hidden layers and activation function.

        Args:
            hidden_dims: List of integers specifying the size of each hidden layer.
            activation: Callable activation function (default: jnp.tanh).
            lb: Lower bound for input normalization (shape: (d,)).
            ub: Upper bound for input normalization (shape: (d,)).
            seed: Random seed for initialization.
        """
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.lb = lb if lb is not None else jnp.array([0.0, 0.0])  # Default bounds
        self.ub = ub if ub is not None else jnp.array([1.0, 1.0])
        self.key = random.PRNGKey(seed)
        self.params = self.initialize_params()

    def initialize_params(self):
        """Initialize parameters (weights, biases, and scalar 'a') using Xavier initialization."""
        params = []
        num_layers = len(self.hidden_dims) + 1  # Hidden layers + output layer
        keys = random.split(self.key, num_layers)
        
        # Initialize trainable scalar 'a'
        a = jnp.sqrt(0.01)  # Same as original
        params.append(a)
        
        # Input layer to first hidden layer
        layers = [2] + list(self.hidden_dims) + [1]  # Input dim = 2, output dim = 1
        for l in range(num_layers):
            W = self.xavier_init((layers[l], layers[l+1]), keys[l])
            b = jnp.zeros((layers[l+1],), dtype=jnp.float32)
            params.append((W, b))
        return params

    def xavier_init(self, size, key):
        """Xavier initialization for weights."""
        in_dim, out_dim = size
        xavier_stddev = jnp.sqrt(2 / (in_dim + out_dim))
        return random.truncated_normal(key, -2, 2, shape=(out_dim, in_dim), dtype=jnp.float32) * xavier_stddev

    def forward(self, params, x):
        """
        Compute the forward pass of the MLP.

        Args:
            params: List containing [a, (W1, b1), (W2, b2), ..., (Wn, bn)].
            x: Input array of shape (batch_size, 2) or (2,).

        Returns:
            Scalar output after squeezing.
        """
        # Normalize input
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        
        # Extract trainable scalar 'a'
        a = params[0]
        
        # Process hidden layers
        for l in range(1, len(params) - 1):
            W, b = params[l]
            x = jnp.dot(x, W.T) + b
            x = self.activation(10.0 * a * x)  # Apply scaling with 'a'
        
        # Output layer
        W, b = params[-1]
        y = jnp.dot(x, W.T) + b
        return jnp.squeeze(y)

    def __call__(self, params, x):
        """Convenience method to match original interface."""
        return self.forward(params, x)