import jax.numpy as jnp
from jax import random

class MLP:
    """Simple MLP for approximating functions without Flax, adapted for variable input dimensions."""
    def __init__(self, input_dim, hidden_dims=(64, 64), activation=jnp.tanh, lb=None, ub=None, normalize_input=True, seed=0):
        """
        Initialize the MLP with custom input dimension, hidden layers, and activation function.

        Args:
            input_dim: Integer specifying the input dimension.
            hidden_dims: Tuple of integers specifying the size of each hidden layer.
            activation: Callable activation function (default: jnp.tanh).
            lb: Lower bound for input normalization (shape: (input_dim,)).
            ub: Upper bound for input normalization (shape: (input_dim,)).
            normalize_input: Boolean to enable/disable input normalization.
            seed: Random seed for initialization.
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.normalize_input = normalize_input
        if normalize_input:
            self.lb = lb if lb is not None else jnp.zeros((input_dim,))
            self.ub = ub if ub is not None else jnp.ones((input_dim,))
        self.key = random.PRNGKey(seed)
        self.params = self.initialize_params()

    def initialize_params(self):
        """Initialize parameters (weights, biases, and scalar 'a') using Xavier initialization."""
        params = []
        num_layers = len(self.hidden_dims) + 1  # Hidden layers + output layer
        keys = random.split(self.key, num_layers)
        
        # Initialize trainable scalar 'a'
        a = jnp.sqrt(0.01)
        params.append(a)
        
        # Layers from input to output
        layers = [self.input_dim] + list(self.hidden_dims) + [1]  # Output dim = 1 for regression
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

    def forward(self, params, x, extra_op=False):
        """
        Compute the forward pass of the MLP.

        Args:
            params: List containing [a, (W1, b1), (W2, b2), ..., (Wn, bn)].
            x: Input array of shape (batch_size, input_dim) or (input_dim,).

        Returns:
            Scalar output after squeezing.
        """
        if self.normalize_input:
            x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        
        a = params[0]
        for l in range(1, len(params) - 1):
            W, b = params[l]
            x = jnp.dot(x, W.T) + b
            x = self.activation(10.0 * a * x)
        
        W, b = params[-1]
        y = jnp.dot(x, W.T) + b
        out_wo_layer = x
        out = jnp.squeeze(y)
        if extra_op:
            return out, out_wo_layer
        else:
            return out
    

    def __call__(self, params, x):
        """Convenience method to match original interface."""
        return self.forward(params, x)

class MG_NN:
    """Multi-Grade Neural Network for regression, stacking shallow MLPs."""
    def __init__(self, input_dim, hidden_dims_list, activation=jnp.tanh, lb=None, ub=None, seed=0):
        """
        Initialize the MG_NN with input dimension and a list of hidden dimensions for each grade.

        Args:
            input_dim: Integer specifying the input dimension of the first grade.
            hidden_dims_list: List of tuples, each specifying hidden_dims for a grade.
            activation: Callable activation function (default: jnp.tanh).
            lb: Lower bound for input normalization in the first grade.
            ub: Upper bound for input normalization in the first grade.
            seed: Random seed for initialization.
        """
        self.input_dim = input_dim
        self.hidden_dims_list = hidden_dims_list
        self.activation = activation
        self.lb = lb
        self.ub = ub
        self.seed = seed
        self.mlps = []
        self.current_grade = 0

    def add_grade(self):
        """Add a new grade to the network."""
        if self.current_grade == 0:
            # First grade: takes original input_dim, with normalization
            mlp = MLP(input_dim=self.input_dim,
                      hidden_dims=self.hidden_dims_list[self.current_grade],
                      activation=self.activation,
                      lb=self.lb,
                      ub=self.ub,
                      normalize_input=True,
                      seed=self.seed)
        else:
            # Subsequent grades: input_dim is the last hidden dim of the previous grade, no normalization
            last_hidden_dim = self.hidden_dims_list[self.current_grade - 1][-1]
            mlp = MLP(input_dim=last_hidden_dim,
                      hidden_dims=self.hidden_dims_list[self.current_grade],
                      activation=self.activation,
                      normalize_input=False,
                      seed=self.seed)
        self.mlps.append(mlp)
        self.current_grade += 1

    def forward(self, x):
        """
        Compute the forward pass by summing the outputs of all grades.

        Args:
            x: Input array of shape (batch_size, input_dim) or (input_dim,).

        Returns:
            Scalar output representing the sum of all grades' predictions.
        """
        temp = x
        sum_out = 0
        for mlp in self.mlps:
            out, out_wo_layer = mlp.forward(mlp.params, temp, extra_op=True)
            sum_out += out
            temp = out_wo_layer
        return sum_out

# Example usage for a regression problem (to be implemented externally):
# 1. Initialize MG_NN with input_dim=2 and hidden_dims_list=[(64,64), (32,32)]
# 2. Add grade 1, train on (x, y) to approximate f(x)
# 3. Compute residual e1 = y - MG_NN.forward(x)
# 4. Add grade 2, train on (MG_NN.mlps[0].forward(x), e1)
# 5. Evaluate final prediction MG_NN.forward(x)