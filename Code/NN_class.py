import jax.numpy as jnp
import flax.linen as nn

# Neural Network Definition using Flax
class MLP(nn.Module):
    """Simple MLP for approximating u_theta(t, x)."""
    hidden_dims: list = (64, 64)

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x.squeeze()