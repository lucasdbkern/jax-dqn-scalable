# DQN network definition using Flax

from flax import linen as nn

class DQN(nn.Module):
    action_dim: int
    hidden_dims: tuple = (128, 128)

    @nn.compact
    def __call__(self, x):
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x
