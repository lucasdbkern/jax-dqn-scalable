print("Script started")


import jax
import jax.numpy as jnp
from src.dqn_network import DQN

# Create a fake CartPole state (4 numbers)
dummy_state = jnp.array(
    [0.1, 0.2, 0.3, 0.4]
)  # -->  array entries represent: Cart Position, Cart Velocity, Pole Angle and Pole Angular Velocity

# Create network for CartPole (2 actions: left, right)
network = DQN(action_dim=2)

# Initialize network parameters
key = jax.random.PRNGKey(0)
params = network.init(key, dummy_state)

# Test: Get Q-values from network
q_values = network.apply(params, dummy_state)

# Check results
print(f"Input shape: {dummy_state.shape}")  # Should be (4,)
print(f"Output shape: {q_values.shape}")  # Should be (2,)
print(f"Q-values: {q_values}")  # Should be 2 numbers
