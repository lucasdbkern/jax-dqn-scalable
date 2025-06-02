import jax
import jax.numpy as jnp
import optax
from src.dqn_network import DQN
from src.replay_buffer import ReplayBuffer
import numpy as np

class DQNAgent:
    """
    Deep Q-Network (DQN) agent with target network and experience replay.
    
    Implements the DQN algorithm from "Human-level control through deep 
    reinforcement learning" (Mnih et al., 2015) with target network 
    stabilization for improved training stability.
    
    Args:
        state_dim (int): Dimension of the observation space
        action_dim (int): Number of discrete actions
        learning_rate (float): Learning rate for the Q-network optimizer
        buffer_capacity (int): Maximum size of the replay buffer
        gamma (float): Discount factor for future rewards
        epsilon (float): Exploration rate for epsilon-greedy policy
        
    Attributes:
        network: JAX/Flax neural network for Q-value approximation
        target_params: Parameters of the target network
        step_count: Number of training steps completed
    """
     
    def __init__(self, state_dim, action_dim, learning_rate=0.001408, buffer_capacity=42976, gamma=0.9937, epsilon=0.0130):
        # set up
        self.network = DQN(action_dim)
        self.replay_buffer = ReplayBuffer(capacity = buffer_capacity) #some high number?
        self.optimizer = optax.adam(learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon

        # initialising network parameters
        key = jax.random.PRNGKey(0)
        
        dummy_input = jnp.zeros((state_dim), dtype=jnp.float32)
        self.params = self.network.init(key, dummy_input)
        self.opt_state = self.optimizer.init(self.params)

        # add target network
        self.target_params = self.params.copy()
        self.step_count = 0
        self.target_update_freq = 500

        # store
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state):
        if np.random.uniform()< self.epsilon:
            action = np.random.choice(self.action_dim)
        else:
            q_values = self.network.apply(self.params, state)  
            action = np.argmax(q_values)
        return int(action)

    def store_experience(self, state, action, reward, next_state, terminated, truncated):
        self.replay_buffer.add(state, action, reward, next_state, terminated, truncated)

    
    def train_step(self, batch_size=32):
        """
    Performs one training step using a batch from the replay buffer.
    
    Samples a batch of experiences and updates the Q-network using the
    Bellman equation with target network for stable learning.
    
    Args:
        batch_size (int): Number of experiences to sample for training
        
    Returns:
        float: Training loss value, or None if insufficient data
    """
        # sample batch
        if len(self.replay_buffer) < batch_size:
            return         
        batch = self.replay_buffer.sample(batch_size)

        # unpack batch 
        states = jnp.array([experience[0] for experience in batch]).astype(jnp.float32)
        actions = jnp.array([experience[1] for experience in batch], dtype=jnp.int32)
        rewards = jnp.array([experience[2] for experience in batch],dtype=jnp.float32)
        next_states = jnp.array([experience[3] for experience in batch]).astype(jnp.float32)
        terminated = jnp.array([experience[4] for experience in batch],dtype=jnp.float32) 
        truncated = jnp.array([experience[5] for experience in batch], dtype=jnp.float32) 

      
        def loss_function(params): #compute how far off the Q-network is from the Bellman target.
            current_q_values = self.network.apply(params, states) #q value of all actions
            current_q = current_q_values[jnp.arange(batch_size), actions] # q values of taken actions


            next_q_values = self.network.apply(self.target_params, next_states)

            #next_q_values = self.network.apply(params, next_states)
            max_next_q_value = jnp.max(next_q_values, axis=1)

            # Q-learning target: 
            target_q = rewards + self.gamma * max_next_q_value * (1-terminated) 
                            
            # mean loss 
            loss = jnp.mean((current_q - target_q)**2)
            return loss
    
        # Compute gradients and update
        loss_value, grads = jax.value_and_grad(loss_function)(self.params) #  computes the loss and its gradient with respect to the network parameters
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state) # uses the optimizer to convert gradients into update steps 
        self.params = optax.apply_updates(self.params, updates) # apply updates to adjust model parameters 

        # Increment step counter and update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_params = self.params.copy()  # Simple copy!

        return loss_value

    

  


        
