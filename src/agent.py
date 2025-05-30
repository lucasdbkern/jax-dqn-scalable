import jax
import jax.numpy as jnp
import optax
from src.dqn_network import DQN
from src.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate = 1e-3):
        # set up
        self.network = DQN(action_dim)
        self.replay_buffer = ReplayBuffer(capacity = 2000) #some high number?
        self.optimizer = optax.adam(learning_rate)

        # initialising network parameters
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((state_dim))
        self.params = self.network.init(key, dummy_input)
        self.opt_state = self.optimizer.init(self.params)

        # store
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state, epsilon=0.1):
        if jnp.random.uniform()< epsilon:
            action = jnp.random.choice(self.action_dim)
        else:
            q_values = self.network.apply(self.params, state)  
            action = jnp.argmax(q_values)
        return action

    def store_experience(self, state, action, reward, next_state, terminated, truncated):
        self.replay_buffer.add(state, action, reward, next_state, terminated, truncated)

    
    def train_step(self, batch_size=32):
        # sample batch
        if len(self.replay_buffer) < batch_size:
            return         
        batch = self.replay_buffer.sample(batch_size)

        # unpack batch 
        states = jnp.array([experience[0] for experience in batch])
        actions = jnp.array([experience[1] for experience in batch])
        rewards = jnp.array([experience[2] for experience in batch])
        next_states = jnp.array([experience[3] for experience in batch])
        terminated = jnp.array([experience[4] for experience in batch])
        truncated = jnp.array([experience[5] for experience in batch])

      
        def loss_function(params): #compute how far off the Q-network is from the Bellman target.
            current_q_values = self.network.apply(params, states) #q value of all actions
            current_q = current_q_values[jnp.arange(batch_size), actions] # q values of taken actions

            next_q_values = self.network.apply(params, next_states) # second forward pass 
            max_next_q_value = jnp.max(next_q_values, axis=1)

            # Q-learning target: 
            gamma = 0.99 # future discount factor
            target_q = rewards + gamma * max_next_q_value * (1-terminated) 
                            
            # mean loss 
            loss = jnp.mean((current_q - target_q)**2)
            return loss
    
        # Compute gradients and update
        loss_value, grads = jax.value_and_grad(loss_function)(self.params) #  computes the loss and its gradient with respect to the network parameters
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state) # uses the optimizer to convert gradients into update steps 
        self.params = optax.apply_updates(self.params, updates) # apply updates to adjust model parameters 

        return loss_value

    

  


        
