import numpy as np

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling agent experiences.
    
    Implements a circular buffer that stores experience tuples (s, a, r, s', 
    terminated, truncated) and provides random sampling for training. This 
    breaks temporal correlations in the data and improves learning stability.
    
    The buffer distinguishes between 'terminated' (episode ended naturally) 
    and 'truncated' (episode ended due to time limits) following Gymnasium 
    conventions for proper temporal difference learning.
    
    Args:
        capacity (int): Maximum number of experiences to store
        obs_dim (int): Dimension of observation space
        action_dim (int): Dimension of action (typically 1 for discrete actions)
        
    Attributes:
        capacity (int): Maximum buffer size
        obs_dim (int): Observation space dimension
        action_dim (int): Action space dimension
        buffer (list): Internal storage for experiences
        position (int): Current position in circular buffer
        size (int): Current number of stored experiences
    """
    def __init__(self, capacity):
        self.capacity = capacity # of the storage/buffer
        self.storage = [] # storage 
        self.position = 0 # where to add next item        
    
    def add(self, state, action, reward, next_state, terminated, truncated):
        """
        Add a new experience to the replay buffer.
        
        If buffer is at capacity, overwrites the oldest experience
        (circular buffer behavior).
        
        Args:
            obs (jnp.ndarray): Current observation
            action (int): Action taken
            reward (float): Reward received
            next_obs (jnp.ndarray): Next observation
            terminated (bool): True if episode ended naturally (e.g., goal reached)
            truncated (bool): True if episode ended due to time limit
        """
        experience = (state, action, reward, next_state, terminated, truncated)

        if len(self.storage) < self.capacity:
            self.storage.append(experience)
        else:
            self.storage[self.position % self.capacity] = experience 
        
        self.position +=1 
    
    def sample(self, batch_size):
        """
        Sample a random batch of experiences from the buffer.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            list: List of experience tuples, each containing 
                (obs, action, reward, next_obs, terminated, truncated)
                
        Raises:
            ValueError: If batch_size > current buffer size
        """
        if len(self.storage) < batch_size:
            batch_size = len(self.storage)

        indices = np.random.choice(len(self.storage), batch_size, replace=False)
        return [self.storage[i] for i in indices]
    

    def __len__(self):
        """
        Get the current number of experiences in the buffer.
        
        Returns:
            int: Number of stored experiences
        """
        return len(self.storage)
    
       