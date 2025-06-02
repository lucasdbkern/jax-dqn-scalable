import gymnasium as gym
import jax.numpy as jnp

"""
    Vectorized environment wrapper for parallel training across multiple environments.
    
    Creates multiple independent copies of an environment and steps them in parallel
    to increase data collection efficiency and training throughput. Each environment
    maintains its own state and can reset independently when episodes terminate.
    
    This implementation follows the Gymnasium API and handles both natural episode
    termination and time-based truncation appropriately for each environment.
    
    Args:
        env_name (str): Name of the Gymnasium environment (e.g., "CartPole-v1")
        n_envs (int): Number of parallel environments to create
        seed (int, optional): Base random seed for reproducibility. Each
            environment gets seed + env_index. Defaults to None.
            
    Attributes:
        n_envs (int): Number of parallel environments
        envs (list): List of individual Gymnasium environments
        single_observation_space: Observation space of a single environment
        single_action_space: Action space of a single environment
        
    Example:
        >>> vec_env = VectorizedEnvironment("CartPole-v1", n_envs=8)
        >>> observations = vec_env.reset()  # Shape: (8, obs_dim)
        >>> actions = [env.action_space.sample() for _ in range(8)]
        >>> next_obs, rewards, terminated, truncated = vec_env.step(actions)
        >>> vec_env.close()
    """

class VectorizedCartPole:
    def __init__(self, num_envs=8):
        self.num_envs = num_envs
        self.envs = []
        for i in range(num_envs):
            env = gym.make('CartPole-v1')
            env.reset(seed=i)  # ✅ Different seed for each env
            self.envs.append(env)
        self.dones = [False] * num_envs # which nr of envs are done 
    
    def reset(self):
        """
        Reset all environments to their initial states.
        
        Returns:
            jnp.ndarray: Stacked initial observations from all environments
                Shape: (n_envs, *observation_shape)
        """
        states = []
        for env in self.envs:
            state, info = env.reset()
            states.append(state)

        return jnp.array(states)

       
    
    def step(self, actions):
        """
        Step all environments with the given actions.
        
        Automatically resets any environment that terminates or gets truncated,
        ensuring continuous data collection for training.
        
        Args:
            actions (list or jnp.ndarray): Actions for each environment
                Length must equal n_envs
                
        Returns:
            tuple: (next_observations, rewards, terminated, truncated)
                - next_observations (jnp.ndarray): Shape (n_envs, *obs_shape)
                - rewards (jnp.ndarray): Shape (n_envs,)
                - terminated (jnp.ndarray): Boolean array, shape (n_envs,)
                - truncated (jnp.ndarray): Boolean array, shape (n_envs,)
        """
        next_states = []
        rewards = []
        terminated_list = []
        truncated_list = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            next_state, reward, terminated, truncated, info  = env.step(action)

            if truncated or terminated:
                next_state, info = env.reset ()

            next_states.append(next_state)
            rewards.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)

        return (jnp.array(next_states),   
        jnp.array(rewards),        
        jnp.array(terminated_list),
        jnp.array(truncated_list))