import jax.numpy as jnp 
import gymnasium as gym

class CartPoleWrapper:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state_dim= 4
        self.action_dim = 2

    def reset(self):
        observation, info = self.env.reset()
        return observation
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated  
        return obs, reward, done, info

