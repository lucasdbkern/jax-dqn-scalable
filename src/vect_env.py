import gymnasium as gym
import jax.numpy as jnp

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
        states = []
        for env in self.envs:
            state, info = env.reset()
            states.append(state)

        return jnp.array(states)

       
    
    def step(self, actions):
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