from src.agent import DQNAgent
from src.environment import CartPoleWrapper

def train():
    # Create environment and agent
    env = CartPoleWrapper()
    agent = DQNAgent(state_dim=4, action_dim=2)
    
    # Training loop
    for episode in range(1000):
        state = env.reset()
        reward_score = 0
        done = False

        while True:
            action = agent.select_action(state) # select action
            next_state, reward, terminated, truncated, info = env.step(action) #take action
            agent.store_experience(state, action, reward, next_state, terminated, truncated)
            agent.train_step()

            state = next_state
            reward_score += reward
            
            if terminated or truncated:  
                break

        print(f"Episode {episode}: Reward = {reward_score}")

if __name__ == "__main__":
    train()