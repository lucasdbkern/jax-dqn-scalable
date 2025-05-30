from src.agent import DQNAgent
from src.environment import CartPoleWrapper
import matplotlib.pyplot as plt
import numpy as np

def train():
    # Create environment and agent
    env = CartPoleWrapper()
    agent = DQNAgent(state_dim=4, action_dim=2)

    # Tracking list
    episode_rewards = []
    episode_lengths = []
    training_losses = []
    
    # Training loop
    for episode in range(1000):
        print(f"Starting episode {episode}")
        state = env.reset()
        episode_reward_score = 0
        episode_length = 0 

        while True:
            action = agent.select_action(state) # select action
            next_state, reward, terminated, truncated, info = env.step(action) #take action
            agent.store_experience(state, action, reward, next_state, terminated, truncated)

            loss = agent.train_step()
            if loss is not None:
                training_losses.append(float(loss))

            state = next_state
            episode_reward_score += reward
            episode_length += 1
            
            if terminated or truncated:
                print(f"Episode {episode} ended with reward {episode_reward_score}")
                break

        episode_rewards.append(episode_reward_score)
        episode_lengths.append(episode_length)


    print("Training completed!")
    plot_results(episode_rewards, training_losses)

def plot_results(episode_rewards, training_losses):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Episode rewards
    ax1.plot(episode_rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    
    # Moving average
    window = 50
    moving_avg = [np.mean(episode_rewards[max(0, i-window):i+1]) 
                  for i in range(len(episode_rewards))]
    ax2.plot(moving_avg)
    ax2.set_title('Moving Average Reward (50 episodes)')
    ax2.axhline(y=195, color='r', linestyle='--', label='Solved threshold')
    ax2.legend()
    
    # Training loss
    if training_losses:
        ax3.plot(training_losses)
        ax3.set_title('Training Loss')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Loss')
    
    # Success rate
    success_rate = [np.mean([r >= 195 for r in episode_rewards[max(0, i-100):i+1]]) 
                    for i in range(len(episode_rewards))]
    ax4.plot(success_rate)
    ax4.set_title('Success Rate (last 100 episodes)')
    ax4.set_ylabel('Fraction > 195 reward')
    
    plt.tight_layout()
    plt.savefig('results/training_results.png')
    plt.show()

if __name__ == "__main__":
    train()