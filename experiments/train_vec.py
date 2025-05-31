from src.vect_env import VectorizedCartPole
from src.agent import DQNAgent
import matplotlib.pyplot as plt
import numpy as np


def train():    
    # Create vectorized environment and agent
    env = VectorizedCartPole(num_envs=8)
    agent = DQNAgent(state_dim=4, action_dim=2)
    
    states = env.reset()
    episode_rewards_current = [0] * 8  
    episode_lengths_current = [0] * 8  
    
    # Overall tracking
    completed_episodes = []
    training_losses = []
    completed_episode_count = 0
    step = 0
    
    while completed_episode_count < 1000:
        # Get actions for all 8 environments
        actions = []
        for i in range(8):
            action = agent.select_action(states[i])
            actions.append(int(action))
        
        # Step all environments
        next_states, rewards, terminated, truncated = env.step(actions)
        
        # Store experiences for each environment
        for i in range(8):
            agent.store_experience(states[i], actions[i], rewards[i], 
                                 next_states[i], terminated[i], truncated[i])
            
            # Update episode tracking
            episode_rewards_current[i] += rewards[i] 
            episode_lengths_current[i] += 1
            
            # Check if episode ended for this environment
            if terminated[i] or truncated[i]:
                completed_episodes.append(episode_rewards_current[i])
                completed_episode_count += 1
                print(f"Episode {completed_episode_count} ended with reward {episode_rewards_current[i]}")
                
                # Reset tracking for this environment
                episode_rewards_current[i] = 0
                episode_lengths_current[i] = 0
                
                # Break if we've completed 1000 episodes
                if completed_episode_count >= 1000:
                    break
        
        # Break outer loop if we've completed 1000 episodes
        if completed_episode_count >= 1000:
            break
            
        # Train agent 
        loss = agent.train_step()
        if loss is not None:
            training_losses.append(float(loss))
        
        # Update states
        states = next_states
        step += 1
        
        # Print progress every 1000 steps
        if step % 1000 == 0:
            recent_rewards = completed_episodes[-50:] if len(completed_episodes) >= 50 else completed_episodes
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            print(f"Step {step}: Completed {completed_episode_count} episodes, Avg reward: {avg_reward:.1f}")
    
    print("Training completed!")
    plot_results(completed_episodes, training_losses)

def plot_results(episode_rewards, training_losses):
    # Function stays exactly the same!
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