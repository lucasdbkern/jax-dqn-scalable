# a hyperparameter optimizer for the following parameters:
# from DQN Agent: learning rate and capacity
# from train step: gamma and batch size 
# from select_action: epsilon (exploration rate)
# and batch size
#---------------------------------------------------------

import optuna 
from src.agent import DQNAgent
from src.replay_buffer import ReplayBuffer
from src.vect_env import VectorizedCartPole
import sys
import numpy as np
sys.path.append('/Users/lucaskern/Desktop/Desktop/Github_repos/JAX_DQN_scalable/jax-dqn-scalable/src')

def hyperparamter_training(trial):
    # hyperparams to optimise incl. their ranges
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    epsilon = trial.suggest_float("epsilon", 0.01, 0.3) 
    buffer_capacity = trial.suggest_int("buffer_capacity", 5000, 50000)    
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    env = VectorizedCartPole(num_envs=8)
    agent = DQNAgent(
    state_dim=4, 
    action_dim=2, 
    learning_rate=learning_rate,
    buffer_capacity=buffer_capacity,
    gamma=gamma,  
    epsilon=epsilon)  
    

    states = env.reset()
    episode_rewards_current = [0] * 8  
    episode_lengths_current = [0] * 8  
    
    # Overall tracking
    completed_episodes = []
    training_losses = []
    completed_episode_count = 0
    step = 0
    
    while completed_episode_count < 200:
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
                if completed_episode_count >= 200:
                    break
        
        # Break outer loop if we've completed 1000 episodes
        if completed_episode_count >= 200:
            break
            
        # Train agent 
        loss = agent.train_step(batch_size = batch_size)
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
    recent_rewards = completed_episodes[-50:] if len(completed_episodes) >= 50 else completed_episodes
    return np.mean(recent_rewards) if recent_rewards else 0




def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(hyperparamter_training, n_trials = 20)


    print("Best params:", study.best_params)
    print("Best performance:", study.best_value)

    plot_optimization_results(study)


    return study

if __name__ == "__main__":
    main()


def plot_optimization_results(study):
    import matplotlib.pyplot as plt
    
    # Get trial data
    trials = study.trials
    trial_numbers = [t.number for t in trials if t.value is not None]
    trial_values = [t.value for t in trials if t.value is not None]
    
    # Create plots
    plt.figure(figsize=(12, 8))
    
    # 1. Optimization history
    plt.subplot(2, 2, 1)
    plt.plot(trial_numbers, trial_values, 'b-o', alpha=0.7)
    plt.axhline(y=study.best_value, color='r', linestyle='--', 
               label=f'Best: {study.best_value:.2f}')
    plt.xlabel('Trial Number')
    plt.ylabel('Performance Score')
    plt.title('Optimization Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Parameter values vs performance
    plt.subplot(2, 2, 2)
    learning_rates = [t.params['learning_rate'] for t in trials if t.value is not None]
    plt.scatter(learning_rates, trial_values, alpha=0.6)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Performance')
    plt.title('Learning Rate vs Performance')
    plt.grid(True, alpha=0.3)
    
    # 3. Best parameters bar chart
    plt.subplot(2, 2, 3)
    best_params = study.best_params
    param_names = list(best_params.keys())
    param_values = list(best_params.values())
    
    plt.barh(param_names, param_values)
    plt.xlabel('Parameter Value')
    plt.title('Best Hyperparameters')
    plt.grid(True, alpha=0.3)
    
    # 4. Trial distribution
    plt.subplot(2, 2, 4)
    plt.hist(trial_values, bins=10, alpha=0.7, edgecolor='black')
    plt.axvline(x=study.best_value, color='r', linestyle='--', 
               label=f'Best: {study.best_value:.2f}')
    plt.xlabel('Performance Score')
    plt.ylabel('Number of Trials')
    plt.title('Performance Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nOptimization Summary:")
    print(f"Number of trials: {len(trials)}")
    print(f"Best performance: {study.best_value:.2f}")
    print(f"Best hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    