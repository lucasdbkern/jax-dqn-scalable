# a hyperparameter optimizer for the following parameters:
# from DQN Agent: learning rate and capacity
# from train step: gamma and batch size 
# from select_action: epsilon (exploration rate)
# and batch size
#---------------------------------------------------------

import optuna 
import sys
sys.path.append('/Users/lucaskern/Desktop/Desktop/Github_repos/JAX_DQN_scalable/jax-dqn-scalable/src')
from src.agent import DQNAgent
from src.vect_env import VectorizedCartPole
import numpy as np
import matplotlib.pyplot as plt 

def plot_optimization_results(study):
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get trial data
    trials = study.trials
    trial_numbers = [t.number for t in trials if t.value is not None]
    trial_values = [t.value for t in trials if t.value is not None]
    
    # Create plots
    plt.figure(figsize=(14, 10))
    
    # 1. Optimization Progress
    plt.subplot(2, 2, 1)
    plt.plot(trial_numbers, trial_values, 'b-o', alpha=0.7, markersize=6)
    plt.axhline(y=study.best_value, color='r', linestyle='--', linewidth=2,
               label=f'Best: {study.best_value:.2f}')
    plt.xlabel('Trial Number')
    plt.ylabel('Performance Score')
    plt.title('Optimization Progress', fontsize=14, fontweight='bold')
    plt.legend()
    
    # 2. Learning Rate vs Performance
    plt.subplot(2, 2, 2)
    learning_rates = [t.params['learning_rate'] for t in trials if t.value is not None]
    plt.scatter(learning_rates, trial_values, alpha=0.6, s=50, c='skyblue', edgecolors='navy')
    plt.xscale('log')
    
    # Force specific x-axis ticks for learning rate
    lr_ticks = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    plt.xticks(lr_ticks, ['1e-4', '2e-4', '5e-4', '1e-3', '2e-3', '5e-3', '1e-2'])
    
    plt.xlabel('Learning Rate')
    plt.ylabel('Performance Score')
    plt.title('Learning Rate vs Performance', fontsize=14, fontweight='bold')
    
    # 3. Spider/Radar Chart for Best Parameters
    plt.subplot(2, 2, 3, projection='polar')
    
    best_params = study.best_params
    param_names = list(best_params.keys())
    param_values = list(best_params.values())
    
    # Normalize each parameter to 0-1 scale for the spider chart
    normalized_values = []
    display_values = []  # For showing actual values
    
    for name, value in zip(param_names, param_values):
        if name == 'learning_rate':
            # Scale from 1e-4 to 1e-2 → 0 to 1
            normalized = (np.log10(value) - np.log10(1e-4)) / (np.log10(1e-2) - np.log10(1e-4))
            # Format as 10^-3 style instead of e notation
            exp = int(np.log10(value))
            coeff = value / (10 ** exp)
            if coeff == 1.0:
                display_values.append(f'10^{exp}')
            else:
                display_values.append(f'{coeff:.1f}×10^{exp}')
        elif name == 'gamma':
            # Scale from 0.95 to 0.999 → 0 to 1
            normalized = (value - 0.95) / (0.999 - 0.95)
            display_values.append(f'{value:.3f}')
        elif name == 'epsilon':
            # Scale from 0.01 to 0.3 → 0 to 1
            normalized = (value - 0.01) / (0.3 - 0.01)
            display_values.append(f'{value:.3f}')
        elif name == 'buffer_capacity':
            # Scale from 5000 to 50000 → 0 to 1
            normalized = (value - 5000) / (50000 - 5000)
            display_values.append(f'{int(value)}')
        elif name == 'batch_size':
            # Map categories to positions: 16→0, 32→0.33, 64→0.67, 128→1
            batch_map = {16: 0, 32: 0.33, 64: 0.67, 128: 1.0}
            normalized = batch_map.get(value, 0.5)
            display_values.append(f'{int(value)}')
        else:
            normalized = 0.5  # Default fallback
            display_values.append(str(value))
        
        normalized_values.append(normalized)
    
    # Create angles for each parameter
    angles = np.linspace(0, 2*np.pi, len(param_names), endpoint=False)
    
    # Close the plot by adding the first value at the end
    normalized_values += [normalized_values[0]]
    angles = np.concatenate([angles, [angles[0]]])
    
    # Plot the spider chart
    plt.plot(angles, normalized_values, 'o-', linewidth=2, color='blue', alpha=0.7)
    plt.fill(angles, normalized_values, alpha=0.25, color='lightblue')
    
    # Add parameter names and actual values
    plt.xticks(angles[:-1], [f'{name}\n({val})' for name, val in zip(param_names, display_values)], fontsize=9)
    
    # Set y-axis limits and labels
    plt.ylim(0, 1)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['20%', '40%', '60%', '80%', '100%'], fontsize=8)
    
    plt.title(f'Best Parameters (Score: {study.best_value:.1f})', fontsize=12, fontweight='bold', pad=20)
    
    # 4. Performance Distribution
    plt.subplot(2, 2, 4)
    # Use integer bins for integer trial counts
    n_bins = min(8, len(trial_values))
    plt.hist(trial_values, bins=n_bins, alpha=0.7, 
             color='lightgreen', edgecolor='darkgreen', linewidth=1.5)
    plt.xlabel('Performance Score')
    plt.ylabel('Number of Trials')
    plt.title(f'Performance Distribution (Best: {study.best_value:.1f})', fontsize=14, fontweight='bold')
    # Remove grid and awkward red line
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed summary to console
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"Number of trials completed: {len(trials)}")
    print(f"Best performance achieved: {study.best_value:.2f}")
    print("\nBest hyperparameters:")
    for param, value in study.best_params.items():
        if isinstance(value, float):
            if param == 'learning_rate':
                print(f"  {param:<20}: {value:.6f}")
            else:
                print(f"  {param:<20}: {value:.4f}")
        else:
            print(f"  {param:<20}: {value}")
    
    # Performance statistics
    if len(trial_values) > 1:
        print(f"\nPerformance statistics:")
        print(f"  Mean performance: {np.mean(trial_values):.2f}")
        print(f"  Std deviation:    {np.std(trial_values):.2f}")
        print(f"  Min performance:  {np.min(trial_values):.2f}")
        print(f"  Max performance:  {np.max(trial_values):.2f}")
    
    print("="*60)

def hyperparamter_training(trial):
    # hyperparams to optimise incl. their ranges
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    epsilon = trial.suggest_float("epsilon", 0.01, 0.3) 
    buffer_capacity = trial.suggest_int("buffer_capacity", 5000, 50000)    
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    env = VectorizedCartPole(num_envs=4)
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
    performance = np.mean(recent_rewards) if recent_rewards else 0

    
    import gc
    gc.collect()  # Python garbage collection
    
    return performance 




def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(hyperparamter_training, n_trials = 20)


    print("Best params:", study.best_params)
    print("Best performance:", study.best_value)

    plot_optimization_results(study)


    return study

if __name__ == "__main__":
    main()
