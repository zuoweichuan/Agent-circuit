import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from datetime import datetime
import seaborn as sns

class DataLogger:
    def __init__(self, log_dir='logs'):
        # Create log directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, self.timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize data containers
        self.episode_data = []
        self.current_episode = {
            'actions': [],
            'rewards': [],
            'areas': [],
            'mses': [],
            'lines': []
        }
        self.training_data = {
            'episode_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'entropies': [],
            'advantages': [],
            'best_areas': []
        }
        
        # Plot settings
        plt.style.use('ggplot')
        self.colors = sns.color_palette("muted", 8)

        # Add log file
        self.log_file = os.path.join(self.log_dir, "training_log.txt")
        with open(self.log_file, 'w') as f:
            f.write(f"Training start time: {self.timestamp}\n")
            f.write("=" * 50 + "\n")
        
    def log_step(self, action, reward, area, mse, line):
        """Record data for each step"""
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['areas'].append(area)
        self.current_episode['mses'].append(mse)
        self.current_episode['lines'].append(line)
        
    def end_episode(self, total_reward, best_area=None):
        """End current episode and save data"""
        # Check if there's any step data to avoid index errors
        if not self.current_episode['actions']:
            print("Warning: Attempted to end an episode with no step data")
            return
            
        episode_summary = {
            'episode': len(self.episode_data) + 1,
            'total_reward': total_reward,
            'actions': self.current_episode['actions'].copy(),
            'action_counts': np.bincount(self.current_episode['actions'], minlength=8).tolist(),
            'mean_reward': np.mean(self.current_episode['rewards']),
            'final_area': self.current_episode['areas'][-1],
            'final_mse': self.current_episode['mses'][-1],
            'steps': len(self.current_episode['actions'])
        }
        
        self.episode_data.append(episode_summary)
        self.training_data['episode_rewards'].append(total_reward)
        
        if best_area is not None:
            self.training_data['best_areas'].append(best_area)
        
        # Reset current episode data
        self.current_episode = {
            'actions': [],
            'rewards': [],
            'areas': [],
            'mses': [],
            'lines': []
        }
        
        # Save data every 10 episodes
        if len(self.episode_data) % 10 == 0:
            self.save_data()
            
    def log_training(self, actor_loss, critic_loss, entropy, advantage):
        """Record training metrics"""
        self.training_data['actor_losses'].append(float(actor_loss))
        self.training_data['critic_losses'].append(float(critic_loss))
        self.training_data['entropies'].append(float(entropy))
        self.training_data['advantages'].append(float(advantage))
        
    def save_data(self):
        """Save collected data"""
        # Skip if no data to save
        if not self.episode_data:
            print("No data to save")
            return
            
        # Save episode data as CSV
        pd.DataFrame(self.episode_data).to_csv(
            os.path.join(self.log_dir, f"episode_data_{len(self.episode_data)}.csv"), 
            index=False
        )
        
        # Process training data - fix array length issues
        if any(v for v in self.training_data.values()):
            # Find lengths of each array
            lengths = {k: len(v) for k, v in self.training_data.items() if v}
            
            if not lengths:
                return  # Skip if no valid data
                
            # Find minimum length among all arrays
            min_length = min(lengths.values())
            
            # Truncate all arrays to minimum length
            processed_data = {}
            for k, v in self.training_data.items():
                if v:  # Ensure array is not empty
                    processed_data[k] = v[:min_length]
                else:
                    # For empty arrays, create zero arrays
                    processed_data[k] = [0] * min_length if min_length > 0 else []
            
            # Create dataframe and save
            if min_length > 0:  # Only save when data exists
                training_df = pd.DataFrame(processed_data)
                training_df.to_csv(
                    os.path.join(self.log_dir, f"training_data_{len(self.episode_data)}.csv"), 
                    index=False
                )
                print(f"Saved {min_length} training data records")
            
    def plot_all(self):
        """Generate all visualization charts"""
        try:
            if not self.episode_data:
                print("No data to visualize")
                return
                
            self._plot_rewards()
            self._plot_action_distribution()
            self._plot_area_mse()
            if self.training_data['actor_losses']:
                self._plot_training_metrics()
            self._plot_action_trends()
        except Exception as e:
            print(f"Error generating visualization charts: {e}")
            import traceback
            traceback.print_exc()
        
    def _plot_rewards(self):
        """Plot reward trends"""
        plt.figure(figsize=(10, 6))
        rewards = [ep['total_reward'] for ep in self.episode_data]
        plt.plot(rewards, marker='o', linestyle='-', color=self.colors[0])
        plt.title('Episode Total Reward Trend')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        # Add moving average line
        if len(rewards) > 5:
            window_size = min(10, len(rewards)//2)
            plt.plot(np.convolve(rewards, np.ones(window_size)/window_size, mode='valid'), 
                    linestyle='--', color=self.colors[1], label=f'{window_size}-episode moving average')
            plt.legend()
        plt.savefig(os.path.join(self.log_dir, "rewards_trend.png"), dpi=200)
        plt.close()
        
    def _plot_action_distribution(self):
        """Plot action distribution"""
        plt.figure(figsize=(10, 6))
        action_counts = np.zeros(8)
        for episode in self.episode_data:
            action_counts += np.array(episode['action_counts'])
            
        # Action labels in English
        action_labels = ['No Operation', 'Delete Left Signal', 'Delete Right Signal', 'Replace with AND', 
                         'Replace with OR', 'Replace with XOR', 'Negate Left Signal', 'Negate Right Signal']
        plt.bar(action_labels, action_counts, color=self.colors)
        plt.title('Action Distribution')
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "action_distribution.png"), dpi=200)
        plt.close()
        
    def _plot_area_mse(self):
        """Plot area and MSE trends"""
        # Extract final area and MSE for each episode
        episodes = range(1, len(self.episode_data) + 1)
        areas = [ep['final_area'] for ep in self.episode_data]
        mses = [ep['final_mse'] for ep in self.episode_data]
        
        # Dual Y-axis plot
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = self.colors[0]
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Area', color=color)
        ax1.plot(episodes, areas, marker='o', linestyle='-', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = self.colors[2]
        ax2.set_ylabel('MSE', color=color)
        ax2.plot(episodes, mses, marker='s', linestyle='-', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Area and MSE Trends')
        fig.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "area_mse_trend.png"), dpi=200)
        plt.close()
        
    def _plot_training_metrics(self):
        """Plot training metrics"""
        if not self.training_data['actor_losses']:
            return  # Skip if no training data
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Actor Loss
        axes[0, 0].plot(self.training_data['actor_losses'], color=self.colors[0])
        axes[0, 0].set_title('Actor Loss')
        axes[0, 0].set_xlabel('Update Steps')
        axes[0, 0].set_ylabel('Loss')
        
        # 2. Critic Loss
        axes[0, 1].plot(self.training_data['critic_losses'], color=self.colors[1])
        axes[0, 1].set_title('Critic Loss')
        axes[0, 1].set_xlabel('Update Steps')
        axes[0, 1].set_ylabel('Loss')
        
        # 3. Entropy
        if self.training_data['entropies']:  # Check if entropy data exists
            axes[1, 0].plot(self.training_data['entropies'], color=self.colors[3])
            axes[1, 0].set_title('Policy Entropy')
            axes[1, 0].set_xlabel('Update Steps')
            axes[1, 0].set_ylabel('Entropy')
        
        # 4. Advantages
        if self.training_data['advantages']:  # Check if advantage data exists
            axes[1, 1].plot(self.training_data['advantages'], color=self.colors[4])
            axes[1, 1].set_title('Advantage Function')
            axes[1, 1].set_xlabel('Update Steps')
            axes[1, 1].set_ylabel('Advantage')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "training_metrics.png"), dpi=200)
        plt.close()
        
    def _plot_action_trends(self):
        """Plot action selection trends"""
        if not self.episode_data:
            return
            
        # Action frequency by episode
        window_size = max(1, min(5, len(self.episode_data) // 3))
        action_freq = np.zeros((len(self.episode_data), 8))
        
        for i, episode in enumerate(self.episode_data):
            total = sum(episode['action_counts'])
            if total > 0:  # Avoid division by zero
                action_freq[i] = np.array(episode['action_counts']) / total
        
        # Smooth curves with moving average
        smoothed_freq = np.zeros_like(action_freq)
        for i in range(8):
            if len(action_freq) > window_size:
                smoothed_freq[:, i] = np.convolve(
                    action_freq[:, i], 
                    np.ones(window_size)/window_size, 
                    mode='same'
                )
            else:
                smoothed_freq[:, i] = action_freq[:, i]
        
        plt.figure(figsize=(12, 8))
        action_labels = ['No Operation', 'Delete Left Signal', 'Delete Right Signal', 'Replace with AND', 
                         'Replace with OR', 'Replace with XOR', 'Negate Left Signal', 'Negate Right Signal']
        
        for i in range(8):
            plt.plot(
                range(len(self.episode_data)), 
                smoothed_freq[:, i], 
                marker='.', 
                linestyle='-', 
                color=self.colors[i], 
                label=action_labels[i]
            )
        
        plt.title('Action Selection Trends (Moving Average)')
        plt.xlabel('Episode')
        plt.ylabel('Selection Frequency')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "action_trends.png"), dpi=200)
        plt.close()