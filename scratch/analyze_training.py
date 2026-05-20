import pandas as pd
import numpy as np
import os

sac_csv_path = r"C:\Users\NeverGonnaGiveYouUp\OneDrive\Рабочий стол\study_materials\Skoltech\Reinforcement_Learning\Project 4 TEST\NBV_with_obstacles_and_robot\kaggle_output\output_odin_sac\sac_metrics (1).csv"
step_csv_path = r"C:\Users\NeverGonnaGiveYouUp\OneDrive\Рабочий стол\study_materials\Skoltech\Reinforcement_Learning\Project 4 TEST\NBV_with_obstacles_and_robot\kaggle_output\output_odin_sac\logs\training_metrics (8).csv"

print("--- SAC Metrics File Analysis ---")
if os.path.exists(sac_csv_path):
    df_sac = pd.read_csv(sac_csv_path)
    total_episodes = len(df_sac)
    print(f"Total episodes in SAC metrics: {total_episodes}")
    print(f"Columns: {list(df_sac.columns)}")
    
    # Analyze rewards
    r_mean = df_sac['ep_reward'].mean()
    r_std = df_sac['ep_reward'].std()
    r_min = df_sac['ep_reward'].min()
    r_max = df_sac['ep_reward'].max()
    print(f"Episode Reward - Mean: {r_mean:.2f}, Std: {r_std:.2f}, Min: {r_min:.2f}, Max: {r_max:.2f}")
    
    # Comparison of first 10% vs last 10%
    n_10 = max(1, int(total_episodes * 0.1))
    first_10_rew = df_sac['ep_reward'].head(n_10).mean()
    last_10_rew = df_sac['ep_reward'].tail(n_10).mean()
    print(f"Mean Reward (First 10%): {first_10_rew:.2f}")
    print(f"Mean Reward (Last 10%): {last_10_rew:.2f}")
    print(f"Reward Improvement: {last_10_rew - first_10_rew:+.2f}")
    
    # Success rate (from 'success' column)
    if 'success' in df_sac.columns:
        first_10_succ = df_sac['success'].head(n_10).mean() * 100
        last_10_succ = df_sac['success'].tail(n_10).mean() * 100
        overall_succ = df_sac['success'].mean() * 100
        print(f"Success Rate - Overall: {overall_succ:.2f}%, First 10%: {first_10_succ:.2f}%, Last 10%: {last_10_succ:.2f}%")
        
    # p_hidden
    if 'p_hidden' in df_sac.columns:
        first_10_p = df_sac['p_hidden'].head(n_10).mean()
        last_10_p = df_sac['p_hidden'].tail(n_10).mean()
        print(f"Final p_hidden (First 10%): {first_10_p:.4f}, (Last 10%): {last_10_p:.4f}")
        
    # Loss trends
    for loss_col in ['critic_loss', 'actor_loss', 'coverage_loss']:
        if loss_col in df_sac.columns:
            first_l = df_sac[loss_col].head(n_10).mean()
            last_l = df_sac[loss_col].tail(n_10).mean()
            print(f"{loss_col} - First 10%: {first_l:.4f}, Last 10%: {last_l:.4f}")
else:
    print("SAC CSV not found.")

print("\n--- Step-by-Step Training Metrics File Analysis ---")
if os.path.exists(step_csv_path):
    df_step = pd.read_csv(step_csv_path)
    total_steps = len(df_step)
    print(f"Total steps in step-by-step logs: {total_steps}")
    
    # Group by episode to check completion
    ep_grouped = df_step.groupby('episode')
    ep_count = len(ep_grouped)
    print(f"Total episodes in step logs: {ep_count}")
    
    # Analyze collision rate and out-of-bounds (OOB) rate
    # Collision reward penalty is usually -15.0, OOB is -10.0
    collisions = df_step[df_step['reward'] == -15.0]
    oobs = df_step[df_step['reward'] == -10.0]
    print(f"Total collisions recorded: {len(collisions)} ({len(collisions)/total_steps*100:.2f}% of all steps)")
    print(f"Total OOB recorded: {len(oobs)} ({len(oobs)/total_steps*100:.2f}% of all steps)")
    
    # Calculate collision/OOB rate progression
    # Group steps into 10 bins
    df_step['bin'] = pd.qcut(df_step['episode'], 10, labels=False, duplicates='drop')
    bin_stats = df_step.groupby('bin').agg(
        mean_reward=('reward', 'mean'),
        collision_rate=('reward', lambda x: (x == -15.0).mean() * 100),
        oob_rate=('reward', lambda x: (x == -10.0).mean() * 100),
        p_hidden_mean=('p_hidden', 'mean'),
        mean_classifier_conf=('classifier_confidence_mean', 'mean'),
        mean_found_objects=('found_objects', 'mean')
    )
    print("\nTraining progression in 10 equal bins:")
    print(bin_stats.to_string())
else:
    print("Step CSV not found.")
