import os
import pandas as pd
import numpy as np

def analyze_logs(base_dir):
    print("=========================================")
    print("Analyzing RL Training Results from:")
    print(base_dir)
    print("=========================================\n")

    # Paths to the CSV files
    sac_metrics_path = os.path.join(base_dir, "sac_metrics (1).csv")
    training_metrics_path = os.path.join(base_dir, "logs", "training_metrics (8).csv")

    # 1. Analyze sac_metrics.csv
    if os.path.exists(sac_metrics_path):
        sac_df = pd.read_csv(sac_metrics_path)
        print("--- SAC Episode-Level Metrics (sac_metrics (1).csv) ---")
        print(f"Total Steps: {sac_df['step'].max()}")
        print(f"Total Episodes Logged: {len(sac_df)}")
        print(f"Mean Episode Reward: {sac_df['ep_reward'].mean():.4f}")
        print(f"Max Episode Reward: {sac_df['ep_reward'].max():.4f}")
        print(f"Min Episode Reward: {sac_df['ep_reward'].min():.4f}")
        print(f"Mean Episode Steps: {sac_df['ep_steps'].mean():.2f}")
        
        # Calculate success rate from the last 100 episodes
        last_100_sac = sac_df.tail(100)
        success_rate = (last_100_sac['success'] == 1).mean() * 100
        print(f"Success Rate (Overall): {(sac_df['success'] == 1).mean() * 100:.2f}%")
        print(f"Success Rate (Last 100 episodes): {success_rate:.2f}%")
        print()
    else:
        print(f"File not found: {sac_metrics_path}\n")

    # 2. Analyze training_metrics.csv
    if os.path.exists(training_metrics_path):
        train_df = pd.read_csv(training_metrics_path)
        print("--- Detailed Step-Level Metrics (training_metrics (8).csv) ---")
        print(f"Total recorded steps: {len(train_df)}")
        print(f"Total recorded episodes: {train_df['episode'].max() + 1}")
        
        # Let's group by episode to see outcomes
        ep_groups = train_df.groupby('episode')
        ep_summary = []
        for ep_id, group in ep_groups:
            # Determine outcome of episode
            last_step = group.iloc[-1]
            reward_at_last = last_step['reward']
            cum_reward = last_step['cum_reward']
            
            # Collision: if reward was PENALTY_COLLISION (-15)
            # OOB: if reward was PENALTY_OOB (-10)
            # Success: if success == 1
            has_collision = (group['reward'] == -15.0).any()
            has_oob = (group['reward'] == -10.0).any()
            success = int(last_step['success'] == 1)
            
            # If the episode ended early due to OOB or collision:
            outcome = "trunc/max_steps"
            if success:
                outcome = "success"
            elif has_collision:
                outcome = "collision"
            elif has_oob:
                outcome = "oob"
                
            ep_summary.append({
                'episode': ep_id,
                'outcome': outcome,
                'cum_reward': cum_reward,
                'steps': len(group),
                'found_pct': last_step['objects_found_so_far'] / last_step['total_objects'] if last_step['total_objects'] > 0 else 0
            })
            
        ep_summary_df = pd.DataFrame(ep_summary)
        outcomes_counts = ep_summary_df['outcome'].value_counts()
        outcomes_pct = ep_summary_df['outcome'].value_counts(normalize=True) * 100
        
        print("\nEpisode Outcomes Distribution:")
        for outcome in ['success', 'collision', 'oob', 'trunc/max_steps']:
            count = outcomes_counts.get(outcome, 0)
            pct = outcomes_pct.get(outcome, 0.0)
            print(f"  - {outcome.upper()}: {count} episodes ({pct:.2f}%)")
            
        print(f"\nAverage Cumulative Reward per Outcome:")
        avg_rewards = ep_summary_df.groupby('outcome')['cum_reward'].mean()
        for outcome, avg_r in avg_rewards.items():
            print(f"  - {outcome.upper()}: {avg_r:.4f}")
            
        print(f"\nAverage Steps per Outcome:")
        avg_steps = ep_summary_df.groupby('outcome')['steps'].mean()
        for outcome, avg_s in avg_steps.items():
            print(f"  - {outcome.upper()}: {avg_s:.2f} steps")

        # Analyze rewards and classifier metrics
        print("\nAverage Reward Component Values (across all steps):")
        components = ['rew_coverage', 'rew_classifier', 'rew_all_found', 'rew_success_classified', 'rew_survival']
        for comp in components:
            if comp in train_df.columns:
                print(f"  - {comp}: {train_df[comp].mean():.6f}")
                
        # Analyze confidence
        if 'classifier_confidence_mean' in train_df.columns:
            print(f"\nMean Classifier Confidence (all steps): {train_df['classifier_confidence_mean'].mean():.4f}")
        if 'p_hidden' in train_df.columns:
            print(f"Mean p_hidden (all steps): {train_df['p_hidden'].mean():.4f}")
            print(f"Min p_hidden reached: {train_df['p_hidden'].min():.4f}")
            
    else:
        print(f"File not found: {training_metrics_path}")

if __name__ == "__main__":
    analyze_logs(r"C:\Users\NeverGonnaGiveYouUp\OneDrive\Рабочий стол\study_materials\Skoltech\Reinforcement_Learning\Project 4 TEST\NBV_with_obstacles_and_robot\kaggle_output\output_odin_sac")
