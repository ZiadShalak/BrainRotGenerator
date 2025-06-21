# analyze.py (Updated for Cluster-Based Learning)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# --- Configuration ---
LOG_FILE = Path("./logs/feedback.csv")

def main():
    """Main function to load data and generate plots for the cluster-based agent."""
    print(f"Analyzing log file: {LOG_FILE}")
    if not LOG_FILE.exists():
        print(f"Error: Log file not found at {LOG_FILE}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(LOG_FILE)
    except pd.errors.EmptyDataError:
        print("Log file is empty. No data to analyze.", file=sys.stderr)
        sys.exit(0)

    if df.empty:
        print("Log file has headers but no data. No data to analyze.", file=sys.stderr)
        sys.exit(0)

    # --- Filter for completed trials for performance analysis ---
    df_completed = df[df['outcome'] == 'completed'].copy()
    if df_completed.empty:
        print("No 'completed' trials found in log file. Cannot generate performance plots.", file=sys.stderr)
        # We can still show the plots window, even if it's empty.
        plt.show()
        sys.exit(0)

    # --- Plot 1: Top Performing Image Clusters ---
    print("Generating 'Top Image Clusters' plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    # <-- UPDATED column name here
    img_cat_performance = df_completed.groupby('image_cluster_id')['shaped_reward'].agg(['mean', 'count']).reset_index()
    top_img_cats = img_cat_performance.sort_values('mean', ascending=False).head(15)

    img_barplot = sns.barplot(x='mean', y='image_cluster_id', data=top_img_cats, palette='viridis', ax=ax1)
    ax1.set_title('Top 15 Performing Image Clusters by Average Reward', fontsize=16)
    ax1.set_xlabel('Average Shaped Reward', fontsize=12)
    ax1.set_ylabel('Image Cluster ID', fontsize=12) # <-- UPDATED label
    fig1.tight_layout()

    # --- Plot 2: Top Performing Sound Clusters ---
    print("Generating 'Top Sound Clusters' plot...")
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    # <-- UPDATED column name here
    snd_cat_performance = df_completed.groupby('sound_cluster_id')['shaped_reward'].agg(['mean', 'count']).reset_index()
    top_snd_cats = snd_cat_performance.sort_values('mean', ascending=False).head(15)

    snd_barplot = sns.barplot(x='mean', y='sound_cluster_id', data=top_snd_cats, palette='plasma', ax=ax2)
    ax2.set_title('Top 15 Performing Sound Clusters by Average Reward', fontsize=16)
    ax2.set_xlabel('Average Shaped Reward', fontsize=12)
    ax2.set_ylabel('Sound Cluster ID', fontsize=12) # <-- UPDATED label
    fig2.tight_layout()

    # --- Plot 3: Performance Over Time ---
    print("Generating 'Performance Over Time' plot...")
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    
    # Use .loc to avoid SettingWithCopyWarning
    df_completed.loc[:, 'reward_rolling_avg'] = df_completed['shaped_reward'].rolling(window=20, min_periods=1).mean()

    ax3.plot(df_completed.index, df_completed['shaped_reward'], 'o-', label='Per-Trial Reward', color='lightgray', markersize=4, alpha=0.7)
    ax3.plot(df_completed.index, df_completed['reward_rolling_avg'], 'r-', label='Rolling Average (20 trials)', linewidth=2.5)
    
    ax3.axhline(0, color='black', linestyle='--', linewidth=1, label='Win Threshold')
    
    ax3.set_title('Agent Performance Over Time (Completed Trials Only)', fontsize=16)
    ax3.set_xlabel('Trial Number', fontsize=12)
    ax3.set_ylabel('Shaped Reward', fontsize=12)
    ax3.legend()
    fig3.tight_layout()

    print("\nPlots generated. Close the plot windows to exit.")
    plt.show()

if __name__ == '__main__':
    main()