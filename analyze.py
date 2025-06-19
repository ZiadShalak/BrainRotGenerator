# analyze.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import sys

# --- Configuration ---
# This should match the LOG_FILE path in montage.py
LOG_FILE = Path("./logs/feedback.csv")

def format_montage_label(row):
    """Creates a clean, readable label from the JSON string columns for plotting."""
    try:
        # The 'images' and 'sounds' columns are stored as JSON strings in the CSV
        images = json.loads(row['images'])
        sounds = json.loads(row['sounds'])
        
        # Extract just the filenames for a cleaner label
        img_name = Path(images[0]).name if images else "None"
        snd_name = Path(sounds[0]).name if sounds else "None"
        
        # We'll just use the first image/sound for the label to keep it concise
        return f"Img: {img_name}\nSnd: {snd_name}"
    except (json.JSONDecodeError, IndexError):
        return "Invalid Montage Format"

def main():
    """Main function to load data and generate plots."""
    print(f"Analyzing log file: {LOG_FILE}")
    if not LOG_FILE.exists():
        print(f"Error: Log file not found at {LOG_FILE}")
        print("Please run montage.py first to generate some data.")
        sys.exit(1)

    df = pd.read_csv(LOG_FILE)

    if df.empty:
        print("Log file is empty. No data to analyze.")
        sys.exit(0)

    # --- Data Preparation ---
    # Create a readable label for each unique montage configuration
    df['montage_label'] = df.apply(format_montage_label, axis=1)

    # --- Plot 1: Top 10 Performing Montages ---
    print("Generating 'Top 10 Montages' plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    # Group by the montage label and calculate the mean reward and the number of times it was tried
    montage_performance = df.groupby('montage_label')['shaped_reward'].agg(['mean', 'count']).reset_index()
    top_10_montages = montage_performance.sort_values('mean', ascending=False).head(10)

    barplot = sns.barplot(x='mean', y='montage_label', data=top_10_montages, palette='viridis', ax=ax1)
    ax1.set_title('Top 10 Performing Montages by Average Reward', fontsize=16)
    ax1.set_xlabel('Average Shaped Reward', fontsize=12)
    ax1.set_ylabel('Montage (First Image/Sound)', fontsize=12)

    # Add count annotations to the bars
    for index, row in top_10_montages.iterrows():
        barplot.text(row['mean']/2, index, f" (n={row['count']})", color='white', ha="center", va="center", fontsize=10)

    fig1.tight_layout()

    # --- Plot 2: Performance Over Time ---
    print("Generating 'Performance Over Time' plot...")
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    # Calculate a rolling average to see the trend more clearly
    df['reward_rolling_avg'] = df['shaped_reward'].rolling(window=15, min_periods=1).mean()

    ax2.plot(df.index, df['shaped_reward'], 'o-', label='Per-Trial Reward', color='lightgray', markersize=4, alpha=0.8)
    ax2.plot(df.index, df['reward_rolling_avg'], 'r-', label='Rolling Average (15 trials)', linewidth=2.5)
    
    ax2.axhline(0, color='black', linestyle='--', linewidth=1, label='Win Threshold') # The zero-line indicates breaking even
    
    ax2.set_title('Agent Performance Over Time', fontsize=16)
    ax2.set_xlabel('Trial Number', fontsize=12)
    ax2.set_ylabel('Shaped Reward', fontsize=12)
    ax2.legend()
    fig2.tight_layout()

    print("\nPlots generated. Close the plot windows to exit.")
    plt.show()

if __name__ == '__main__':
    main()