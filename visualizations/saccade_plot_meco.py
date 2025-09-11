import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your data
df = pd.read_csv('../datasets/mecoL1/MECO-en_uk-sac.csv')

def plot_saccade_paths_per_trial(df, uniform_id=None, trialid=None):
    """
    Plot saccade paths for specific participant and trial
    """
    # Filter data
    if uniform_id and trialid:
        filtered_df = df[(df['uniform_id'] == uniform_id) & (df['trialid'] == trialid)]
        title_suffix = f' - {uniform_id}, Trial {trialid}'
    elif uniform_id:
        filtered_df = df[df['uniform_id'] == uniform_id]
        title_suffix = f' - {uniform_id}'
    else:
        filtered_df = df
        title_suffix = ''
    
    if filtered_df.empty:
        print("No data found for the specified criteria")
        return
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Get unique trials for coloring
    unique_trials = filtered_df['trialid'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_trials)))
    
    for trial, color in zip(unique_trials, colors):
        trial_data = filtered_df[filtered_df['trialid'] == trial]
        
        for i, sac in trial_data.iterrows():
            plt.arrow(sac['xs'], sac['ys'], sac['dx'], sac['dy'],
                     head_width=8, head_length=12, fc=color, ec=color, alpha=0.7,
                     length_includes_head=True, width=2)
            
            # Label saccade number
            plt.text(sac['xs'], sac['ys'] - 15, f"s{sac['sacid']}", 
                    fontsize=8, ha='center', color=color)
    
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.title(f'Saccade Paths{title_suffix}')
    plt.gca().invert_yaxis()  # Typically Y increases downward in eye-tracking
    plt.grid(True, alpha=0.3)
    
    # Add legend for trials
    legend_elements = [plt.Line2D([0], [0], color=colors[i], lw=4, 
                      label=f'Trial {trial}') for i, trial in enumerate(unique_trials)]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.show()

# Example usage:
plot_saccade_paths_per_trial(df, uniform_id='en_uk_1', trialid=1)