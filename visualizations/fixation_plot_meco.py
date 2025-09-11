import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_fixation_sequence(df, uniform_id, trialid):
    """
    Plot fixation sequence for a specific participant and trial
    """
    # Filter data for the specific participant and trial
    trial_data = df[(df['uniform_id'] == uniform_id) & (df['trialid'] == trialid)]
    
    if trial_data.empty:
        print(f"No data found for {uniform_id}, trial {trialid}")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Fixation positions with duration as bubble size
    fixations = trial_data.sort_values('start')
    
    # Create bubbles with size proportional to fixation duration
    scatter = ax1.scatter(fixations['xs'], fixations['ys'], 
                         s=fixations['dur']/2,  # Scale duration for visibility
                         c=range(len(fixations)), 
                         cmap='viridis', 
                         alpha=0.7,
                         edgecolors='black',
                         linewidth=0.5)
    
    # Add fixation numbers
    for i, fix in fixations.iterrows():
        ax1.text(fix['xs'], fix['ys'] + 15, f"F{fix['fixid']}", 
                ha='center', fontsize=9, fontweight='bold')
        ax1.text(fix['xs'], fix['ys'] - 20, f"{fix['dur']}ms", 
                ha='center', fontsize=8)
    
    # Draw lines connecting fixations in sequence
    ax1.plot(fixations['xs'], fixations['ys'], 'gray', alpha=0.5, linestyle='--')
    
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.set_title(f'Fixation Sequence - {uniform_id}, Trial {trialid}\n'
                 f'Text: {fixations.iloc[0]["sent"].strip()}')
    ax1.invert_yaxis()  # Typically Y increases downward in eye-tracking
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for fixation sequence
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Fixation Sequence Order')
    
    # Plot 2: Fixation duration timeline
    ax2.bar(range(len(fixations)), fixations['dur'], 
           color=plt.cm.viridis(np.linspace(0, 1, len(fixations))),
           edgecolor='black')
    
    for i, (fix_id, duration) in enumerate(zip(fixations['fixid'], fixations['dur'])):
        ax2.text(i, duration + 5, f"F{fix_id}", ha='center', fontsize=9)
        ax2.text(i, duration/2, f"{duration}ms", ha='center', fontsize=8, 
                color='white', fontweight='bold')
    
    ax2.set_xlabel('Fixation Sequence Number')
    ax2.set_ylabel('Duration (ms)')
    ax2.set_title('Fixation Duration Timeline')
    ax2.set_xticks(range(len(fixations)))
    ax2.set_xticklabels([f'F{fixid}' for fixid in fixations['fixid']])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary information
    print(f"Participant: {uniform_id}")
    print(f"Trial: {trialid}")
    print(f"Text: {fixations.iloc[0]['sent'].strip()}")
    print(f"Total fixations: {len(fixations)}")
    print(f"Total reading time: {fixations['dur'].sum()} ms")
    print(f"Average fixation duration: {fixations['dur'].mean():.1f} ms")
    print(f"Fixation sequence: {list(fixations['fixid'])}")

def plot_word_level_fixations(df, uniform_id, trialid):
    """
    Plot fixations organized by word/IA
    """
    trial_data = df[(df['uniform_id'] == uniform_id) & (df['trialid'] == trialid)]
    
    if trial_data.empty:
        return
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Group by word/IA and plot fixations
    words = trial_data['word'].unique()
    word_positions = {word: i for i, word in enumerate(words)}
    
    for i, fix in trial_data.iterrows():
        word_idx = word_positions[fix['word']]
        plt.scatter(word_idx, fix['dur'], s=150, alpha=0.7, 
                   c=fix['fixid'], cmap='viridis')
        plt.text(word_idx, fix['dur'] + 10, f"F{fix['fixid']}", 
                ha='center', fontsize=9)
    
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    plt.xlabel('Words/Interest Areas')
    plt.ylabel('Fixation Duration (ms)')
    plt.title(f'Fixation Durations by Word - {uniform_id}, Trial {trialid}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Load your data
df = pd.read_csv('../datasets/mecoL1/MECO-en_uk-fix_uk01.csv')

# Example usage:
plot_fixation_sequence(df, 'uk01', 2)
plot_word_level_fixations(df, 'uk01', 2)