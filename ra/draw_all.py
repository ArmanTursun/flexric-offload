import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.ndimage import gaussian_filter1d

dataset = 'dynamic_10_20'
#dataset = 'stationary_15'
#dataset = ''
output_dir = '/home/nakaolab/flexric-offload/ra/power/all_' + dataset + '/'
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
input_dir1 = '/home/nakaolab/flexric-offload/ra/train/new_' + dataset + '/'
input_dir2 = '/home/nakaolab/flexric-offload/ra/no_train/new_' + dataset + '/'
input_dir3 = '/home/nakaolab/flexric-offload/ra/minimum/pcm_result_' + dataset + '.csv'

minimum_value = 74.1 # 20-74.3, 15-74.6, 10-75.1, dynamic-74.7
if dataset == 'stationary_10':
    minimum_value = 75.1
elif dataset == 'stationary_15':
    minimum_value = 74.3
elif dataset == 'stationary_20':
    minimum_value = 74.0
else:
    minimum_value = 74.4

# Function to read and process files from a directory
def read_and_concatenate_files(directory):
    all_data = []
    for folder in sorted(os.listdir(directory)):
        folder_path = os.path.join(directory, folder)
        file_path = os.path.join(folder_path, 'vits_result.csv')
        df = pd.read_csv(file_path)
        df['round'] = os.path.basename(folder)  # Add file identifier for grouping
        #df['avg_regret'] = df.groupby('legend')['avg_regret'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
        #df['avg_regret'] = df.groupby('legend')['avg_regret'].transform(lambda x: gaussian_filter1d(x, sigma=1))
        #df['pwr'] = df.groupby('legend')['pwr'].transform(lambda x: gaussian_filter1d(x, sigma=2))
        df['pwr'] = df.groupby('legend')['pwr'].transform(lambda x: x.ewm(span=100, adjust=False).mean())
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

df1 = pd.DataFrame()
df2 = pd.DataFrame()
df1 = read_and_concatenate_files(input_dir1)
df2 = read_and_concatenate_files(input_dir2)
#df3 = pd.DataFrame()
#df3 = pd.read_csv(input_dir3)
# Generate and save plots
plot_metrics = ['cum_regret'] #, 'action_mcs', 'action_prb' 'tbs'    'pwr'   'avg_regret', 'cum_regret', 'mcs', 'prb'
label = {'pwr': 'Power (Watt)', 'cum_regret': 'Cumulative Regret', 'avg_regret': 'Regret', 'mcs': 'MCS', 'prb': 'PRB', 'action_mcs': 'MCS Cap', 'action_prb': 'PRB Cap', 'tbs': 'TBS'}
scaler = MinMaxScaler(feature_range=(0, 1))
for metric in plot_metrics:
    plt.figure(figsize=(10, 6))
    df = pd.DataFrame()
    df_summary = pd.DataFrame()
    if metric == 'pwr' or metric == 'action_mcs' or metric == 'action_prb' or metric == 'mcs' or metric == 'prb' or metric == 'tbs':
        df = pd.concat([df2, df1], ignore_index=True)
        #df = pd.concat([df1, df2, df3], ignore_index=True)
    else:
        df = df1   
        # Compute inverse weights (lower values get higher weights)
    df['weights'] = 1 / df[metric]  # Inverse relationship
    df['weights'] /= df['weights'].max()  # Normalize to keep values reasonable

    # Compute the quantiles (0.05 and 0.95) and mean
    quantiles = df.groupby('step')[metric].quantile([0.05, 0.95]).unstack(level=1)
    quantiles['mean'] = df.groupby('step')[metric].mean() 

    #df[metric] = scaler.fit_transform(df[[metric]])
    

    
    startidx = 7
    #endidx = 5600
    df = df[df['step'] > startidx]
    #df = df[df['step'] < endidx]
    df = df[df['avg_regret'] < 16]
    if metric == 'pwr':
        df['pwr'] = df['pwr'] - 50.0

    palette = sns.color_palette()

    if metric == 'pwr' or metric == 'cum_regret' or metric == 'avg_regret':
        #sns.lineplot(data=df_sampled, x='step', y='mean', hue='legend', palette=palette)  # Marker for clarity
        #plt.fill_between(quantiles.index, quantiles[0.05], quantiles[0.95], color=palette, alpha=0.2)
        sns.lineplot(data=df, x='step', y=metric, hue='legend', palette=palette, estimator='mean', weights=df['weights'], errorbar=("ci", 85)) #  (palette[1], palette[0])
        #min_line = plt.axhline(y=minimum_value - 50.0 + 0.5, color='red', linestyle='--', linewidth=3) # 20-74.1, 15-74.6, 10-75.1, dynamic-76.11
        plt.xlabel('T($10^3$)', fontsize=24)
        plt.ylabel(label[metric] + ' ($10^3$)', fontsize=24) # .replace('_', ' ').capitalize()
        plt.title(f'{label[metric]} over Steps', fontsize=24) # metric.replace("_", " ").capitalize()
        #plt.xticks([], [])  # Hide x-axis ticks
        tick_positions = np.arange(0, 7001, 1000)  # Positions of xticks
        tick_labels = [str(int(tick / 1000)) for tick in tick_positions]  # Convert to 1, 2, 3...
        plt.xticks(tick_positions, tick_labels, fontsize=24)  # Set formatted xticks
        #plt.text(2100, -2.6, r'$\times10^2$', ha='right', va='bottom', fontsize=20) # -1600
        #plt.xticks(np.arange(0, 2001, 200))
        #plt.yticks(np.arange(23, 35, 1), fontsize=24)
        #plt.yticks(np.arange(0, 13, 2), fontsize=24)
        tick_positions_y = np.arange(0, 25001, 5000)  # Positions of xticks
        tick_labels_y = [str(int(tick / 1000)) for tick in tick_positions_y]  # Convert to 1, 2, 3...
        plt.yticks(tick_positions_y, tick_labels_y, fontsize=24)  # Set formatted xticks
        #plt.yticks(np.arange(0, 8000, 1000), fontsize=24)
        handles, labels = plt.gca().get_legend_handles_labels()
        #min_legend = mlines.Line2D([], [], color = 'red', linestyle='--', linewidth=3, label='Minimum')
        #handles.append(min_legend)
        #labels.append('Minimum')
        plt.legend(handles=handles, labels=labels, fontsize=24, loc='upper right')
    if metric == 'mcs' or metric == 'prb':
        df = df[df[metric] > 0]
        ax = sns.ecdfplot(data=df, x=metric, hue=df['legend'], palette=(palette[1], palette[0])) #  (palette[1], palette[0])
        plt.xlabel(label[metric], fontsize=24)
        if metric == 'mcs':
            plt.xticks(np.arange(11, 26, 4))
        else:
            plt.xticks(np.arange(10, 105, 20))
        plt.ylabel('Proportion', fontsize=24)
        ax.tick_params(axis='both', labelsize=24)
        sns.move_legend(ax, loc='upper left', title=None, fontsize=24)
        plt.title(f'{label[metric]} Distribution', fontsize=24) # metric.replace("_", " ").capitalize()
    #plt.legend(title='Legend', fontsize=12) # title='Legend', 
    #plt.grid(True)   
    plot_path = os.path.join(output_dir, f'{metric}.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")
    
