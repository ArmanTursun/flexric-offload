import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#dataset = 'dynamic_10_20'
dataset = 'stationary_20'
#dataset = ''
output_dir = '/home/nakaolab/ra/power/all_' + dataset + '/'
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
input_dir1 = '/home/nakaolab/ra/train/new_' + dataset + '/'
input_dir2 = '/home/nakaolab/ra/no_train/new_' + dataset + '/'
input_dir3 = '/home/nakaolab/ra/minimum/pcm_result_' + dataset + '.csv'

minimum_value = 74.1 # 20-74.1, 15-74.6, 10-75.1, dynamic-76.11
if dataset == 'stationary_10':
    minimum_value = 75.1
elif dataset == 'stationary_15':
    minimum_value = 74.6
elif dataset == 'stationary_20':
    minimum_value = 74.3
else:
    minimum_value = 74.7

# Function to read and process files from a directory
def read_and_concatenate_files(directory):
    all_data = []
    for folder in sorted(os.listdir(directory)):
        folder_path = os.path.join(directory, folder)
        file_path = os.path.join(folder_path, 'vits_result.csv')
        df = pd.read_csv(file_path)
        df['round'] = os.path.basename(folder)  # Add file identifier for grouping
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

df1 = pd.DataFrame()
df2 = pd.DataFrame()
df1 = read_and_concatenate_files(input_dir1)
df2 = read_and_concatenate_files(input_dir2)
#df3 = pd.DataFrame()
#df3 = pd.read_csv(input_dir3)
# Generate and save plots
plot_metrics = ['pwr'] #, 'action_mcs', 'action_prb' 'tbs'       'avg_regret', 'cum_regret', 'mcs', 'prb'
label = {'pwr': 'Power', 'cum_regret': 'Cumulative Regret', 'avg_regret': 'Regret', 'mcs': 'MCS', 'prb': 'PRB', 'action_mcs': 'MCS Cap', 'action_prb': 'PRB Cap', 'tbs': 'TBS'}
scaler = MinMaxScaler(feature_range=(0, 1))
for metric in plot_metrics:
    df = pd.DataFrame()
    if metric == 'pwr' or metric == 'action_mcs' or metric == 'action_prb' or metric == 'mcs' or metric == 'prb' or metric == 'tbs':
        df = pd.concat([df1, df2], ignore_index=True)
        #df = pd.concat([df1, df2, df3], ignore_index=True)
    else:
        df = df1   
    #df[metric] = scaler.fit_transform(df[[metric]])
    plt.figure(figsize=(10, 6))

    
    #startidx = 7
    #endidx = 5600
    #df = df[df['step'] > startidx]
    #df = df[df['step'] < endidx]
    #df = df[df['avg_regret'] < 16]
    #df = df[df['pwr'] - 2.0]

    if metric == 'pwr' or metric == 'cum_regret' or metric == 'avg_regret':
        sns.lineplot(data=df, x='step', y=metric, hue='legend') 
        min_line = plt.axhline(y=minimum_value, color='red', linestyle='--', linewidth = 2) # 20-74.1, 15-74.6, 10-75.1, dynamic-76.11
        plt.xlabel('Time', fontsize=14)
        plt.ylabel(label[metric], fontsize=14) # .replace('_', ' ').capitalize()
    #plt.ylabel('Power', fontsize=14)
        plt.title(f'{label[metric]} over Time', fontsize=16) # metric.replace("_", " ").capitalize()
        plt.xticks([], [])  # Hide x-axis ticks
        plt.yticks(np.arange(70, 86, 1))
        #plt.legend(fontsize=12) # title='Legend', 
        handles, labels = plt.gca().get_legend_handles_labels()
        min_legend = mlines.Line2D([], [], color = 'red', linestyle='--', linewidth=2, label='Minimum')
        handles.append(min_legend)
        labels.append('Minimum')
        plt.legend(handles=handles, labels=labels, fontsize=12)
    #"""
    if metric == 'mcs' or metric == 'prb':
        df = df[df[metric] > 0]
        ax = sns.ecdfplot(data=df, x=metric, hue=df['legend'])
        sns.move_legend(ax, loc='upper left', title=None, fontsize=12)
        plt.title(f'{label[metric]} Distribution', fontsize=16) # metric.replace("_", " ").capitalize()
    #"""
    #plt.legend(title='Legend', fontsize=12) # title='Legend', 
    plt.grid(True)   
    plot_path = os.path.join(output_dir, f'{metric}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")
    
