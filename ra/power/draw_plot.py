import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Directories containing the data files
no_train_dir = '/home/nakaolab/ra/power/no_train/'
train_dir = '/home/nakaolab/ra/power/train/'

# Output directory for the plots
output_dir = '/home/nakaolab/ra/power/'
os.makedirs(output_dir, exist_ok=True)

# Metric to plot
metric = 'PkgWatt'  # This column exists in all files

# Function to read and process files from a directory
def read_and_concatenate_files(directory):
    all_data = []
    for file in sorted(os.listdir(directory)):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            df['file'] = os.path.basename(file)  # Add file identifier for grouping
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

# Read and combine data
no_train_data = read_and_concatenate_files(no_train_dir)
train_data = read_and_concatenate_files(train_dir)

# Add dataset identifier
no_train_data['legend'] = 'Baseline'
train_data['legend'] = 'VITS'

# Combine datasets
combined_data = pd.concat([no_train_data, train_data], ignore_index=True)

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(data=combined_data, x=combined_data.index, y=metric, hue='legend', style='file', markers=True)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Power (W)', fontsize=14)
plt.title('Power Consumption over Time', fontsize=16)
plt.legend(fontsize=12, title='Dataset')
plt.grid(True)

# Save the plot
plot_path = os.path.join(output_dir, f'{metric}_comparison.png')
plt.savefig(plot_path)
plt.close()
print(f"Plot saved to {plot_path}")

