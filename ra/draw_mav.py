import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the files
file1 = 'no_train/test/vits_result.csv'
file2 = 'train/test/vits_result.csv'

data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)

#data1 = data1[:2000]
#data1 = data1[data1['prb'] > 20]
#data2 = data2[-2000:]
#data2 = data2[data2['prb'] > 20]

# Define the window size for moving average
window_size = 25

# Calculate moving averages for mcs, pwr, and prb columns in both files
metrics = ['mcs', 'pwr', 'prb']
moving_avg_data1 = {metric: data1[metric].rolling(window=window_size).mean() for metric in metrics}
moving_avg_data2 = {metric: data2[metric].rolling(window=window_size).mean() for metric in metrics}

# Plot the moving averages for each metric
for metric in metrics:
    plt.figure(figsize=(10, 6))

    # Plot the moving average for file1
    plt.plot(moving_avg_data1[metric], label=f'Original', color='blue')

    # Plot the moving average for file2
    plt.plot(moving_avg_data2[metric], label=f'VITS', color='orange')

    # Add labels, legend, and title
    plt.xlabel('T', fontsize=14)
    plt.ylabel(f'{metric.capitalize()}', fontsize=14)
    plt.title(f'Moving Average of {metric.capitalize()}', fontsize=16)
    plt.legend(fontsize=12)
    plt.xticks([])

    # Show grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f'plots/{metric}_moving_average_plot.png')
    #plt.show()

