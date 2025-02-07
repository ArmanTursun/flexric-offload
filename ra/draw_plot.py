import pandas as pd
import matplotlib.pyplot as plt

# File paths
file1 = 'no_train/test/vits_result.csv'
file2 = 'train/test/vits_result.csv'

# Read the CSV files
data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)

#data1 = data1[:2000]
#data1 = data1[data1['prb'] > 20]
#data2 = data2[-2000:]
#data2 = data2[data2['prb'] > 20]

# Check if required columns exist
required_columns = ['step', 'mcs', 'prb', 'pwr']
for col in required_columns:
    if col not in data1.columns or col not in data2.columns:
        raise ValueError(f"Column '{col}' is missing in one of the files.")

# Plot MCS line plot
plt.figure(figsize=(10, 6))
plt.plot(data1['step'], data1['mcs'], label='Original', marker='o', linestyle='-', color='blue', alpha=0.5)
plt.plot(data2['step'], data2['mcs'], label='VITS', marker='o', linestyle='-', color='red', alpha=0.5)
plt.xlabel('T', fontsize=14)
plt.ylabel('MCS', fontsize=14)
plt.title('MCS over Steps', fontsize=16)
plt.legend(fontsize=12)
plt.xticks([])
plt.grid(False)
plt.tight_layout()
plt.savefig('plots/mcs_line_plot.png')
#plt.show()

# Plot PRB line plot
plt.figure(figsize=(10, 6))
plt.plot(data1['step'], data1['prb'], label='Original', marker='o', linestyle='-', color='blue', alpha=0.5)
plt.plot(data2['step'], data2['prb'], label='VITS', marker='o', linestyle='-', color='red', alpha=0.5)
plt.xlabel('T', fontsize=14)
plt.ylabel('PRB', fontsize=14)
plt.title('PRB over Steps', fontsize=16)
plt.legend(fontsize=12)
plt.xticks([])
plt.grid(False)
plt.tight_layout()
plt.savefig('plots/prb_line_plot.png')
#plt.show()

# Plot PWR dot plot
plt.figure(figsize=(10, 6))
plt.scatter(data1['step'], data1['pwr'], label='Original', color='blue', alpha=0.5)
plt.scatter(data2['step'], data2['pwr'], label='VITS', color='red', alpha=0.5)
plt.xlabel('T', fontsize=14)
plt.ylabel('PWR', fontsize=14)
plt.title('Power over Steps', fontsize=16)
plt.legend(fontsize=12)
plt.xticks([])
plt.grid(False)
plt.tight_layout()
plt.savefig('plots/pwr_dot_plot.png')
#plt.show()

