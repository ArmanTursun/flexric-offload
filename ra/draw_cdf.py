import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define a function to calculate CDF
def calculate_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf

# File paths
file1 = "no_train/test/vits_result.csv"
file2 = "train/test/vits_result.csv"

# Load the data
file1_data = pd.read_csv(file1)
file2_data = pd.read_csv(file2)

#file1_data = file1_data[:1000]
#file1_data = file1_data[file1_data['prb'] > 20]
#file2_data = file2_data[-1000:]
#file2_data = file2_data[file2_data['prb'] > 20]

# Define the columns to plot
columns_to_plot = ["mcs", "prb", "pwr"]
labels = ["Original", "VITS"]

for column in columns_to_plot:
    # Calculate CDF for file1
    sorted_file1, cdf_file1 = calculate_cdf(file1_data[column])

    # Calculate CDF for file2
    sorted_file2, cdf_file2 = calculate_cdf(file2_data[column])

    # Plot the CDF
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_file1, cdf_file1, label=f"{labels[0]}", linestyle='-', marker='')
    plt.plot(sorted_file2, cdf_file2, label=f"{labels[1]}", linestyle='-', marker='')

    # Customize the plot
    plt.xlabel(f"{column.upper()} Values", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title(f"CDF of {column.upper()} for Original vs VITS", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(f"plots/cdf_{column}.png")
    #plt.show()

