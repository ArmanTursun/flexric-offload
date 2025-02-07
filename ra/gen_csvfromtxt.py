import csv
import re

# Input and output file names
input_filename = "/home/nakaolab/pcm_result_stationary_20.txt"
output_filename = "/home/nakaolab/ra/minimum/pcm_result_stationary_20.csv"

# Initialize data storage
data = []

# Read and extract required values
with open(input_filename, "r") as file:
    for index, line in enumerate(file):
        match = re.search(r"Watts:\s*([\d\.]+)", line)
        if match:
            pwr_value = float(match.group(1))
            data.append(["Minimum", index, pwr_value])

# Write to CSV file
with open(output_filename, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["legend", "step", "pwr"])  # Write header
    writer.writerows(data)  # Write data

print(f"Extracted data saved to {output_filename}")

