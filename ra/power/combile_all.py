import os
import pandas as pd

# Directory containing no_train files
no_train_dir = '/home/nakaolab/ra/power/train/'

# Output file path
output_file = '/home/nakaolab/ra/power/train_combined.csv'

# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame()

# Iterate through all files in the no_train directory
for file in sorted(os.listdir(no_train_dir)):
    if file.endswith('.csv'):
        file_path = os.path.join(no_train_dir, file)
        df = pd.read_csv(file_path)

        # Create the 'algo' column with all values set to 'Baseline'
        df['algo'] = 'VITS'

        # Create the 'index' column using the index of each row in the original file
        df['index'] = df.index

        # Keep only the required columns: 'algo', 'index', and 'pwr'
        df = df[['algo', 'index', 'PkgWatt']]  # Assuming 'PkgWatt' is the power column
        df.rename(columns={'PkgWatt': 'pwr'}, inplace=True)

        # Concatenate the current file's data to the combined DataFrame
        combined_data = pd.concat([combined_data, df], ignore_index=True)

# Save the combined data to the output CSV file
combined_data.to_csv(output_file, index=False)
print(f"Combined data saved to {output_file}")

