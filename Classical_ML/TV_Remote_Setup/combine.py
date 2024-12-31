import pandas as pd
import os

num_receivers = 20

for n in range(0, 65001, 100):

    all_dataframes = []
    for i in range(1, num_receivers + 1):
        directory = f'Receiver{i}'  # Adjust directory naming as necessary
        file_path = os.path.join(directory, f'length_features_reception{n}.csv')
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Add a 'Label' column with the Receiver number
            df['Label'] = i
            all_dataframes.append(df)

    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    combined_csv_path = f'combined_reception/complete_length_features_reception{n}.csv'  # Adjust the save path
    combined_df.to_csv(combined_csv_path, index=False)

    print(f'Combined DataFrame saved to {combined_csv_path}')