import pandas as pd
import os

# Set the directory containing your files
directory = 'Receiver1'

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.startswith('trace_num') and filename.endswith('.csv') in filename:
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        
        # Load the CSV file
        data = pd.read_csv(file_path)
        
        # Check if 'Received:' is in the 'Channel (V)' column and remove it
        data['Channel (V)'] = data['Channel (V)'].replace(to_replace=r'^Received:\s*', value='', regex=True).astype(float)
        
        # Save the cleaned data back to the CSV
        new_filename = file_path.replace('.csv', '_revised.csv')

        data.to_csv(new_filename, index=False)
        print(f'Cleaned and saved "{new_filename}"')

print('All files have been cleaned.')