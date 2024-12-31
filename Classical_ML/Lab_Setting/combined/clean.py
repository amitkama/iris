import pandas as pd
import os

def clean_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Remove any empty columns that might have been read as part of the CSV due to trailing commas
    df.dropna(axis=1, how='all', inplace=True)
    
    # Fill empty cells (represented by consecutive commas in CSV) with a placeholder or appropriate value
    # Option 1: Fill with a constant such as 0 or a specific string like 'missing'
    df.fillna(0, inplace=True)  # Adjust this based on your data type or needs

    # Option 2: Fill with mean or median if the columns are numeric
    # Uncomment and use the below code if appropriate
    # for col in df.select_dtypes(include=['float', 'int']).columns:
    #     df[col].fillna(df[col].mean(), inplace=True)

    # Ensure that all floating point numbers are displayed in a consistent format
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    
    # Convert the last column to integer if necessary
    if not df.empty and df.iloc[:, -1].dtype == float:
        df.iloc[:, -1] = df.iloc[:, -1].astype(int)
    
    # Save the cleaned and processed dataset
    cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
    df.to_csv(cleaned_file_path, index=False)
    
    print(f"Cleaned data saved to {cleaned_file_path}")


for n in range(4990, 65001, 100):
    # File path to the dataset
    file_path = f'complete_length_features{n}.csv'
    clean_data(file_path)