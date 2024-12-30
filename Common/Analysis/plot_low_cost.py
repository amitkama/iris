import os
import pandas as pd
import matplotlib.pyplot as plt

# Loop through each Receiver folder from Receiver1 to Receiver20
for receiver_num in range(1, 21):  # Ensure it includes Receiver20
    folder_name = f"Receiver{receiver_num}"
    
    # Check if the folder exists
    if os.path.exists(folder_name):
        print(f"Processing {folder_name}...")

        # Loop through each file number within the folder
        for i in range(1, 101):  # Adjust range for the number of trace files per receiver
            file_pattern = f"trace_num{i}.csv"
            
            # Find files with matching pattern
            matching_files = [f for f in os.listdir(folder_name) if f.startswith(f"trace_num{i}") and f.endswith(".csv")]
            
            if matching_files:
                filename = matching_files[0]  # Take the first matched file
                filepath = os.path.join(folder_name, filename)
                plotname = os.path.join(folder_name, f"plot{i}.png")

                # Read the CSV file into a DataFrame
                df = pd.read_csv(filepath)

                # Strip whitespaces from column names
                df.columns = df.columns.str.strip()

                # Convert 'Time (us)' column to numeric and then to milliseconds
                df['Time (ms)'] = pd.to_numeric(df['Time (us)'], errors='coerce') / 1000

                # Extract the columns
                time_ms = df['Time (ms)']  # This is now time in milliseconds
                voltage_V = df['Channel (V)'] / 1000  # Convert millivolts to volts

                # Create a new figure for each iteration
                plt.figure(figsize=(20, 12))

                # Plot the data
                plt.plot(time_ms, voltage_V, label='Voltage (V)')

                # Add labels and legend
                plt.xlabel('Time (ms)')
                plt.ylabel('Voltage (V)')
                plt.legend()

                # Adjust x-axis limits
                plt.xlim(time_ms.min(), time_ms.max())

                # Save the plot to a file
                plt.savefig(plotname, dpi=300)

                # Clear the figure to free memory for the next iteration
                plt.clf()

                print(f"Plot saved for {filename} as {plotname}")
            else:
                print(f"No file found for pattern trace_num_{i}_*.csv in {folder_name}")
    else:
        print(f"Folder {folder_name} does not exist.")

print("Processing complete.")
