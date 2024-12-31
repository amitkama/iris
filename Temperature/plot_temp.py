import os
import matplotlib.pyplot as plt

# Paths to the Cold and Warm folders
cold_folder = 'cold'
warm_folder = 'warm'

# Function to extract temperature from the file name
def extract_temperature(file_name):
    parts = file_name.split('_')
    temperature = float(parts[-1].split('.csv')[0])
    return temperature

# Lists to hold temperatures for cold and warm datasets
cold_temperatures = []
warm_temperatures = []

# Iterate over all files in the Cold folder
for receiver in range(1, 21):
    receiver_folder = os.path.join(cold_folder, f'Receiver{receiver}')
    for file_name in os.listdir(receiver_folder):
        if file_name.endswith('.csv'):
            temperature = extract_temperature(file_name)
            cold_temperatures.append(temperature)

# Iterate over all files in the Warm folder
for receiver in range(1, 21):
    receiver_folder = os.path.join(warm_folder, f'Receiver{receiver}')
    for file_name in os.listdir(receiver_folder):
        if file_name.endswith('.csv'):
            temperature = extract_temperature(file_name)
            warm_temperatures.append(temperature)

# Plot the temperatures with different colors for Cold and Warm
plt.figure(figsize=(10, 6))

# Plot Cold temperatures
plt.scatter(range(len(cold_temperatures)), cold_temperatures, edgecolors='#4682B4', facecolors='none', label='Low-temperature Dataset', s=20)

# Plot Warm temperatures
plt.scatter(range(len(warm_temperatures)), warm_temperatures, edgecolors='#d74a49', facecolors='none', marker='^', label='High-temperature Dataset', s=20)

# Add labels and title
plt.xlabel('Trace Number', fontsize=14)
plt.ylabel('Temperature', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.title('Temperature Comparison Between Cold and Warm Traces')
plt.legend()

# Show the plot
plt.savefig('Temperature.pdf', format='pdf', dpi=300)

# Calculate and print the max and min temperatures for the Cold dataset
max_cold_temp = max(cold_temperatures)
min_cold_temp = min(cold_temperatures)

# Calculate and print the max and min temperatures for the Warm dataset
max_warm_temp = max(warm_temperatures)
min_warm_temp = min(warm_temperatures)

print(f"Cold Dataset: Max Temp = {max_cold_temp}, Min Temp = {min_cold_temp}")
print(f"Warm Dataset: Max Temp = {max_warm_temp}, Min Temp = {min_warm_temp}")
