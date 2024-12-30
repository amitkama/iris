import pandas as pd

# Load the uploaded CSV file
file_path = 'Results_GPU_DL_1000.csv'
data = pd.read_csv(file_path)

# Remove the first column
data = data.iloc[:, 1:]

# Calculate the avg and stdev for all 500 rows
avg_all_rows = data.mean()
stdev_all_rows = data.std()

# Calculate the avg and stdev for the last 400 rows
avg_last_400 = data.tail(400).mean()
stdev_last_400 = data.tail(400).std()

print("Average and standard deviation for all 500 rows:")
print(avg_all_rows)
print(stdev_all_rows)
print("Average and standard deviation for the last 400 rows:")
print(avg_last_400)
print(stdev_last_400)
