import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import random
import time

# Assuming the CNNModel is defined here or you have imported it from another file
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=30, stride=10, padding=2),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=30, stride=5, padding=2),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=30, stride=5, padding=2),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=30, stride=5, padding=2),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.95)

        self._initialize_fc_input_size()

        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 20)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute to [batch_size, num_channels, sequence_length]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        x = self.dropout(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def _initialize_fc_input_size(self):
        dummy_input = torch.zeros((1, 5, 65000))  # 5 input channels, 65000 sequence length
        dummy_output = self.layer4(self.layer3(self.layer2(self.layer1(dummy_input))))
        self.fc_input_size = dummy_output.numel()

# Load data function
def load_data(file_path):
    data = pd.read_csv(file_path, usecols=[1, 2])  # Load transmission and reception columns
    data['col3'] = np.where(data.iloc[:, 1] > 4.4, 1, 0)  # When reception > 4.4, mark 1, else 0
    data['col4'] = np.where(data.iloc[:, 1] < 0.2, 1, 0)  # When reception < 0.2, mark 1, else 0
    data['col5'] = np.where(data.iloc[:, 0] > 1, 1, 0)    # When transmission > 1, mark 1, else 0

    # Return selected columns
    return data[['Channel A (V)', 'Channel B (V)', 'col3', 'col4', 'col5']].values

# Load all data function
def load_all_data():
    num_receivers = 20
    num_files_per_receiver = 100
    all_data = []
    labels = []
    
    for receiver in range(1, num_receivers + 1):
        directory = f'Receiver{receiver}'
        for i in range(1, num_files_per_receiver + 1):
            file_name = f'trace_num{i}.csv'
            file_path = os.path.join(directory, file_name)
            trace_data = load_data(file_path)
            all_data.append(trace_data)
            labels.append(receiver - 1)  # Label for each receiver
    
    return np.array(all_data), np.array(labels)

# Preprocess the data
data, labels = load_all_data()

# Standardize the data
scalers = [StandardScaler() for _ in range(5)]  # 5 columns to scale
for i in range(5):
    data[:, :, i] = scalers[i].fit_transform(data[:, :, i])

# Convert data and labels to PyTorch tensors
data = torch.FloatTensor(data)
labels = torch.LongTensor(labels)

# Create the test set (no train/validation splitting since we're using only test set here)
test_dataset = TensorDataset(data, labels)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the trained model
model = CNNModel()
model.load_state_dict(torch.load('1DCNN.pth', weights_only=True))  # Load model weights only
model.eval()  # Set the model to evaluation mode

# Randomly select 1000 traces or the maximum available if fewer than 1000
random.seed(42)  # For reproducibility
total_test_size = len(test_loader.dataset)  # Total number of test samples
num_traces = min(1000, total_test_size)  # Choose 1000 or the total number of test samples, whichever is smaller
random_indices = random.sample(range(total_test_size), num_traces)  # Randomly select indices

# Prepare DataLoader for the selected traces
selected_traces = torch.utils.data.Subset(test_dataset, random_indices)
selected_loader = DataLoader(selected_traces, batch_size=1, shuffle=False)

# Function to measure inference time for a single trace
def measure_inference_time(model, data_loader):
    inference_times = []
    for inputs, _ in data_loader:
        start_time = time.time()  # Start timer
        with torch.no_grad():
            _ = model(inputs)  # Perform inference
        inference_time = time.time() - start_time  # Calculate time taken
        inference_times.append(inference_time)
    return inference_times

# Measure inference time for the selected 500 traces
inference_times = measure_inference_time(model, selected_loader)

# Save inference times to a CSV file
df = pd.DataFrame({'Trace Index': random_indices, 'Inference Time (seconds)': inference_times})
df.to_csv('inference_times.csv', index=False)

print("Inference times saved to 'inference_times_DLPi_2000.csv'")
