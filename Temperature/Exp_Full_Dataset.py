import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data function (no temperature channel)
def load_data(file_path):
    data = pd.read_csv(file_path, usecols=[1, 2])  # Load transmission and reception columns (Channel A and Channel B)

    # Create additional columns based on the conditions
    data['col3'] = np.where(data.iloc[:, 1] > 4.4, 1, 0)  # Reception > 4.4 condition
    data['col4'] = np.where(data.iloc[:, 1] < 0.2, 1, 0)  # Reception < 0.2 condition
    data['col5'] = np.where(data.iloc[:, 0] > 1, 1, 0)  # Transmission > 1 condition

    return data[['Channel A (V)', 'Channel B (V)', 'col3', 'col4', 'col5']].values

# Function to load all data from the specified directory
def load_all_data_from_directory(base_directory, offset):
    num_receivers = 20
    num_files_per_receiver = 100
    all_data = []
    labels = []
    
    for receiver in range(1, num_receivers + 1):
        directory = os.path.join(base_directory, f'Receiver{receiver}')
        for i in range(1, num_files_per_receiver + 1):
            matching_files = [f for f in os.listdir(directory) if f.startswith(f"tracenum_{i}_") and f.endswith(".csv")]
            file_name = matching_files[0]
            file_path = os.path.join(directory, file_name)
            trace_data = load_data(file_path)
            all_data.append(trace_data)
            labels.append(receiver - 1 + offset)  # Offset labels for cold dataset
    
    return np.array(all_data), np.array(labels)

# Load and preprocess data for Warm and Cold datasets
warm_data, warm_labels = load_all_data_from_directory('cold', offset=0)
cold_data, cold_labels = load_all_data_from_directory('warm', offset=20)

# Combine warm and cold data
all_data = np.concatenate((warm_data, cold_data), axis=0)
all_labels = np.concatenate((warm_labels, cold_labels), axis=0)

# Standardize the columns
scalers = [StandardScaler() for _ in range(5)]
for i in range(5):
    all_data[:, :, i] = scalers[i].fit_transform(all_data[:, :, i])

# Convert data to PyTorch tensors and move to device
all_data = torch.FloatTensor(all_data).to(device)
all_labels = torch.LongTensor(all_labels).to(device)

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=30, stride=10, padding=2).to(device),
            nn.ReLU().to(device),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=30, stride=5, padding=2).to(device),
            nn.ReLU().to(device),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=30, stride=5, padding=2).to(device),
            nn.ReLU().to(device),
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=30, stride=5, padding=2).to(device),
            nn.ReLU().to(device),
        )
        self.flatten = nn.Flatten().to(device)
        self.dropout = nn.Dropout(0.95).to(device)
        self._initialize_fc_input_size()
        self.fc1 = nn.Linear(self.fc_input_size, 512).to(device)
        self.fc2 = nn.Linear(512, 128).to(device)
        self.fc3 = nn.Linear(128, 40).to(device)

    def forward(self, x):
        x = x.permute(0, 2, 1).to(device)
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
        dummy_input = torch.zeros((1, 5, 65000), device=device)
        dummy_output = self.layer4(self.layer3(self.layer2(self.layer1(dummy_input))))
        self.fc_input_size = dummy_output.numel()

# Instantiate model, loss, and optimizer
def create_model():
    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-3)
    return model, criterion, optimizer

# Run 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_indices, test_indices) in enumerate(kf.split(all_data)):
    train_data, test_data = all_data[train_indices], all_data[test_indices]
    train_labels, test_labels = all_labels[train_indices], all_labels[test_indices]

    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model, criterion, optimizer = create_model()
    
    for epoch in range(64):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    accuracy = sum((torch.argmax(model(inputs.to(device)), dim=1) == labels.to(device)).sum().item() for inputs, labels in test_loader) / len(test_labels)
    fold_accuracies.append(accuracy)
    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

print(f"Final Average Accuracy: {np.mean(fold_accuracies):.4f}")
