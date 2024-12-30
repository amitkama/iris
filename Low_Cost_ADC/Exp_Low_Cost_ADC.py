import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Load data function with additional columns, excluding column 0 (original time) and column 6
def load_data(file_path):
    data = pd.read_csv(file_path, usecols=[1])  # Load the transmission data (single column)

    # Ensure all data has the same length by padding with zeros or truncating to 2000 rows
    padded_data = np.zeros(2000)
    trace_length = min(len(data), 2000)
    padded_data[:trace_length] = data.values.flatten()[:trace_length]

    # Check for NaNs or inf values in the padded data
    if np.isnan(padded_data).any() or np.isinf(padded_data).any():
        raise ValueError(f"Invalid values found in {file_path}")

    # Define the first and second halves of the padded data
    col1 = padded_data[:1000]
    col2 = padded_data[1000:2000]

    # Create additional columns based on conditions
    col3 = np.where(col1 > 240, 1, 0)  # Condition for col1 > 240
    col4 = np.where(col1 < 20, 1, 0)   # Condition for col1 < 20
    col5 = np.where(col2 > 75, 1, 0)   # Condition for col2 > 75

    # Combine the columns into a multi-column array
    stacked_data = np.column_stack([col1, col2, col3, col4, col5])

    # Pad with zeros to reach 65000 rows
    padding_length = 65000 - stacked_data.shape[0]
    padded_data = np.pad(stacked_data, ((0, padding_length), (0, 0)), 'constant')

    return padded_data

# Load all data from files
def load_all_data():
    num_receivers = 20
    num_files_per_receiver = 100
    all_data = []
    labels = []
    
    for receiver in range(1, num_receivers + 1):
        directory = f'Receiver{receiver}'
        for i in range(1, num_files_per_receiver + 1):
            file_name = f'trace_num{i}_revised.csv'
            file_path = os.path.join(directory, file_name)
            trace_data = load_data(file_path)
            all_data.append(trace_data)
            labels.append(receiver - 1)
    
    return np.array(all_data), np.array(labels)

# Load and preprocess the data
data, labels = load_all_data()

# Standardize the columns that are being used
scalers = [StandardScaler() for _ in range(5)]  # We have 5 columns: 1, 2, col3, col4, col5
for i in range(5):
    data[:, :, i] = scalers[i].fit_transform(data[:, :, i])

# Convert data to PyTorch tensors
data = torch.FloatTensor(data)
labels = torch.LongTensor(labels)

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the CNN model excluding the unused columns (same as before)
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
        
        # Compute the correct size for fc1 based on the input dimensions
        self._initialize_fc_input_size()

        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 20)

    def forward(self, x):
        x = x.to(device)  # Move inputs to the device
        x = x.permute(0, 2, 1)
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
        # Move the layers temporarily to the correct device
        self.layer1.to(device)
        self.layer2.to(device)
        self.layer3.to(device)
        self.layer4.to(device)
        
        # Create a dummy input tensor on the correct device
        dummy_input = torch.zeros((1, 5, 65000)).to(device)
        
        # Pass through the layers to determine the output size
        dummy_output = self.layer4(self.layer3(self.layer2(self.layer1(dummy_input))))
        
        # Set the fully connected layer input size
        self.fc_input_size = dummy_output.numel()    

# Adjust data processing to send inputs/labels to the GPU
def move_to_device(inputs, labels, device):
    return inputs.to(device), labels.to(device)

# Training function with early stopping (updated for CUDA)
def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, patience):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch_idx in range(128):  # Running up to 128 epochs
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = move_to_device(inputs, labels, device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Print epoch loss
        print(f"Epoch {epoch_idx + 1}, Loss: {running_loss:.4f}")

        # Validation phase
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = move_to_device(inputs, labels, device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

# Testing function (updated for CUDA)
def test(model, test_loader):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = move_to_device(inputs, labels, device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# K-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_val_indices, test_indices) in enumerate(kf.split(data)):
    print(f"Fold {fold + 1}")

    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.2, random_state=42)

    X_train, X_val, X_test = data[train_indices], data[val_indices], data[test_indices]
    y_train, y_val, y_test = labels[train_indices], labels[val_indices], labels[test_indices]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=8, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=8, shuffle=False)

    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-3)

    train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, patience=11)
    accuracy = test(model, test_loader)
    fold_accuracies.append(accuracy)
    print(f"Fold {fold + 1} Test Accuracy: {accuracy:.4f}")

# Calculate final statistics
average_accuracy = np.mean(fold_accuracies)
std_dev_accuracy = np.std(fold_accuracies)
median_accuracy = np.median(fold_accuracies)
min_accuracy = np.min(fold_accuracies)
max_accuracy = np.max(fold_accuracies)

# Print final statistics
print(f"\nFinal Cross-Validation Results:")
print(f"Average Test Accuracy: {average_accuracy:.4f}")
print(f"Standard Deviation: {std_dev_accuracy:.4f}")
print(f"Median Accuracy: {median_accuracy:.4f}")
print(f"Minimum Test Accuracy: {min_accuracy:.4f}")
print(f"Maximum Test Accuracy: {max_accuracy:.4f}")
