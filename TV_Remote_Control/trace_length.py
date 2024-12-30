import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data function with additional columns, excluding column 0 (original time) and column 6
def load_data(file_path, n):
    data = pd.read_csv(file_path, usecols=[1, 2])  # Load transmission and reception columns
    
    # Adjust the length of each column to keep the first 'i' rows and set the rest to 0 for a length of 65000
    for col in data.columns:
        data[col] = np.where(data.index <= n, data[col], 0)
    
    # Create additional columns based on the conditions
    data['col3'] = np.where(data.iloc[:, 1] > 4.5, np.where(data.index <= n, 1, 0), 0)  # When reception > 4.5, mark 1, else 0
    data['col4'] = np.where(data.iloc[:, 1] < 0.2, np.where(data.index <= n, 1, 0), 0)  # When reception < 0.2, mark 1, else 0
    data['col5'] = np.where(data.iloc[:, 0] > 1.5, np.where(data.index <= n, 1, 0), 0)  # When transmission > 1.5, mark 1, else 0

    # Now we exclude column 0 (original time) and column 6
    return data[['Channel A (V)', 'Channel B (V)', 'col3', 'col4', 'col5']].values

# Load all data from files
def load_all_data(n):
    num_receivers = 20
    num_files_per_receiver = 100
    all_data = []
    labels = []
    
    for receiver in range(1, num_receivers + 1):
        directory = f'Receiver{receiver}'
        for i in range(1, num_files_per_receiver + 1):
            file_name = f'trace_num{i}.csv'
            file_path = os.path.join(directory, file_name)
            trace_data = load_data(file_path, n)
            all_data.append(trace_data)
            labels.append(receiver - 1)
    
    return np.array(all_data), np.array(labels)

# Define the CNN model excluding the unused columns
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=30, stride=10, padding=2),  # 5 input channels
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
            nn.Conv1d(256, 512, kernel_size=30, stride=5, padding=2),  # Added fourth layer
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
    
    # Helper function to compute the output size dynamically
    def _initialize_fc_input_size(self):
        # Create a dummy input tensor with the same number of channels and a small sequence length
        dummy_input = torch.zeros((1, 5, 65000)).to(next(self.parameters()).device)  # Ensure dummy input matches model's device
        dummy_output = self.layer4(self.layer3(self.layer2(self.layer1(dummy_input))))
        self.fc_input_size = dummy_output.numel()

# Validation function
def validate(model, val_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = correct / total
    return val_accuracy

# Training function with early stopping
def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, patience):
    print("In training")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch_idx in range(128):  # Running up to 64 epochs
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch_idx + 1}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Early stopping condition can be applied here based on validation accuracy/loss if needed.
        
# Test function
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Initialize an empty DataFrame to collect accuracy results
all_model_accuracies = pd.DataFrame()

for n in range(0, 65001, 100):
    print(f"Run {n}")

    # Load and preprocess the data
    data, labels = load_all_data(n)

    # Standardize the columns that are being used
    scalers = [StandardScaler() for _ in range(5)]  # We have 5 columns: 1, 2, col3, col4, col5
    for j in range(5):
        data[:, :, j] = scalers[j].fit_transform(data[:, :, j])

    # Convert data to PyTorch tensors
    data = torch.FloatTensor(data).to(device)
    labels = torch.LongTensor(labels).to(device)

    # Split data into train and validation sets for each run
    train_indices, test_indices = train_test_split(np.arange(data.shape[0]), test_size=0.2, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42)

    X_train, X_val, X_test = data[train_indices], data[val_indices], data[test_indices]
    y_train, y_val, y_test = labels[train_indices], labels[val_indices], labels[test_indices]

    # Dataset and DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Create new model, criterion, and optimizer for each run
    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-3)

    # Train the model
    train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, patience=11)

    # Test the model
    accuracy = test(model, test_loader)
    print(f"Run {n} Test Accuracy: {accuracy:.4f}")

    # Collect accuracy results for this iteration
    accuracies_df = pd.DataFrame({"Accuracy": [accuracy]}, index=[n])  # Corrected line
    all_model_accuracies = pd.concat([all_model_accuracies, accuracies_df])

# Save the accuracy results into a CSV file
all_model_accuracies.to_csv('trace_length.csv')
