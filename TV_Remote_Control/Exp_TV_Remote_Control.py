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
    data = pd.read_csv(file_path, usecols=[1, 2])  # Load transmission and reception columns

    # Create additional columns based on the conditions
    data['col3'] = np.where(data.iloc[:, 1] > 4.5, 1, 0)  # When reception > 1, keep transmission time, else 0
    data['col4'] = np.where(data.iloc[:, 1] < 0.2, 1, 0)  # When reception < 1, keep transmission time, else 0
    data['col5'] = np.where(data.iloc[:, 0] > 1.5, 1, 0)  # When transmission > 1, keep transmission time, else 0

    # Now we exclude column 0 (original time) and column 6
    return data[['Channel A (V)', 'Channel B (V)', 'col3', 'col4', 'col5']].values

# Load all data from files
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
            labels.append(receiver - 1)
    
    return np.array(all_data), np.array(labels)

# Load and preprocess the data
data, labels = load_all_data()

# Standardize the columns that are being used
scalers = [StandardScaler() for _ in range(5)]  # We have 5 columns: 1, 2, col3, col4, col5
for i in range(5):
    data[:, :, i] = scalers[i].fit_transform(data[:, :, i])

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert data to PyTorch tensors and move to the device
data = torch.FloatTensor(data).to(device)
labels = torch.LongTensor(labels).to(device)

# Define the CNN model excluding the unused columns
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
        # Create a dummy input tensor and move it to the same device as the model
        dummy_input = torch.zeros((1, 5, 65000), device=next(self.parameters()).device)
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
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch_idx in range(128):
        model.train()
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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

# Implement 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_val_indices, test_indices) in enumerate(kf.split(data)):
    print(f"Fold {fold + 1}")

    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.2, random_state=42)
    
    X_train, X_val, X_test = data[train_indices], data[val_indices], data[test_indices]
    y_train, y_val, y_test = labels[train_indices], labels[val_indices], labels[test_indices]

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-3)

    train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, patience=11)

    accuracy = test(model, test_loader)
    fold_accuracies.append(accuracy)
    print(f"Fold {fold + 1} Test Accuracy: {accuracy:.4f}")

# Final statistics
average_accuracy = np.mean(fold_accuracies)
std_dev_accuracy = np.std(fold_accuracies)
median_accuracy = np.median(fold_accuracies)
min_accuracy = np.min(fold_accuracies)
max_accuracy = np.max(fold_accuracies)

print(f"\n10-Fold Cross-Validation Results:")
print(f"Average Test Accuracy: {average_accuracy:.4f}")
print(f"Standard Deviation: {std_dev_accuracy:.4f}")
print(f"Median Accuracy: {median_accuracy:.4f}")
print(f"Minimum Test Accuracy: {min_accuracy:.4f}")
print(f"Maximum Test Accuracy: {max_accuracy:.4f}")
