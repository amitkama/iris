import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold

# Load data function (no temperature channel)
def load_data(file_path):
    data = pd.read_csv(file_path, usecols=[1, 2])  # Load transmission and reception columns (Channel A and Channel B)

    # Create additional columns based on the conditions
    data['col3'] = np.where(data.iloc[:, 1] > 4.4, 1, 0)  # Reception > 4.4 condition
    data['col4'] = np.where(data.iloc[:, 1] < 0.2, 1, 0)  # Reception < 0.2 condition
    data['col5'] = np.where(data.iloc[:, 0] > 1, 1, 0)  # Transmission > 1 condition

    # Return only 5 channels now (excluding temperature)
    return data[['Channel A (V)', 'Channel B (V)', 'col3', 'col4', 'col5']].values

# Function to load all data from the specified directory (e.g., Cold or Warm)
def load_all_data_from_directory(base_directory):
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
            labels.append(receiver - 1)  # Assuming receiver is used as the label
    
    return np.array(all_data), np.array(labels)

# Define the CNN model (5 input channels)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=30, stride=10, padding=2),  # 5 input channels now
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
        dummy_input = torch.zeros((1, 5, 65000))  # Adjusted to 5 input channels
        dummy_output = self.layer4(self.layer3(self.layer2(self.layer1(dummy_input))))
        self.fc_input_size = dummy_output.numel()

# Instantiate the model, define the loss function and the optimizer with L2 regularization (weight decay)
def create_model():
    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-3)
    return model, criterion, optimizer

# Training function with early stopping
def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, patience):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch_idx in range(64):  # Running up to 64 epochs
        model.train()
        losses = []
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())

        avg_train_loss = np.mean(losses)
        val_loss, val_accuracy = validate(model, val_loader, criterion)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'model.pth')  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

# Validation function
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = correct / total
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss, val_accuracy

# Test function
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Run scenario with 10-fold cross-validation (Train on Warm, Test on Cold)
def run_scenario_10fold_warm_on_cold(train_data, train_labels, test_data, test_labels, scenario_name):
    print(f"\nRunning Scenario: {scenario_name} with 10-Fold Cross-Validation")
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(train_data)):
        print(f"Fold {fold + 1}")

        # Split data into train and validation sets
        train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42)
        
        X_train, X_val = train_data[train_indices], train_data[val_indices]
        y_train, y_val = train_labels[train_indices], train_labels[val_indices]
        
        # Dataset and DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(test_data, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Create new model, criterion, and optimizer
        model, criterion, optimizer = create_model()

        # Train the model with early stopping
        train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, patience=11)

        # Load the best model
        model.load_state_dict(torch.load('model.pth'))

        # Evaluate on the test set
        accuracy = test(model, test_loader)
        fold_accuracies.append(accuracy)
        print(f"Fold {fold + 1} Test Accuracy: {accuracy:.4f}")

    # Report final statistics
    average_accuracy = np.mean(fold_accuracies)
    std_dev_accuracy = np.std(fold_accuracies)
    median_accuracy = np.median(fold_accuracies)
    min_accuracy = np.min(fold_accuracies)
    max_accuracy = np.max(fold_accuracies)

    print(f"\nScenario {scenario_name} Results:")
    print(f"Average Test Accuracy: {average_accuracy:.4f}")
    print(f"Standard Deviation: {std_dev_accuracy:.4f}")
    print(f"Median Accuracy: {median_accuracy:.4f}")
    print(f"Minimum Test Accuracy: {min_accuracy:.4f}")
    print(f"Maximum Test Accuracy: {max_accuracy:.4f}")

# Load and preprocess data for Warm and Cold datasets
warm_data, warm_labels = load_all_data_from_directory('warm')
cold_data, cold_labels = load_all_data_from_directory('cold')

# Standardize the columns (5 channels now, excluding temperature)
scalers = [StandardScaler() for _ in range(5)]
for i in range(5):
    warm_data[:, :, i] = scalers[i].fit_transform(warm_data[:, :, i])
    cold_data[:, :, i] = scalers[i].transform(cold_data[:, :, i])  # Use the same scalers for cold data

# Convert data to PyTorch tensors
warm_data = torch.FloatTensor(warm_data)
warm_labels = torch.LongTensor(warm_labels)
cold_data = torch.FloatTensor(cold_data)
cold_labels = torch.LongTensor(cold_labels)

# Run the scenario with 10-fold cross-validation
run_scenario_10fold_warm_on_cold(warm_data, warm_labels, cold_data, cold_labels, "Train/Validate on Warm, Test on Cold")
