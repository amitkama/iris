import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load data function with additional columns
def load_data(file_path):
    data = pd.read_csv(file_path, usecols=[2])  # Load reception column

    # Create additional columns based on the conditions
    data['col2'] = np.where(data.iloc[:, 0] > 4.4, 1, 0)  # When reception > 4.4, mark 1, else 0
    data['col3'] = np.where(data.iloc[:, 0] < 0.2, 1, 0)  # When reception < 0.2, mark 1, else 0

    # Return the selected columns as values
    return data[['Channel B (V)', 'col2', 'col3']].values

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
scalers = [StandardScaler() for _ in range(3)]  # We have 3 columns: Channel B (V), col2, col3
for i in range(3):
    data[:, :, i] = scalers[i].fit_transform(data[:, :, i])

# Convert data to PyTorch tensors
data = torch.FloatTensor(data)
labels = torch.LongTensor(labels)

# Define the CNN model adjusted for 3 input channels
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=30, stride=10, padding=2),  # 3 input channels
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
        dummy_input = torch.zeros((1, 3, 65000))  # Updated to 3 channels
        dummy_output = self.layer4(self.layer3(self.layer2(self.layer1(dummy_input))))
        self.fc_input_size = dummy_output.numel()

# Instantiate the model, define the loss function and the optimizer with L2 regularization (weight decay)
def create_model():
    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-3)  # L2 regularization with weight decay
    return model, criterion, optimizer

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

# Training function with early stopping
def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, patience):
    best_val_loss = float('inf')
    patience_counter = 0
    print("started training...")

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
        
        print(f'Epoch {epoch_idx + 1}: Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        
        # Early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_time_reception_extra_model.pth')  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch_idx + 1}")
                break

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

# Implement 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_indices, test_indices) in enumerate(kf.split(data)):
    print(f"Fold {fold + 1}")

    # Split data into train and validation sets
    train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42)
    
    X_train, X_val, X_test = data[train_indices], data[val_indices], data[test_indices]
    y_train, y_val, y_test = labels[train_indices], labels[val_indices], labels[test_indices]
    
    # Dataset and DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Create new model, criterion, and optimizer for each fold
    model, criterion, optimizer = create_model()

    # Train the model with early stopping
    train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, patience=11)

    # Load the best model
    model.load_state_dict(torch.load('best_time_reception_extra_model.pth'))

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

print(f"Average Test Accuracy: {average_accuracy:.4f}")
print(f"Standard Deviation: {std_dev_accuracy:.4f}")
print(f"Median Accuracy: {median_accuracy:.4f}")
print(f"Minimum Test Accuracy: {min_accuracy:.4f}")
print(f"Maximum Test Accuracy: {max_accuracy:.4f}")