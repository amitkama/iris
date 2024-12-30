import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data function with adjustable trace length
def load_data(file_path, n):
    data = pd.read_csv(file_path, usecols=[2])  # Load reception column
    
    # Adjust the length of each column to keep the first 'n' rows and set the rest to 0
    data.iloc[n:] = 0

    # Create additional columns based on the conditions
    data['col2'] = np.where(data.iloc[:, 0] > 4.5, np.where(data.index < n, 1, 0), 0)
    data['col3'] = np.where(data.iloc[:, 0] < 0.2, np.where(data.index < n, 1, 0), 0)

    return data[['Channel B (V)', 'col2', 'col3']].values

# Load all data from files with adjustable trace length
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

# Define the CNN model adjusted for 3 input channels
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=30, stride=10, padding=2),
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

    def initialize_fc(self, device):
        """Initializes the fully connected layers dynamically after moving to device."""
        dummy_input = torch.zeros((1, 3, 65000), device=device)
        dummy_output = self.layer4(self.layer3(self.layer2(self.layer1(dummy_input))))
        fc_input_size = dummy_output.numel()

        self.fc1 = nn.Linear(fc_input_size, 512).to(device)
        self.fc2 = nn.Linear(512, 128).to(device)
        self.fc3 = nn.Linear(128, 20).to(device)

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

def create_model():
    model = CNNModel().to(device)
    model.initialize_fc(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-3)
    return model, criterion, optimizer

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = correct / total
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss, val_accuracy

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

def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, patience):
    best_val_loss = float('inf')
    patience_counter = 0
    print("started training...")

    for epoch_idx in range(128):  # Running up to 64 epochs
        model.train()
        losses = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())

        avg_train_loss = np.mean(losses)
        val_loss, val_accuracy = validate(model, val_loader, criterion)
        
        print(f'Epoch {epoch_idx + 1}: Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'Rx_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch_idx + 1}")
                break

# Initialize an empty DataFrame to collect accuracy results
all_model_accuracies = pd.DataFrame()

# Run for different trace lengths
for n in range(0, 65001, 100):
    print(f"Running for trace length {n}")
    data, labels = load_all_data(n)

    # Standardize data
    scalers = [StandardScaler() for _ in range(3)]
    for i in range(3):
        data[:, :, i] = scalers[i].fit_transform(data[:, :, i])
    
    data = torch.FloatTensor(data).to(device)
    labels = torch.LongTensor(labels).to(device)

    train_indices, test_indices = train_test_split(range(len(data)), test_size=0.2, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42)

    X_train, X_val, X_test = data[train_indices], data[val_indices], data[test_indices]
    y_train, y_val, y_test = labels[train_indices], labels[val_indices], labels[test_indices]

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model, criterion, optimizer = create_model()

    train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, patience=11)
    model.load_state_dict(torch.load('Rx_model.pth'))

    test_accuracy = test(model, test_loader)
    print(f"Test Accuracy for trace length {n}: {test_accuracy:.4f}")
    
    # Use pd.concat to add results
    results = {'Trace Length': [n], 'Accuracy': [test_accuracy]}
    results_df = pd.DataFrame(results)
    all_model_accuracies = pd.concat([all_model_accuracies, results_df], ignore_index=True)

# Save the results
all_model_accuracies.to_csv('trace_length_Rx_only.csv', index=False)
