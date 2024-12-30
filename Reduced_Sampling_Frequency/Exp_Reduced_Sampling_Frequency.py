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
import matplotlib.pyplot as plt

# Load data function with downsampling simulation
def load_data(file_path, downsample_mimic_factor):
    data = pd.read_csv(file_path, usecols=[1, 2])  # Load transmission and reception columns

    # Create additional columns based on the conditions
    data['col3'] = np.where(data.iloc[:, 1] > 4.4, 1, 0)  # When reception > 4.4, set 1, else 0
    data['col4'] = np.where(data.iloc[:, 1] < 0.2, 1, 0)  # When reception < 0.2, set 1, else 0
    data['col5'] = np.where(data.iloc[:, 0] > 1, 1, 0)    # When transmission > 1, set 1, else 0

    # Perform simulated downsampling by replacing every nth value with the previous one
    if downsample_mimic_factor > 1:
        for col in data.columns:
            data[col] = mimic_downsampling(data[col], downsample_mimic_factor)

    # Now we exclude column 0 (original time) and column 6
    return data[['Channel A (V)', 'Channel B (V)', 'col3', 'col4', 'col5']].values

# Helper function to mimic downsampling
def mimic_downsampling(series, factor):
    """Simulates downsampling by replacing entire segments with the value at the start of the segment."""
    downsampled_series = series.copy()
    
    for i in range(0, len(series), factor):
        # Take the first value of each segment
        value_to_copy = downsampled_series[i]
        # Replace the entire segment with this value
        downsampled_series[i:i + factor] = value_to_copy
    
    return downsampled_series

# Load all data from files
def load_all_data(downsample_mimic_factor=1):
    num_receivers = 20
    num_files_per_receiver = 100
    all_data = []
    labels = []
    
    for receiver in range(1, num_receivers + 1):
        directory = f'Receiver{receiver}'
        for i in range(1, num_files_per_receiver + 1):
            file_name = f'trace_num{i}.csv'
            file_path = os.path.join(directory, file_name)
            trace_data = load_data(file_path, downsample_mimic_factor=downsample_mimic_factor)
            all_data.append(trace_data)
            labels.append(receiver - 1)
    
    return np.array(all_data), np.array(labels)

# Define the CNN model
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
        dummy_input = torch.zeros((1, 5, 65000))  # Based on the original length
        dummy_output = self.layer4(self.layer3(self.layer2(self.layer1(dummy_input))))
        self.fc_input_size = dummy_output.numel()

# Instantiate the model, define the loss function and the optimizer
def train_and_evaluate(downsample_factor, patience=11):
    print(f"Running experiment with downsample factor {downsample_factor}")
    
    # Load the downsampled data
    data, labels = load_all_data(downsample_mimic_factor=downsample_factor)
    
    # Standardize the columns that are being used
    scalers = [StandardScaler() for _ in range(5)]  # We have 5 columns: 1, 2, col3, col4, col5
    for i in range(5):
        data[:, :, i] = scalers[i].fit_transform(data[:, :, i])

    # Convert data to PyTorch tensors
    train_indices, test_indices = train_test_split(np.arange(data.shape[0]), test_size=0.2, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42)

    data = torch.FloatTensor(data)
    labels = torch.LongTensor(labels)

    X_train, X_val, X_test = data[train_indices], data[val_indices], data[test_indices]
    y_train, y_val, y_test = labels[train_indices], labels[val_indices], labels[test_indices]

    # Dataset and DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize the model, criterion, and optimizer
    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-3)

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

        for epoch_idx in range(64):  # Running up to 64 epochs
            model.train()
            losses = []
            
            for inputs, labels in tqdm(train_loader):
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
                torch.save(model.state_dict(), f'best_model_factor_{downsample_factor}.pth')  # Save the best model
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
    
    # Train the model with early stopping
    train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, patience=patience)

    # Load the best model
    model.load_state_dict(torch.load(f'best_model_factor_{downsample_factor}.pth'))

    # Evaluate on the test set
    accuracy = test(model, test_loader)
    print(f"Test Accuracy with downsample factor {downsample_factor}: {accuracy:.4f}")
    return accuracy

# Run experiments for different downsample factors
downsample_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
# downsample_factors = [1]
accuracies = []

for factor in downsample_factors:
    accuracy = train_and_evaluate(factor)
    accuracies.append(accuracy)

# Plotting the results
sampling_rates = [65000 // factor for factor in downsample_factors]

plt.figure(figsize=(8, 6))
plt.plot(sampling_rates, accuracies, marker='o')
plt.title('Accuracy vs Sampling Rate')
plt.xlabel('Sampling Rate (Number of Samples)')
plt.ylabel('Accuracy')
plt.title('Effect of Downsampling on Test Accuracy')
plt.grid(True)
plt.savefig('downsampling.png')