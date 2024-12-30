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

# Define the CNN model
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
        dummy_input = torch.zeros((1, 5, 65000))
        dummy_output = self.layer4(self.layer3(self.layer2(self.layer1(dummy_input))))
        self.fc_input_size = dummy_output.numel()

# Load data function with downsampling simulation
def load_data(file_path, downsample_mimic_factor):
    print(f"[DEBUG] Loading data from {file_path} with downsample factor {downsample_mimic_factor}")
    data = pd.read_csv(file_path, usecols=[1, 2])  # Load transmission and reception columns

    # Create additional columns based on the conditions
    data['col3'] = np.where(data.iloc[:, 1] > 4.4, 1, 0)
    data['col4'] = np.where(data.iloc[:, 1] < 0.2, 1, 0)
    data['col5'] = np.where(data.iloc[:, 0] > 1, 1, 0)

    # Perform simulated downsampling
    if downsample_mimic_factor > 1:
        for col in data.columns:
            data[col] = mimic_downsampling(data[col], downsample_mimic_factor)

    print(f"[DEBUG] Finished loading data from {file_path}")
    return data[['Channel A (V)', 'Channel B (V)', 'col3', 'col4', 'col5']].values

# Helper function to mimic downsampling
def mimic_downsampling(series, factor):
    print(f"[DEBUG] Mimicking downsampling with factor {factor}")
    downsampled_series = series.copy()
    for i in range(0, len(series), factor):
        value_to_copy = downsampled_series[i]
        downsampled_series[i:i + factor] = value_to_copy
    return downsampled_series

# Load all data from files
def load_all_data(downsample_mimic_factor=1):
    print(f"[DEBUG] Loading all data with downsample factor {downsample_mimic_factor}")
    num_receivers = 20
    num_files_per_receiver = 100
    all_data = []
    labels = []
    
    for receiver in range(1, num_receivers + 1):
        print(f"[DEBUG] Processing Receiver {receiver}")
        directory = f'Receiver{receiver}'
        for i in range(1, num_files_per_receiver + 1):
            file_name = f'trace_num{i}.csv'
            file_path = os.path.join(directory, file_name)
            trace_data = load_data(file_path, downsample_mimic_factor=downsample_mimic_factor)
            all_data.append(trace_data)
            labels.append(receiver - 1)
    
    print(f"[DEBUG] Finished loading all data")
    return np.array(all_data), np.array(labels)

# Training with early stopping
def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, patience, device):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch_idx in range(64):  # Up to 64 epochs
        print(f"[DEBUG] Starting epoch {epoch_idx + 1}")
        model.train()
        train_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch_idx + 1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        print(f"[DEBUG] Epoch {epoch_idx + 1} completed. Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model.pth')  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[DEBUG] Early stopping at epoch {epoch_idx + 1}")
                break

# Validation function
def validate(model, val_loader, criterion, device):
    print(f"[DEBUG] Validating model")
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
    print(f"[DEBUG] Validation completed. Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")
    return avg_val_loss, val_accuracy

# Test function
def test(model, test_loader, device):
    print(f"[DEBUG] Testing model")
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
    
    accuracy = correct / total
    print(f"[DEBUG] Testing completed. Test Accuracy = {accuracy:.4f}")
    return accuracy

# Train and evaluate with K-fold cross-validation
def train_and_evaluate_with_kfold(downsample_factor, patience=11, n_splits=10):
    print(f"[DEBUG] Running experiment with downsample factor {downsample_factor} using {n_splits}-fold cross-validation")
    data, labels = load_all_data(downsample_mimic_factor=downsample_factor)
    scalers = [StandardScaler() for _ in range(5)]
    for i in range(5):
        print(f"[DEBUG] Standardizing column {i}")
        data[:, :, i] = scalers[i].fit_transform(data[:, :, i])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Using device: {device}")

    data = torch.FloatTensor(data).to(device)
    labels = torch.LongTensor(labels).to(device)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_val_indices, test_indices) in enumerate(kf.split(data)):
        print(f"[DEBUG] Fold {fold + 1}/{n_splits} started")
        train_indices, val_indices = train_test_split(train_val_indices, test_size=0.2, random_state=42)
        X_train, X_val, X_test = data[train_indices], data[val_indices], data[test_indices]
        y_train, y_val, y_test = labels[train_indices], labels[val_indices], labels[test_indices]

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)

        model = CNNModel().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-3)

        train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, patience, device)
        model.load_state_dict(torch.load(f'best_model.pth'))
        accuracy = test(model, test_loader, device)
        print(f"[DEBUG] Fold {fold + 1} Test Accuracy: {accuracy:.4f}")
        fold_accuracies.append(accuracy)

    average_accuracy = np.mean(fold_accuracies)
    std_dev_accuracy = np.std(fold_accuracies)
    median_accuracy = np.median(fold_accuracies)
    min_accuracy = np.min(fold_accuracies)
    max_accuracy = np.max(fold_accuracies)

    print(f"[DEBUG] Statistics for Downsample Factor {downsample_factor}:")
    print(f"  Average Test Accuracy: {average_accuracy:.4f}")
    print(f"  Standard Deviation: {std_dev_accuracy:.4f}")
    print(f"  Median Accuracy: {median_accuracy:.4f}")
    print(f"  Minimum Accuracy: {min_accuracy:.4f}")
    print(f"  Maximum Accuracy: {max_accuracy:.4f}")

    return average_accuracy, std_dev_accuracy, median_accuracy, min_accuracy, max_accuracy

# Run experiments for different downsample factors
downsample_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
results = []

for factor in downsample_factors:
    print(f"[DEBUG] Starting experiment for downsample factor {factor}")
    torch.cuda.empty_cache()  # Clear GPU memory
    avg_acc, std_dev, median_acc, min_acc, max_acc = train_and_evaluate_with_kfold(factor)
    results.append({
        'Downsample Factor': factor,
        'Average Accuracy': avg_acc,
        'Standard Deviation': std_dev,
        'Median Accuracy': median_acc,
        'Minimum Accuracy': min_acc,
        'Maximum Accuracy': max_acc
    })
    print(f"[DEBUG] Finished experiment for downsample factor {factor}")

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('model_accuracies_kfold_statistics.csv', index=False)
print("[DEBUG] Results with statistics saved to 'model_accuracies_kfold_statistics.csv'")
