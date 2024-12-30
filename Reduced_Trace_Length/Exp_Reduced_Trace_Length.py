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

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data function with additional columns
def load_data(file_path, n):
    data = pd.read_csv(file_path, usecols=[1, 2])  # Load transmission and reception columns
    
    for col in data.columns:
        data[col] = np.where(data.index <= n, data[col], 0)
    
    data['col3'] = np.where(data.iloc[:, 1] > 4.4, np.where(data.index <= n, 1, 0), 0)
    data['col4'] = np.where(data.iloc[:, 1] < 0.2, np.where(data.index <= n, 1, 0), 0)
    data['col5'] = np.where(data.iloc[:, 0] > 1, np.where(data.index <= n, 1, 0), 0)

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

        self.fc1 = None
        self.fc2 = None
        self.fc3 = None
        self.fc_input_size = None

    def initialize_fc_layers(self, device):
        dummy_input = torch.zeros((1, 5, 65000)).to(device)
        with torch.no_grad():
            dummy_output = self.layer4(self.layer3(self.layer2(self.layer1(dummy_input))))
        self.fc_input_size = dummy_output.numel()

        self.fc1 = nn.Linear(self.fc_input_size, 512).to(device)
        self.fc2 = nn.Linear(512, 128).to(device)
        self.fc3 = nn.Linear(128, 20).to(device)

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

# Instantiate the model, define the loss function, and the optimizer
def create_model():
    model = CNNModel().to(device)
    model.initialize_fc_layers(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-3)
    return model, criterion, optimizer

# Validation function
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

# Training function
def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, patience):
    best_val_loss = float('inf')
    patience_counter = 0
    print("started training...")

    for epoch_idx in range(64):
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
            torch.save(model.state_dict(), 'model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch_idx + 1}")
                break

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

# Main script
all_model_accuracies = pd.DataFrame()

for n in range(0, 65001, 100):
    print(f"Run {n}")

    data, labels = load_all_data(n)

    scalers = [StandardScaler() for _ in range(5)]
    for j in range(5):
        data[:, :, j] = scalers[j].fit_transform(data[:, :, j])

    data = torch.FloatTensor(data).to(device)
    labels = torch.LongTensor(labels).to(device)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_indices, test_indices) in enumerate(kf.split(data)):
        print(f"Fold {fold + 1}")

        train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42)
        
        X_train, X_val, X_test = data[train_indices], data[val_indices], data[test_indices]
        y_train, y_val, y_test = labels[train_indices], labels[val_indices], labels[test_indices]
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)

        model, criterion, optimizer = create_model()
        train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, patience=11)
        model.load_state_dict(torch.load('model.pth'))

        accuracy = test(model, test_loader)
        fold_accuracies.append(accuracy)
        print(f"Fold {fold + 1} Test Accuracy: {accuracy:.4f}")

    all_model_accuracies = pd.concat([all_model_accuracies, pd.DataFrame({
        "Average": [np.mean(fold_accuracies)],
        "Stdev": [np.std(fold_accuracies)],
        "Median": [np.median(fold_accuracies)],
        "Minimum": [np.min(fold_accuracies)],
        "Maximum": [np.max(fold_accuracies)],
    }, index=[n])])

all_model_accuracies.to_csv('Exp_Reduced_Trace_Length.csv')
