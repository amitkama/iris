import pandas as pd
import joblib
import numpy as np
import os
from scipy import stats
import time
import random
import torch

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to append statistical features
def append_statistical_features(features, origin):
    features_tensor = torch.tensor(features).to(device)  # Move features to GPU
    statistical_features = [
        torch.mean(features_tensor).item(),  # Mean
        torch.median(features_tensor).item(),  # Median
        torch.mode(features_tensor)[0].item() if len(features_tensor) > 0 else 0,  # Mode
        torch.std(features_tensor, unbiased=True).item(),  # Standard Deviation
        torch.var(features_tensor, unbiased=True).item(),  # Variance
        torch.min(features_tensor).item() if len(features_tensor) > 0 else 0,  # Minimum
        torch.max(features_tensor).item() if len(features_tensor) > 0 else 0,  # Maximum
        (torch.max(features_tensor) - torch.min(features_tensor)).item() if len(features_tensor) > 0 else 0,  # Range
        torch.quantile(features_tensor, 0.25).item() if len(features_tensor) > 0 else 0,  # 1st Quartile
        torch.quantile(features_tensor, 0.50).item() if len(features_tensor) > 0 else 0,  # 2nd Quartile (Median)
        torch.quantile(features_tensor, 0.75).item() if len(features_tensor) > 0 else 0,  # 3rd Quartile
        (torch.quantile(features_tensor, 0.75) - torch.quantile(features_tensor, 0.25)).item() if len(features_tensor) > 0 else 0,  # IQR
        torch.skew(features_tensor).item(),  # Skewness (can use torch-specific implementation)
        torch.kurtosis(features_tensor).item(),  # Kurtosis (can use torch-specific implementation)
        torch.sum(features_tensor).item(),  # Sum
        len(features_tensor)  # Count
    ]
    return origin + statistical_features  # Return the original list with statistical features appended

# Function to extract features from a single trace
def extract_features_from_trace(data):
    # Assuming data is a pandas DataFrame; convert necessary parts to torch Tensors
    above_4 = torch.tensor(data['Channel B (V)'] > 1).to(device)
    below_1 = torch.tensor(data['Channel B (V)'] < 1).to(device)

    # The following operations are tensorized
    transition_indices_above_4_to_below_1 = above_4 & below_1.roll(-1, dims=0)
    transition_indices_below_1_to_above_4_corrected = below_1 & above_4.roll(-1, dims=0)

    transition_times_above_4_to_below_1 = torch.tensor(data['Time (ns)'][transition_indices_above_4_to_below_1]).to(device)
    transition_times_below_1_to_above_4_corrected = torch.tensor(data['Time (ns)'][transition_indices_below_1_to_above_4_corrected]).to(device)

    first_above_1_index = torch.tensor(data.index[data['Channel A (V)'] > 1].min()).to(device)
    first_time_above_1_A = torch.tensor(data.loc[first_above_1_index, 'Time (ns)']).to(device)

    # Continue similarly tensorizing the rest of your calculations
    first_times_above_1_A_after_transitions = []
    first_indexes_above_1_A_after_transitions = []

    for transition_index in transition_indices_below_1_to_above_4_corrected[transition_indices_below_1_to_above_4_corrected].nonzero(as_tuple=True)[0].tolist():
        data_subset = data.loc[transition_index + 1:]
        first_above_1_indexes_after_transition = data_subset.index[data_subset['Channel A (V)'] > 1]
        if not first_above_1_indexes_after_transition.empty:
            first_above_1_index_after_transition = torch.tensor(first_above_1_indexes_after_transition.min()).to(device)
            first_time_above_1_A_after_transition = torch.tensor(data.loc[first_above_1_index_after_transition, 'Time (ns)']).to(device)
            first_times_above_1_A_after_transitions.append(first_time_above_1_A_after_transition)
            first_indexes_above_1_A_after_transitions.append(first_above_1_index_after_transition)

    # You will continue this pattern of tensorizing the calculations and statistical analysis
    
    # Extract and return features (example of how to tensorize extraction)
    features = []
    indexes1 = torch.tensor(transition_times_above_4_to_below_1.index.tolist()).to(device)
    times1 = transition_times_above_4_to_below_1.tolist()
    
    # Append features and apply statistical analysis
    for index in indexes1.tolist():
        features.append(index)
    features = append_statistical_features(indexes1.tolist(), features)
    
    # Similarly process the rest of your data
    return features

# Load the model from the joblib file (this remains on the CPU unless you switch to a GPU-compatible model)
joblib_file = "random_forest_model.joblib"
model = joblib.load(joblib_file)

# Collect all trace files from Receiver1 to Receiver20 folders
trace_files = []
for receiver in range(1, 21):  # Receiver1 to Receiver20
    directory = f'Receiver{receiver}'
    files_in_directory = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    trace_files.extend(files_in_directory)

# Randomly select 500 traces (or fewer if not enough traces)
random.seed(42)
num_traces = min(1000, len(trace_files))
selected_trace_files = random.sample(trace_files, num_traces)

# List to store inference times and trace information
inference_times = []
processed_traces = []

# Perform inference on the selected traces and measure inference time
for trace_file in selected_trace_files:
    try:
        # Load the trace data
        data = pd.read_csv(trace_file)

        # Extract features from the trace using GPU tensor operations
        features = extract_features_from_trace(data)

        # Convert the features into a DataFrame (expected format for the model)
        features_df = pd.DataFrame([features]).to(device)  # Move features to GPU

        # Measure inference time (note: RandomForest in scikit-learn cannot run on GPU)
        start_time = time.time()
        predicted_class = model.predict(features_df.cpu())  # Ensure to move back to CPU for model inference
        inference_time = time.time() - start_time  # Calculate inference time

        # Save inference time and trace file name
        inference_times.append(inference_time)
        processed_traces.append(trace_file)

    except Exception as e:
        print(f"Error processing {trace_file}: {e}")
        continue  # Skip the trace if any error occurs

# Save the inference times to a CSV file
df = pd.DataFrame({'Trace File': processed_traces, 'Inference Time (seconds)': inference_times})
df.to_csv('inference_times_ML_GPU_500.csv', index=False)

print("Inference times saved to 'inference_times_ML_GPU_500.csv'")
