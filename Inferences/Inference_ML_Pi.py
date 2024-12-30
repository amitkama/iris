import pandas as pd
import joblib
import time
import random
import torch

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-processed features from the CSV file
features_df = pd.read_csv('complete_df_features3.csv')

# Drop the last column (assuming it is the label)
features_df = features_df.iloc[:, :-1]  # Drop the last column

# Convert the features DataFrame into a torch tensor and move it to the GPU
features_tensor = torch.tensor(features_df.values).to(device)

# Load the model from the joblib file (this remains on CPU unless switching to cuML)
joblib_file = "random_forest_model.joblib"
model = joblib.load(joblib_file)

# Randomly select 1000 indices (or fewer if not enough traces)
random.seed(42)
num_traces = min(1000, len(features_tensor))
selected_indices = random.sample(range(len(features_tensor)), num_traces)

# List to store inference times
inference_times = []

# Perform inference on the selected traces and measure inference time
for idx in selected_indices:
    try:
        # Get the features for the current trace (use the row corresponding to the index)
        features = features_tensor[[idx]].cpu().numpy()  # Move the tensor to CPU and convert to numpy

        # Measure inference time
        start_time = time.time()
        predicted_class = model.predict(features)  # Perform inference
        inference_time = time.time() - start_time  # Calculate inference time

        # Save inference time
        inference_times.append(inference_time)

    except Exception as e:
        print(f"Error processing trace index {idx}: {e}")
        continue  # Skip the trace if any error occurs

# Save the inference times to a CSV file
df = pd.DataFrame({'Trace Index': selected_indices, 'Inference Time (seconds)': inference_times})
df.to_csv('ml_inference_times_Pi_1000_new.csv', index=False)

print("Inference times saved to 'ml_inference_times_Pi_1000_new.csv'")
