import pandas as pd
import joblib
import numpy as np
import os
from scipy import stats
import time
import random

# Function to append statistical features
def append_statistical_features(features, origin):
    statistical_features = [
        np.mean(features),  # Mean
        np.median(features),  # Median
        stats.mode(features)[0][0] if len(features) > 0 else 0,  # Mode
        np.std(features, ddof=1),  # Standard Deviation
        np.var(features, ddof=1),  # Variance
        np.min(features) if len(features) > 0 else 0,  # Minimum
        np.max(features) if len(features) > 0 else 0,  # Maximum
        (np.max(features) - np.min(features)) if len(features) > 0 else 0,  # Range
        np.percentile(features, 25) if len(features) > 0 else 0,  # 1st Quartile
        np.percentile(features, 50) if len(features) > 0 else 0,  # 2nd Quartile (Median)
        np.percentile(features, 75) if len(features) > 0 else 0,  # 3rd Quartile
        (np.percentile(features, 75) - np.percentile(features, 25)) if len(features) > 0 else 0,  # IQR
        stats.skew(features),  # Skewness
        stats.kurtosis(features),  # Kurtosis
        np.sum(features),  # Sum
        len(features)  # Count
    ]
    return origin + statistical_features  # Return the original list with statistical features appended

# Function to extract features from a single trace
def extract_features_from_trace(data):
    above_4 = data['Channel B (V)'] > 1
    below_1 = data['Channel B (V)'] < 1
    transition_indices_above_4_to_below_1 = above_4 & below_1.shift(-1, fill_value=False)
    transition_indices_below_1_to_above_4_corrected = below_1 & above_4.shift(-1)

    transition_times_above_4_to_below_1 = data['Time (ns)'][transition_indices_above_4_to_below_1]
    transition_times_below_1_to_above_4_corrected = data['Time (ns)'][transition_indices_below_1_to_above_4_corrected]

    first_above_1_index = data.index[data['Channel A (V)'] > 1].min()
    first_time_above_1_A = data.loc[first_above_1_index, 'Time (ns)']

    first_times_above_1_A_after_transitions = []
    first_indexes_above_1_A_after_transitions = []
    for transition_index in transition_indices_below_1_to_above_4_corrected[transition_indices_below_1_to_above_4_corrected].index:
        data_subset = data.loc[transition_index + 1:]
        first_above_1_indexes_after_transition = data_subset.index[data_subset['Channel A (V)'] > 1]
        if not first_above_1_indexes_after_transition.empty:
            first_above_1_index_after_transition = first_above_1_indexes_after_transition.min()
            first_time_above_1_A_after_transition = data.loc[first_above_1_index_after_transition, 'Time (ns)']
            first_times_above_1_A_after_transitions.append(first_time_above_1_A_after_transition)
            first_indexes_above_1_A_after_transitions.append(first_above_1_index_after_transition)

    last_times_above_1_A_before_transitions = []
    last_indexes_above_1_A_before_transitions = []
    for transition_index in transition_indices_below_1_to_above_4_corrected[transition_indices_below_1_to_above_4_corrected].index:
        data_subset = data.loc[:transition_index - 1]
        last_above_1_indexes_before_transition = data_subset.index[data_subset['Channel A (V)'] > 1]
        if not last_above_1_indexes_before_transition.empty:
            last_above_1_index_before_transition = last_above_1_indexes_before_transition.max()
            last_time_above_1_A_before_transition = data.loc[last_above_1_index_before_transition, 'Time (ns)']
            last_times_above_1_A_before_transitions.append(last_time_above_1_A_before_transition)
            last_indexes_above_1_A_before_transitions.append(last_above_1_index_before_transition)

    combined_times = pd.concat([transition_times_above_4_to_below_1, transition_times_below_1_to_above_4_corrected, pd.Series([first_time_above_1_A])])
    combined_times = combined_times[:-1]
    relative_times = combined_times - first_time_above_1_A

    features = []

    indexes1 = combined_times.index.tolist()
    times1 = combined_times.values.tolist()
    for index in indexes1:
        features.append(index)
    features = append_statistical_features(indexes1, features)
    for time in times1:
        features.append(time)
    features = append_statistical_features(times1, features)

    indexes2 = relative_times.index.tolist()
    times2 = relative_times.values.tolist()
    for index in indexes2:
        features.append(index)
    features = append_statistical_features(indexes2, features)
    for time in times2:
        features.append(time)
    features = append_statistical_features(times2, features)

    indices_list1 = transition_times_above_4_to_below_1.index.tolist()
    values_list1 = transition_times_above_4_to_below_1.values.tolist()
    indices_list2 = transition_times_below_1_to_above_4_corrected.index.tolist()
    values_list2 = transition_times_below_1_to_above_4_corrected.values.tolist()
    values_difference = [a - b for a, b in zip(values_list2, values_list1)]
    indices_difference = [a - b for a, b in zip(indices_list2, indices_list1)]
    for i in range(len(indices_difference)):
        features.append(indices_difference[i])
    features = append_statistical_features(indices_difference, features)
    for j in range(len(values_difference)):
        features.append(values_difference[j])
    features = append_statistical_features(values_difference, features)

    beginning_chunk = first_times_above_1_A_after_transitions[:-1]
    beginning_chunk.insert(0, first_time_above_1_A)
    Processing_time_values = [a - b for a, b in zip(transition_times_above_4_to_below_1, beginning_chunk)]
    for k in range(len(Processing_time_values)):
        features.append(Processing_time_values[k])
    features = append_statistical_features(Processing_time_values, features)

    beginning_chunk_index = first_indexes_above_1_A_after_transitions[:-1]
    beginning_chunk_index.insert(0, first_above_1_index)
    Processing_time_indexes = [a - b for a, b in zip(transition_times_above_4_to_below_1.index.tolist(), beginning_chunk_index)]
    for t in range(len(Processing_time_indexes)):
        features.append(Processing_time_indexes[t])
    features = append_statistical_features(Processing_time_indexes, features)

    Processing_time_end_values = [a - b for a, b in zip(transition_times_below_1_to_above_4_corrected.values.tolist(), last_times_above_1_A_before_transitions)]
    for s in range(len(Processing_time_end_values)):
        features.append(Processing_time_end_values[s])
    features = append_statistical_features(Processing_time_end_values, features)

    Processing_time_end_indexes = [a - b for a, b in zip(transition_times_below_1_to_above_4_corrected.index.tolist(), last_indexes_above_1_A_before_transitions)]
    for u in range(len(Processing_time_end_indexes)):
        features.append(Processing_time_end_indexes[u])
    features = append_statistical_features(Processing_time_end_indexes, features)

    red_dots = transition_times_above_4_to_below_1[1:]
    green_dots = transition_times_below_1_to_above_4_corrected[:-1]
    indices_list3 = red_dots.index.tolist()
    values_list3 = red_dots.values.tolist()
    indices_list4 = green_dots.index.tolist()
    values_list4 = green_dots.values.tolist()
    values_difference_upper_chunk = [a - b for a, b in zip(values_list3, values_list4)]
    for l in range(len(values_difference_upper_chunk)):
        features.append(values_difference_upper_chunk[l])
    features = append_statistical_features(values_difference_upper_chunk, features)

    indices_difference_upper_chunk = [a - b for a, b in zip(indices_list3, indices_list4)]
    for h in range(len(indices_difference_upper_chunk)):
        features.append(indices_difference_upper_chunk[h])
    features = append_statistical_features(indices_difference_upper_chunk, features)

    return features

# Load the model from the joblib file
joblib_file = "random_forest_model.joblib"
model = joblib.load(joblib_file)

# Collect all trace files from Receiver1 to Receiver20 folders
trace_files = []
for receiver in range(1, 21):  # Receiver1 to Receiver20
    directory = f'Receiver{receiver}'
    files_in_directory = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    trace_files.extend(files_in_directory)

# Randomly select 1000 traces (or fewer if not enough traces)
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

        # Extract features from the trace
        features = extract_features_from_trace(data)

        # Convert the features into a DataFrame (expected format for the model)
        features_df = pd.DataFrame([features])

        # Measure inference time
        start_time = time.time()
        predicted_class = model.predict(features_df)  # Perform inference
        inference_time = time.time() - start_time  # Calculate inference time

        # Save inference time and trace file name
        inference_times.append(inference_time)
        processed_traces.append(trace_file)

    except Exception as e:
        print(f"Error processing {trace_file}: {e}")
        continue  # Skip the trace if any error occurs

# Save the inference times to a CSV file
df = pd.DataFrame({'Trace File': processed_traces, 'Inference Time (seconds)': inference_times})
df.to_csv('ml_inference_times.csv', index=False)

print("Inference times saved to 'ml_inference_times.csv'")
