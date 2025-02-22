import pandas as pd
import os
import numpy as np
from scipy import stats

def append_statistical_features(features, origin, name, receiver, i):
    statistical_features = [
        np.mean(features), np.median(features),
        stats.mode(features)[0][0] if len(features) > 0 else 0,  # Mode
        np.std(features, ddof=1), np.var(features, ddof=1),
        np.min(features) if len(features) > 0 else 0, np.max(features) if len(features) > 0 else 0,
        (np.max(features) - np.min(features)) if len(features) > 0 else 0,
        np.percentile(features, 25) if len(features) > 0 else 0,
        np.percentile(features, 50) if len(features) > 0 else 0,
        np.percentile(features, 75) if len(features) > 0 else 0,
        (np.percentile(features, 75) - np.percentile(features, 25)) if len(features) > 0 else 0,
        stats.skew(features), stats.kurtosis(features),
        np.sum(features), len(features)
    ]

    # Ensure column names are updated dynamically for every file
    if receiver == 1 and i == 1:  # Only update once per feature category
        column_names.extend([
            f'Mean_{name}', f'Median_{name}', f'Mode_{name}', f'Stdev_{name}',
            f'Variance_{name}', f'Min_{name}', f'Max_{name}', f'Range_{name}',
            f'1st_Quartile_{name}', f'2nd_Quartile_{name}', f'3rd_Quartile_{name}',
            f'IQR_{name}', f'Skewness_{name}', f'Kurtosis_{name}', f'Sum_{name}', f'Count_{name}'
        ])

    origin.extend(statistical_features)  # Modify in place

column_names = []  # List to store features names for importance score
for receiver in range(1, 21):

    # Assuming all files are in the same directory and follow the naming convention
    directory = f'Receiver{receiver}'  # Update this to the directory containing your files
    num_files = 100
    all_features = []  # List to store features for all files

    # Construct the file path
    print("starting {}", receiver)
    for i in range(1, num_files + 1):
        file_name = f'trace_num{i}.csv'
        file_path = os.path.join(directory, file_name)
        
        # Load the data
        data = pd.read_csv(file_path)

        # Identifying transitions for Channel B from above 4V to below 1V
        above_4 = data['Channel B (V)'] > 1
        below_1 = data['Channel B (V)'] < 1
        transition_indices_above_4_to_below_1 = above_4 & below_1.shift(-1, fill_value=False)
        transition_indices_below_1_to_above_4_corrected = below_1 & above_4.shift(-1)

        transition_times_above_4_to_below_1 = data['Time (ns)'][transition_indices_above_4_to_below_1]
        transition_times_below_1_to_above_4_corrected = data['Time (ns)'][transition_indices_below_1_to_above_4_corrected]
        
        # Identifying the first time Channel A was above 1V
        first_above_1_index = data.index[data['Channel A (V)'] > 1].min()
        first_time_above_1_A = data.loc[first_above_1_index, 'Time (ns)']

        # Lists to store the times and indexes when Channel A first exceeds 1V after each transition
        first_times_above_1_A_after_transitions = []
        first_indexes_above_1_A_after_transitions = []

        # Iterate through each transition index for Channel B from below 1V to above 4V
        for transition_index in transition_indices_below_1_to_above_4_corrected[transition_indices_below_1_to_above_4_corrected].index:
            # Find the part of the DataFrame after the current transition
            data_subset = data.loc[transition_index + 1:]
            
            # Find the first index where Channel A's voltage exceeds 1V in the subset
            first_above_1_indexes_after_transition = data_subset.index[data_subset['Channel A (V)'] > 1]
            if not first_above_1_indexes_after_transition.empty:
                first_above_1_index_after_transition = first_above_1_indexes_after_transition.min()
                # Record the time and index corresponding to this index
                first_time_above_1_A_after_transition = data.loc[first_above_1_index_after_transition, 'Time (ns)']
                first_times_above_1_A_after_transitions.append(first_time_above_1_A_after_transition)
                first_indexes_above_1_A_after_transitions.append(first_above_1_index_after_transition)

        # Lists to store the last times and indexes when Channel A exceeds 1V before each transition
        last_times_above_1_A_before_transitions = []
        last_indexes_above_1_A_before_transitions = []

        # Iterate through each transition index for Channel B from below 1V to above 4V
        for transition_index in transition_indices_below_1_to_above_4_corrected[transition_indices_below_1_to_above_4_corrected].index:
            # Find the part of the DataFrame before the current transition
            data_subset = data.loc[:transition_index - 1]
            
            # Find the last index where Channel A's voltage exceeds 1V in the subset
            last_above_1_indexes_before_transition = data_subset.index[data_subset['Channel A (V)'] > 1]
            if not last_above_1_indexes_before_transition.empty:
                last_above_1_index_before_transition = last_above_1_indexes_before_transition.max()
                # Record the time and index corresponding to this index
                last_time_above_1_A_before_transition = data.loc[last_above_1_index_before_transition, 'Time (ns)']
                last_times_above_1_A_before_transitions.append(last_time_above_1_A_before_transition)
                last_indexes_above_1_A_before_transitions.append(last_above_1_index_before_transition)

        combined_times = pd.concat([
            transition_times_above_4_to_below_1, 
            transition_times_below_1_to_above_4_corrected, 
            pd.Series([first_time_above_1_A])
        ])

        combined_times = combined_times[:-1]  # This excludes the last entry
        relative_times = combined_times - first_time_above_1_A

        features = []
        print("Column count:", len(column_names))
        print("Feature shape:", len(features))

        # Prepare features for combined_times
        indexes1 = combined_times.index.tolist()
        times1 = combined_times.values.tolist()
        
        for index in indexes1:
            features.append(index)  # Append index
            if receiver == 1 and i == 1:
                column_names.append(f'indexes1_{index}')

        print("Column count:", len(column_names))
        print("Feature shape:", len(features))
        
        append_statistical_features(indexes1, features, "indexes1", receiver, i)
        
        print("Column count:", len(column_names))
        print("Feature shape:", len(features))

        for time in times1:
            features.append(time)   # Append corresponding time
            if receiver == 1 and i == 1:
                column_names.append(f'times1_{time}')
        
        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))
        
        append_statistical_features(times1, features, "times1", receiver, i)

        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))
        
        # Prepare features for relative_times
        indexes2 = relative_times.index.tolist()
        times2 = relative_times.values.tolist()
        
        for index in indexes2:
            features.append(index)  # Append index
            if receiver == 1 and i == 1:
                column_names.append(f'indexes2_{index}')
        
        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))

        append_statistical_features(indexes2, features, "indexes2", receiver, i)

        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))
        
        for time in times2:
            features.append(time)   # Append corresponding time
            if receiver == 1 and i == 1:
                column_names.append(f'times2_{time}')

        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))
        
        append_statistical_features(times2, features, "times2", receiver, i)

        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))

        # Prepare features for 'chunks length'
        indices_list1 = transition_times_above_4_to_below_1.index.tolist()
        values_list1 = transition_times_above_4_to_below_1.values.tolist()

        indices_list2 = transition_times_below_1_to_above_4_corrected.index.tolist()
        values_list2 = transition_times_below_1_to_above_4_corrected.values.tolist()
        
        values_difference = [a - b for a, b in zip(values_list2, values_list1)]
        indices_difference = [a - b for a, b in zip(indices_list2, indices_list1)]

        for f in range(len(indices_difference)):
            features.append(indices_difference[f])  # Append index
            if receiver == 1 and i == 1:
                column_names.append(f'indices_difference_{f}')
        
        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))

        append_statistical_features(indices_difference, features, "indices_difference", receiver, i)

        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))
        
        for j in range(len(values_difference)):
            features.append(values_difference[j])  # Append value
            if receiver == 1 and i == 1:
                column_names.append(f'values_difference_{j}')

        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))
            
        append_statistical_features(values_difference, features, "values_difference", receiver, i)

        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))
        
        beginning_chunk = first_times_above_1_A_after_transitions[:-1]
        beginning_chunk.insert(0, first_time_above_1_A)
        Processing_time_values = [a - b for a, b in zip(transition_times_above_4_to_below_1, beginning_chunk)]
        
        for k in range(len(Processing_time_values)):
            features.append(Processing_time_values[k])  # Append value
            if receiver == 1 and i == 1:
                column_names.append(f'Processing_time_values_{k}')

        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))    
        
        append_statistical_features(Processing_time_values, features, "Processing_time_values", receiver, i)

        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))

        beginning_chunk_index = first_indexes_above_1_A_after_transitions[:-1]
        beginning_chunk_index.insert(0, first_above_1_index)
        Processing_time_indexes = [a - b for a, b in zip(transition_times_above_4_to_below_1.index.tolist(), beginning_chunk_index)]

        for t in range(len(Processing_time_indexes)):
            features.append(Processing_time_indexes[t])  # Append index
            if receiver == 1 and i == 1:
                column_names.append(f'Processing_time_indexes_{t}')

        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))

        append_statistical_features(Processing_time_indexes, features, "Processing_time_indexes", receiver, i)
        
        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))
        
        Processing_time_end_values = [a - b for a, b in zip(transition_times_below_1_to_above_4_corrected.values.tolist(), last_times_above_1_A_before_transitions)]
        for s in range(len(Processing_time_end_values)):
            features.append(Processing_time_end_values[s])  # Append value
            if receiver == 1 and i == 1:
                column_names.append(f'Processing_time_end_values_{s}')
        
        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))
    
        append_statistical_features(Processing_time_end_values, features, "Processing_time_end_values", receiver, i)

        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))

        Processing_time_end_indexes = [a - b for a, b in zip(transition_times_below_1_to_above_4_corrected.index.tolist(), last_indexes_above_1_A_before_transitions)]
        for u in range(len(Processing_time_end_indexes)):
            features.append(Processing_time_end_indexes[u])  # Append index
            if receiver == 1 and i == 1:
                column_names.append(f'Processing_time_end_indexes_{u}')
        
        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))
    
        append_statistical_features(Processing_time_end_indexes, features, "Processing_time_end_indexes", receiver, i)

        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))
        
        red_dots = transition_times_above_4_to_below_1[1:]
        green_dots = transition_times_below_1_to_above_4_corrected[:-1]
        
        indices_list3 = red_dots.index.tolist()
        values_list3 = red_dots.values.tolist()

        indices_list4 = green_dots.index.tolist()
        values_list4 = green_dots.values.tolist()

        values_difference_upper_chunk = [a - b for a, b in zip(values_list3, values_list4)]
        for l in range(len(values_difference_upper_chunk)):
            features.append(values_difference_upper_chunk[l])  # Append index
            if receiver == 1 and i == 1:
                column_names.append(f'values_difference_upper_chunk_{l}')
    
        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))
        
        append_statistical_features(values_difference_upper_chunk, features, "values_difference_upper_chunk", receiver, i)
        
        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))

        indices_difference_upper_chunk = [a - b for a, b in zip(indices_list3, indices_list4)]
        for h in range(len(indices_difference_upper_chunk)):
            features.append(indices_difference_upper_chunk[h])  # Append index
            if receiver == 1 and i == 1:
                column_names.append(f'indices_difference_upper_chunk_{h}')

        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))

        append_statistical_features(indices_difference_upper_chunk, features, "indices_difference_upper_chunk", receiver, i)

        print("Final column count:", len(column_names))
        print("Feature shape:", len(features))    
        
        # Append features for this file to the all_features list
        all_features.append(features)

    # After the loop, create a DataFrame with all features from all files
    # column_names = ['Feature_{}'.format(i+1) for i in range(len(all_features[0]))]  # Based on the first file's features
    print("Final column count:", len(column_names))
    print("Feature shape:", len(features))

    df_features_all = pd.DataFrame(all_features, columns=column_names)

    # Specify the path where you want to save the CSV file
    csv_file_path = f'Receiver{receiver}/complete_df_features_for_score.csv'  # Adjust this path as needed

    # Use the to_csv method to save the DataFrame to a CSV file
    df_features_all.to_csv(csv_file_path, index=False)
