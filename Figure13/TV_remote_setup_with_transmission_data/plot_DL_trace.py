import pandas as pd
import matplotlib.pyplot as plt

def plot_all_models_and_trace(accuracy_file, trace_file):
    # Load the model accuracies dataset
    df_accuracies = pd.read_csv(accuracy_file)
    df_accuracies['Time (ms)'] = df_accuracies['Time (ns)'] / 1e6  # Convert nanoseconds to milliseconds

    # Load the trace dataset
    df_trace = pd.read_csv(trace_file)
    df_trace['Time (ms)'] = df_trace['Time (ns)'] / 1e6  # Convert nanoseconds to milliseconds

    # Check for required voltage columns
    if 'Channel A (V)' in df_trace.columns and 'Channel B (V)' in df_trace.columns:
        channel_A = df_trace['Channel A (V)']
        channel_B = df_trace['Channel B (V)']
        time_trace = df_trace['Time (ms)']
    else:
        raise ValueError("Trace file missing required columns 'Channel A (V)' and 'Channel B (V)'.")

    # Prepare to plot
    time_accuracies = df_accuracies['Time (ms)']
    models = ['Deep Learning', 'CatBoost', 'LightGBM', 'Random Forest', 'XGBoost', 'Decision Tree', 'Gradient Boosting']
    existing_models = [model for model in models if model in df_accuracies.columns]

    plt.figure(figsize=(16, 10))

    # Plot the model accuracies
    for model in existing_models:
        plt.plot(time_accuracies, df_accuracies[model], label=model, linewidth=2)

    # Add a secondary y-axis for the trace voltages
    ax1 = plt.gca()  # Get current axis for model accuracies
    ax2 = ax1.twinx()  # Create a twin axis for the voltage traces

    # Plot the traces
    ax2.plot(time_trace, channel_A, linewidth=1.5, alpha=0.2)
    ax2.plot(time_trace, channel_B, linewidth=1.5, alpha=0.2)

    # Explicitly set axis limits
    ax1.set_ylim(0, 1.03)  # Set limits for Accuracy (primary Y-axis)
    ax2.set_ylim(-0.5, 4.9)     # Set limits for Voltage (secondary Y-axis)

    # Customize axes
    ax1.set_xlabel('Time (ms)', fontsize=20)
    ax1.set_ylabel('Accuracy', fontsize=20)
    ax2.set_ylabel('Voltage (V)', fontsize=20)

    ax1.grid(True, axis='x')
    ax1.grid(True, axis='y')

    # Remove x-axis margins
    min_time = min(time_accuracies.min(), time_trace.min())
    max_time = max(time_accuracies.max(), time_trace.max())
    plt.xlim(min_time, max_time)

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, fontsize=18, loc='lower right')

    # Set x-axis ticks
    plt.xticks(ticks=range(int(min_time), int(max_time) + 1, 10), fontsize=18)
    # plt.yticks(fontsize=18)
    
    # Customize tick font sizes
    plt.xticks(fontsize=18)  # x-axis ticks
    ax1.tick_params(axis='y', labelsize=18)  # Primary y-axis ticks
    ax2.tick_params(axis='y', labelsize=18)  # Secondary y-axis ticks
    ax1.tick_params(axis='x', labelsize=18)
    plt.tight_layout()

    # Save and show the plot
    plt.savefig('TV_remote_setup_with_transmission_data.pdf', format='pdf', dpi=300)
    plt.show()

# File paths
accuracy_file = 'model_accuracies.csv'
trace_file = 'trace_num1.csv'  # Replace with the specific trace file
plot_all_models_and_trace(accuracy_file, trace_file)
