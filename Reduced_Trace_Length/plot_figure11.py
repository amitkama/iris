import pandas as pd
import matplotlib.pyplot as plt

def plot_with_error_bars(file_path):
    # Load the dataset from the specified file path
    df = pd.read_csv(file_path)

    # Convert Time (ns) to milliseconds for better readability on the x-axis
    df['Time (ms)'] = df['Time (ns)'] / 1e6  # Convert nanoseconds to milliseconds
    time_values = df['Time (ms)'].values
    accuracy_column = 'Accuracy'  # Column name for accuracy
    stdev_column = 'Stdev'  # Column name for standard deviation

    # Calculate boundaries
    lower_bound = df[accuracy_column] - df[stdev_column]
    upper_bound = df[accuracy_column] + df[stdev_column]

    plt.figure(figsize=(14, 8))

    # Plot the accuracy values as a function of time
    line, = plt.plot(time_values, df[accuracy_column], label='Accuracy', linewidth=2)
    line_color = line.get_color()  # Capture the color of the line to reuse for markers

    # Add shaded area for accuracy ± stdev
    shaded_area = plt.fill_between(
        time_values, lower_bound, upper_bound,
        color=line_color, alpha=0.2, label='Accuracy ± Stdev'
    )

    # Highlight the first point (at 0 ms) with a marker
    if 0 in time_values:
        first_accuracy = df.loc[df['Time (ms)'] == 0, accuracy_column].values[0]
        first_stdev = df.loc[df['Time (ms)'] == 0, stdev_column].values[0]
        plt.plot(0, first_accuracy, marker='o', markersize=8, color=line_color)
        plt.errorbar(0, first_accuracy, yerr=first_stdev, fmt='o', color=line_color, capsize=5)

    # Add less visible black vertical lines at specified times (converted to milliseconds) with legend entries
    vertical_lines = {
        6053228 / 1e6: 'Avg Transmission Start',
        6265551 / 1e6: 'Avg Reception Start'
    }  # Convert nanoseconds to milliseconds and add labels
    vline_handles = []
    for x, label in vertical_lines.items():
        vline = plt.axvline(x=x, color='black', linestyle='--', linewidth=1, alpha=0.3, label=label)
        vline_handles.append(vline)

    # Add titles and labels
    plt.xlabel('Time (ms)', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.grid(True, axis='x')
    plt.grid(True, axis='y')  # Display horizontal grid lines only

    # Set x-ticks to show every 10 ms
    plt.xticks(ticks=range(0, int(max(time_values)) + 1, 10), fontsize=18)  # Step of 10 ms
    plt.yticks(fontsize=18)

    # Set the x-axis limits to remove margins and make it start from exactly 0 and end at the max time
    plt.xlim(0, max(time_values))

    # Adjust legend order explicitly
    handles, labels = plt.gca().get_legend_handles_labels()
    new_order = [
        labels.index('Accuracy'),
        labels.index('Accuracy ± Stdev'),
        labels.index('Avg Transmission Start'),
        labels.index('Avg Reception Start')
    ]
    plt.legend([handles[i] for i in new_order], [labels[i] for i in new_order], fontsize=18, loc='lower right')

    plt.tight_layout()

    # Save and show the plot
    plt.savefig('Figure11.pdf', format='pdf', dpi=300)
    plt.show()

# Path to the dataset
file_path = 'Figure11_data.csv'
plot_with_error_bars(file_path)
