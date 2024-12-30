import matplotlib.pyplot as plt

# X and Y values
x_values = [797.193, 398.597, 199.298, 99.649, 49.825, 24.912, 12.456, 6.228, 3.114, 1.557, 0.779, 0.389, 0.195, 0.097, 0.049, 0.024, 0.012]
y_values = [0.9925, 0.9900, 0.9875, 0.9800, 0.9650, 0.9400, 0.8725, 0.8375, 0.7300, 0.6500, 0.4575, 0.3025, 0.2200, 0.2025, 0.1650, 0.1500, 0.1150]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, marker='o')
plt.xlabel('Sampling Rate (kS/s)', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.title('Effect of the sampling rate on the classification accuracy')
plt.grid(True)
plt.savefig('downsampling_highres.pdf', format='pdf', dpi=300)
