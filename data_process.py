import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
data = pd.read_csv("data.csv", header=None)

# Extract column 0 and convert to a list
data0 = data[0].tolist()

# Plot the original time series
plt.figure(figsize=(12, 6))
plt.plot(data0, label="Original Time Series", color="blue", linewidth=0.7)
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Original Time Series")
plt.legend()
plt.grid()
plt.show()

# Create sequences of length 3072 with a step of 50
seq = []
for i in range(0, len(data0), 50):
    if (i + 3072) > len(data0):
        break
    array = np.asarray(data0[i:i+3072])
    seq.append(array)

# Reshape the sequence array
seq = np.asarray(seq).reshape(len(seq), 3072)
seq = seq[:3072, :]

# Save the resulting data to a text file
np.savetxt("original_segmented.txt", seq, fmt='%f', delimiter=',')

# Plot the first few segmented sequences for visualization
plt.figure(figsize=(12, 6))
for i in range(min(5, len(seq))):  # Plot the first 5 sequences
    plt.plot(seq[i], label=f"Segment {i+1}")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("First Few Segmented Sequences")
plt.legend()
plt.grid()
plt.show()