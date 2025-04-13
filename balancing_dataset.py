import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

column_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

df = pd.read_csv("CVI-Project/driving_log.csv", names=column_names)

# Define threshold for "straight"
straight_threshold = 0.05

steering = df['steering']

# Keep only a small portion of straight-driving samples
straight = df[np.abs(steering) < straight_threshold]
straight = straight.sample(frac=0.1, random_state=42)  # Keep 10%

# Keep all turning samples
turning = df[np.abs(steering) >= straight_threshold]

# Combine
balanced_df = pd.concat([straight, turning])

print("Original:", len(df))
print("Balanced:", len(balanced_df))


# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(balanced_df['steering'], bins=30, color='skyblue', edgecolor='black')
plt.title("Steering Angle Distribution")
plt.xlabel("Steering Angle")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
