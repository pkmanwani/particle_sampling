import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_gaussian_particles(num_particles, mean_x, std_dev_x, mean_y, std_dev_y):
    x_positions = np.random.normal(loc=mean_x, scale=std_dev_x, size=num_particles)
    y_positions = np.random.normal(loc=mean_y, scale=std_dev_y, size=num_particles)
    particles = np.column_stack((x_positions, y_positions))
    return particles

def assign_linear_weights(particles):
    x_positions = particles[:, 0]
    sorted_indices = np.argsort(x_positions)
    weights = np.linspace(1, 0.1, len(x_positions))
    sorted_weights = np.zeros_like(weights)
    sorted_weights[sorted_indices] = weights
    return sorted_weights

def weighted_mean(particles, weights):
    return np.sum(weights[:, np.newaxis] * particles, axis=0) / np.sum(weights)

def weighted_std(particles, weights, weighted_mean):
    variance = np.sum(weights[:, np.newaxis] * (particles - weighted_mean)**2, axis=0) / np.sum(weights)
    return np.sqrt(variance)

# Parameters
num_particles = int(input('How many particles?'))
mean_x = 0.0
std_dev_x = 1.0
mean_y = 0.0
std_dev_y = 10.0

# Generate particles
particles = generate_gaussian_particles(num_particles, mean_x, std_dev_x, mean_y, std_dev_y)

# Assign linear weights based on x positions
weights = assign_linear_weights(particles)

# Normalize weights
normalized_weights = weights / np.sum(weights)

# Calculate unweighted mean and standard deviation
unw_mean = np.mean(particles, axis=0)
unw_std_dev = np.std(particles, axis=0)

# Calculate weighted mean and standard deviation
w_mean = weighted_mean(particles, normalized_weights)
w_std_dev = weighted_std(particles, normalized_weights, w_mean)

# Create a DataFrame and save to weighted CSV
df = pd.DataFrame(particles, columns=['x', 'y'])
df['q'] = normalized_weights
df.to_csv('weighted_particles.csv', index=False)

# Plot the particles
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Unweighted particles plot
scatter1 = axes[0].scatter(particles[:, 0], particles[:, 1], c='blue', alpha=0.6, edgecolors='w', linewidth=0.5)
axes[0].set_title('Unweighted Particles')
axes[0].set_xlabel('X Position')
axes[0].set_ylabel('Y Position')
axes[0].grid(True)

# Weighted particles plot
scatter2 = axes[1].scatter(particles[:, 0], particles[:, 1], c=normalized_weights, cmap='viridis', alpha=0.6, edgecolors='w', linewidth=0.5)
axes[1].set_title('Weighted Particles')
axes[1].set_xlabel('X Position')
axes[1].set_ylabel('Y Position')
axes[1].grid(True)

# Add color bar to show weights
cbar = fig.colorbar(scatter2, ax=axes[1], orientation='vertical', label='Weight')

# Display means and std deviations
print(f"Unweighted Mean: {unw_mean}")
print(f"Unweighted Standard Deviation: {unw_std_dev}")
print(f"Weighted Mean: {w_mean}")
print(f"Weighted Standard Deviation: {w_std_dev}")

plt.show()
