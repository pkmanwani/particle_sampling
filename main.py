import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load particles and weights from CSV
df = pd.read_csv('weighted_particles.csv')
particles = df[['x', 'y']].to_numpy()
weights = df['q'].to_numpy()

def systematic_resample(particles, weights):
    N = len(weights)
    #print(np.arange(N))
    positions = (np.arange(N) + np.random.uniform(0, 1)) / N
    #print(positions)
    indexes = np.zeros(N, dtype=int)
    cumulative_sum = np.cumsum(weights)
    #print(cumulative_sum)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return particles[indexes]

def stratified_resample(particles, weights):
    N = len(weights)
    positions = (np.random.uniform(0, 1, N) + np.arange(N)) / N
    indexes = np.zeros(N, dtype=int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return particles[indexes]

def weighted_mean(particles, weights):
    return np.sum(weights[:, np.newaxis] * particles, axis=0) / np.sum(weights)

def weighted_std(particles, weights, weighted_mean):
    variance = np.sum(weights[:, np.newaxis] * (particles - weighted_mean)**2, axis=0) / np.sum(weights)
    return np.sqrt(variance)

# Calculate weighted mean and standard deviation
w_mean = weighted_mean(particles, weights)
w_std_dev = weighted_std(particles, weights, w_mean)
# Perform resampling
resampled_particles_systematic = systematic_resample(particles, weights)
resampled_particles_stratified = stratified_resample(particles, weights)

# Calculate mean and standard deviation of each resampled distribution
mean_systematic = np.mean(resampled_particles_systematic, axis=0)
std_dev_systematic = np.std(resampled_particles_systematic, axis=0)

mean_stratified = np.mean(resampled_particles_stratified, axis=0)
std_dev_stratified = np.std(resampled_particles_stratified, axis=0)

print("Original:")
print(f"Mean: {w_mean}")
print(f"Standard Deviation: {w_std_dev}")

print("Systematic Resampling:")
print(f"Mean: {mean_systematic}")
print(f"Standard Deviation: {std_dev_systematic}")

print("\nStratified Resampling:")
print(f"Mean: {mean_stratified}")
print(f"Standard Deviation: {std_dev_stratified}")

# Plot the original and resampled particles
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

# Original particles plot
scatter1 = axes[0].scatter(particles[:, 0], particles[:, 1], c=weights, cmap='viridis', alpha=0.6, edgecolors='w', linewidth=0.5)
axes[0].set_title('Original Particles')
axes[0].set_xlabel('X Position')
axes[0].set_ylabel('Y Position')
axes[0].grid(True)
cbar1 = fig.colorbar(scatter1, ax=axes[0], orientation='vertical', label='Weight')

# Systematically resampled particles plot
scatter2 = axes[1].scatter(resampled_particles_systematic[:, 0], resampled_particles_systematic[:, 1], alpha=0.6, edgecolors='w', linewidth=0.5)
axes[1].set_title('Systematically Resampled Particles')
axes[1].set_xlabel('X Position')
axes[1].set_ylabel('Y Position')
axes[1].grid(True)

# Stratified resampled particles plot
scatter3 = axes[2].scatter(resampled_particles_stratified[:, 0], resampled_particles_stratified[:, 1], alpha=0.6, edgecolors='w', linewidth=0.5)
axes[2].set_title('Stratified Resampled Particles')
axes[2].set_xlabel('X Position')
axes[2].set_ylabel('Y Position')
axes[2].grid(True)

plt.show()
