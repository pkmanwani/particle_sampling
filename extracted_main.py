import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Define folder containing CSV files
folder_path = 'raw_csv'  # Update this path as necessary

# Define folder to save resampled CSV files
resampled_folder_path = 'resampled_csv'

# Create resampled folder if it doesn't exist
os.makedirs(resampled_folder_path, exist_ok=True)

# Get all CSV files in the folder
file_paths = glob.glob(os.path.join(folder_path, '*.csv'))

c = 299792458  # Speed of light in m/s

def systematic_resample(particles, weights, num_particles, allow_duplication=True):
    """
    Resample particles systematically to a specified number of particles.
    Allows for duplication of particles based on their weights if allow_duplication is True.
    """
    if not allow_duplication:
        # Without duplication, perform a simple resampling
        indexes = np.random.choice(len(particles), size=num_particles, replace=False, p=weights / np.sum(weights))
        return particles[indexes]

    N = len(weights)
    positions = (np.arange(num_particles) + np.random.uniform(0, 1)) / num_particles
    indexes = np.zeros(num_particles, dtype=int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < num_particles:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return particles[indexes]


def stratified_resample(particles, weights, num_particles, allow_duplication=True):
    """
    Resample particles stratified to a specified number of particles.
    Allows for duplication of particles based on their weights if allow_duplication is True.
    """
    if not allow_duplication:
        # Without duplication, perform a simple resampling
        indexes = np.random.choice(len(particles), size=num_particles, replace=False, p=weights / np.sum(weights))
        return particles[indexes]

    N = len(weights)
    positions = (np.random.uniform(0, 1, num_particles) + np.arange(num_particles)) / num_particles
    indexes = np.zeros(num_particles, dtype=int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < num_particles:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return particles[indexes]


def weighted_mean(particles, weights):
    """Calculate the weighted mean of particles."""
    return np.sum(weights[:, np.newaxis] * particles, axis=0) / np.sum(weights)


def weighted_std(particles, weights, weighted_mean):
    """Calculate the weighted standard deviation of particles."""
    variance = np.sum(weights[:, np.newaxis] * (particles - weighted_mean) ** 2, axis=0) / np.sum(weights)
    return np.sqrt(variance)


# Ask the user to specify the number of particles for resampling
num_particles = int(input("Enter the number of particles for resampling: "))

# Ask the user if they want duplication
allow_duplication = input("Allow duplication during resampling? (yes/no): ").strip().lower() == 'yes'

# Iterate over each CSV file
for file_path in file_paths:
    # Load particles and weights from CSV
    df = pd.read_csv(file_path, skiprows=2)  # Skip the first two rows to get to the data

    # Extract particles (x, xp, y, yp), weights (q), and additional data (pz, z)
    particles = df[['x', 'xp', 'y', 'yp', 't', 'p']].to_numpy()
    weights = df['q'].to_numpy()
    z = df['t'].to_numpy() * c
    print('Number of particles:', len(z))

    # Extract Charge and Distance from the header
    with open(file_path, 'r') as file:
        header_lines = [next(file).strip() for _ in range(2)]
    charge_line = header_lines[0].split(":")[1].strip()  # Extract charge
    distance_line = header_lines[1].split(":")[1].strip()  # Extract distance
    charge = float(charge_line.split(" ")[0])  # Extract charge value
    distance = float(distance_line.split(" ")[0])  # Extract distance value

    # Calculate weighted mean and standard deviation
    w_mean = weighted_mean(particles, weights)
    w_std_dev = weighted_std(particles, weights, w_mean)

    # Perform resampling with the specified number of particles and duplication option
    resampled_particles_systematic = systematic_resample(particles, weights, num_particles, allow_duplication)
    resampled_particles_stratified = stratified_resample(particles, weights, num_particles, allow_duplication)

    # Calculate mean and standard deviation of each resampled distribution
    mean_systematic = np.mean(resampled_particles_systematic, axis=0)
    std_dev_systematic = np.std(resampled_particles_systematic, axis=0)

    mean_stratified = np.mean(resampled_particles_stratified, axis=0)
    std_dev_stratified = np.std(resampled_particles_stratified, axis=0)

    # Print results for each file
    print(f"File: {file_path}")
    print(f"Charge: {charge} pC, Distance: {distance} cm")
    print("Original:")
    print(f"Mean: {w_mean}")
    print(f"Standard Deviation: {w_std_dev}")

    print("Systematic Resampling:")
    print(f"Mean: {mean_systematic}")
    print(f"Standard Deviation: {std_dev_systematic}")

    print("\nStratified Resampling:")
    print(f"Mean: {mean_stratified}")
    print(f"Standard Deviation: {std_dev_stratified}")
    print("\n")

    # Plot the original and resampled particles along with pz vs. z
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    fig.suptitle(f'Particle Data for {os.path.basename(file_path)}')

    # Original particles plot
    scatter1 = axes[0, 0].scatter(particles[:, 0], particles[:, 2], c=weights, cmap='viridis', alpha=0.6, edgecolors='w', linewidth=0.5)
    axes[0, 0].set_title('Original Particles (x vs y)')
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Y Position')
    axes[0, 0].grid(True)
    cbar1 = fig.colorbar(scatter1, ax=axes[0, 0], orientation='vertical', label='Weight')

    # Systematically resampled particles plot
    scatter2 = axes[0, 1].scatter(resampled_particles_systematic[:, 0], resampled_particles_systematic[:, 2], alpha=0.6, edgecolors='w', linewidth=0.5)
    axes[0, 1].set_title('Systematic Resampling (x vs y)')
    axes[0, 1].set_xlabel('X Position')
    axes[0, 1].set_ylabel('Y Position')
    axes[0, 1].grid(True)

    # Stratified resampled particles plot
    scatter3 = axes[0, 2].scatter(resampled_particles_stratified[:, 0], resampled_particles_stratified[:, 2], alpha=0.6, edgecolors='w', linewidth=0.5)
    axes[0, 2].set_title('Stratified Resampling (x vs y)')
    axes[0, 2].set_xlabel('X Position')
    axes[0, 2].set_ylabel('Y Position')
    axes[0, 2].grid(True)

    # Original p vs. z plot
    scatter4 = axes[1, 0].scatter(z, df['p'].to_numpy(), c=weights, cmap='plasma', alpha=0.6, edgecolors='w', linewidth=0.5)
    axes[1, 0].set_title('Original p vs. z')
    axes[1, 0].set_xlabel('Z Position')
    axes[1, 0].set_ylabel('P')
    axes[1, 0].grid(True)
    cbar4 = fig.colorbar(scatter4, ax=axes[1, 0], orientation='vertical', label='Weight')

    # Systematically resampled p vs. z plot
    resampled_indexes_systematic = systematic_resample(np.arange(len(df['p'])), weights, num_particles, allow_duplication)
    resampled_p_systematic = df['p'].to_numpy()[resampled_indexes_systematic]
    resampled_z_systematic = z[resampled_indexes_systematic]
    scatter5 = axes[1, 1].scatter(resampled_z_systematic, resampled_p_systematic, alpha=0.6, edgecolors='w', linewidth=0.5)
    axes[1, 1].set_title('Systematic Resampling (p vs z)')
    axes[1, 1].set_xlabel('Z Position')
    axes[1, 1].set_ylabel('P')
    axes[1, 1].grid(True)

    # Stratified resampled p vs. z plot
    resampled_indexes_stratified = stratified_resample(np.arange(len(df['p'])), weights, num_particles, allow_duplication)
    resampled_p_stratified = df['p'].to_numpy()[resampled_indexes_stratified]
    resampled_z_stratified = z[resampled_indexes_stratified]
    scatter6 = axes[1, 2].scatter(resampled_z_stratified, resampled_p_stratified, alpha=0.6, edgecolors='w', linewidth=0.5)
    axes[1, 2].set_title('Stratified Resampling (p vs z)')
    axes[1, 2].set_xlabel('Z Position')
    axes[1, 2].set_ylabel('P')
    axes[1, 2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Ask the user if they want to save the resampled CSV files
    save_resampled = input("Would you like to save the resampled data to CSV files? (yes/no): ").strip().lower() == 'yes'

    # Save resampled data if the user agreed
    if save_resampled:
        # Create DataFrames for resampled data
        resampled_df_systematic = pd.DataFrame({
            'x': resampled_particles_systematic[:, 0],
            'xp': resampled_particles_systematic[:, 1],
            'y': resampled_particles_systematic[:, 2],
            'yp': resampled_particles_systematic[:, 3],
            't': resampled_particles_systematic[:, 4],
            'p': resampled_particles_systematic[:, 5]
        })

        resampled_df_stratified = pd.DataFrame({
            'x': resampled_particles_stratified[:, 0],
            'xp': resampled_particles_stratified[:, 1],
            'y': resampled_particles_stratified[:, 2],
            'yp': resampled_particles_stratified[:, 3],
            't': resampled_particles_stratified[:, 4],
            'p': resampled_particles_stratified[:, 5]
        })

        # Save systematic resampled data
        systematic_filename = os.path.join(resampled_folder_path, f"systematic_{os.path.basename(file_path)}")
        resampled_df_systematic.to_csv(systematic_filename, index=False)
        print(f"Systematic resampled data saved to {systematic_filename}")

        # Save stratified resampled data
        stratified_filename = os.path.join(resampled_folder_path, f"stratified_{os.path.basename(file_path)}")
        resampled_df_stratified.to_csv(stratified_filename, index=False)
        print(f"Stratified resampled data saved to {stratified_filename}")
