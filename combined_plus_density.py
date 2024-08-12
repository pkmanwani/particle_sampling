import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

def extract_data(h5_filename):
	# Open the HDF5 file
    with h5py.File(h5_filename, 'r') as f:
        # Read data from the HDF5 file
        #constants
        c=2.98e8 #m/s
        kpn1 = 53  # um
        point_factor = 1  # Define if it's used
        fraction = 0.01  # Define if it's used
        #
        energy = f['ene'][:]  # Total energy
        t = (np.array(f['x1']) - f.attrs['TIME'])/c # Time: t=z/c
        x = np.array(f['x2']) # Position x
        y =  np.array(f['x3']) # Position y
        p = np.array(f['p1']) # Momentum component along z
        xp = np.array(f['p2'])/p # slope along x(unit=mradians)
        yp= np.array(f['p3'])/p # slope along y(unit=mradians)
        #Additional data
        z = np.array(f['x1']) - f.attrs['TIME']
        q=np.array(f['q']) #the qs
        rel_q = point_factor * q * 20 / np.min(q)
        n0 = 1.0e22  # m^-3
        grid = 128 * 96 * 96  # grid dimensions
        vol = 12 * 2 * 2 * (kpn1 * 1e-6) ** 3 / (grid * fraction)  # m^3
        charge = np.sum(q) * vol * n0 * 1.6022e-19 * 1e12
        x_m = x * kpn1 * 1e-6
        y_m = y * kpn1 * 1e-6
        z_m = z * kpn1 * 1e-6
        # Prepare data for output in text format, suitable for txt
        data = {
            'x': x_m,
            'xp': xp,
            'y': y_m,
            'yp': yp,
            't': t,
            'p': p
        }

        additional={
            'x': x_m,
            'y': y_m,
            'z': z_m/c,
            'q': q/np.sum(q) 
        }

        return data, additional 

def systematic_resample(num, particles, weights):
    N = len(weights) #hd
    #print(np.arange(N))
    positions = (np.arange(num) + np.random.uniform(0, 1)) / num
    #print(positions)
    indexes = np.zeros(num, dtype=int)
    cumulative_sum = np.cumsum(weights)
    #print(cumulative_sum)
    i, j = 0, 0
    while i < num and j< N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return particles[indexes]

def stratified_resample(num, particles, weights):
    N = len(weights)
    positions = (np.random.uniform(0, 1, num) + np.arange(num)) / num
    indexes = np.zeros(num, dtype=int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < num and j<N:
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



def write_sdds(txt_filename, data):
    # Column headers based on your screenshot
    headers = ["x", "xp", "y", "yp", "t", "p"]
    # Write to file
    with open(txt_filename, 'w') as f:
        # Write the SDDS header
        f.write("SDDS1\n")
        f.write("&description contents=\"example\", &end\n")
        f.write("&column name=x, units=m, type=double, &end\n")
        f.write("&column name=xp, symbol=x', type=double, &end\n")
        f.write("&column name=y, units=m, type=double, &end\n")
        f.write("&column name=yp, symbol=y', type=double, &end\n")
        f.write("&column name=t, units=s, type=double, &end\n")
        f.write("&column name=p, units=\"m$be$nc\", type=double, &end\n")
        f.write("&data mode=ascii, &end\n")
        f.write("! page number 1\n")

        # Calculate the number of rows dynamically
        num_rows = len(data['x'])+1  # Assuming 'x' is representative of all columns
        f.write(f"{num_rows:4d}\n")  # '4d' ensures the number takes up 4 spaces

        zeros = "0.000000000000000e+00"  # 15 digits after decimal
        formatted_zeros = " ".join([zeros] * 6) + "\n"
        f.write(f"{formatted_zeros}")  # Writes zeros twice
        #f.write(f"{formatted_zeros}")

        rows = zip(*[data[h] for h in headers])
        for row in rows:
            first_line = " ".join(f"{v:.15e}" for v in row[:])
            #second_line = "\t".join(f"{v:.15e}" for v in row[3:])
            f.write(f"{first_line}\n") 
            #f.write(f"{second_line}\n")

#Following two are responsible for checking the resulting SDDS file's format. 
#Most common error is inconsistent line count causing an Ascii read mode issue
def is_ascii(file_path):
    with open(file_path, 'r') as file:
        try:
            file.read().encode('ascii')
        except UnicodeEncodeError:
            return False
        return True
def check_sdds_file(file_path):
    headers_expected = ['x', 'xp', 'y', 'yp', 't', 'p']
    num_columns = len(headers_expected)
    problems = []

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Check Headers
        if not lines[0].strip() == 'SDDS1':
            problems.append("Missing or incorrect file identifier; expected 'SDDS1'.")

        data_mode_found = False
        column_count = 0
        for line in lines:
            if '&column' in line:
                column_count += 1
            if '&data mode=ascii, &end' in line:
                data_mode_found = True
                break

        if not data_mode_found:
            problems.append("Data mode declaration missing or incorrect.")

        if column_count != num_columns:
            problems.append(f"Expected {num_columns} columns, found {column_count}.")

        # Check data rows
        data_start_index = lines.index('&data mode=ascii, &end\n') + 2
        num_rows_declared = int(lines[data_start_index].strip())
        data_lines = lines[data_start_index + 1:]

        if num_rows_declared != len(data_lines):
            problems.append(f"Declared number of rows ({num_rows_declared}) does not match actual rows ({len(data_lines)}).")

        # Check data format
        for i, data in enumerate(data_lines, start=1):
            data_parts = data.strip().split()
            if len(data_parts) != num_columns:
                problems.append(f"Row {i + data_start_index} has incorrect number of columns.")
            for part in data_parts:
                try:
                    float(part)  # Assuming all data should be valid floats
                except ValueError:
                    problems.append(f"Invalid number format in row {i + data_start_index}.")

    except Exception as e:
        problems.append(f"Error reading file: {e}")

    return problems

#Implementation: 
data, additional = extract_data(input('Name of your h5 file: '))
df = pd.DataFrame(data)
particles = df[['x', 'xp', 'y', 'yp', 't', 'p']].to_numpy()
weights = additional['q']
z=additional['z']
w_mean = weighted_mean(particles, weights)
w_std_dev = weighted_std(particles, weights, w_mean)
# Perform resampling
number=int(input('Number of particles to resample: '))
resampled_particles_systematic = systematic_resample(number, particles, weights)
resampled_particles_stratified = stratified_resample(number, particles, weights)

Density=True #if true, then density plot; else, then scatter plot; you SHOULD NOT use scatter plot
if Density==True:
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

    fig, axes = plt.subplots(2, 3, figsize=(21, 6))
    print("Shape of particles:", particles.shape)
    density1 = axes[0, 0].hexbin(particles[:, 0], particles[:, 2], C=weights,gridsize=50, cmap='viridis')
    axes[0, 0].set_title('Hexbin Density of Original Particles (x vs y)')
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Y Position')
    axes[0, 0].grid(True)
    cbar1 = fig.colorbar(density1, ax=axes[0, 0], orientation='vertical', label='Density')

    density2 = axes[0, 1].hexbin(resampled_particles_systematic[:, 0], resampled_particles_systematic[:, 2],gridsize=50, cmap='viridis')
    axes[0, 1].set_title('Systematic Resampling (x vs y)')
    axes[0, 1].set_xlabel('X Position')
    axes[0, 1].set_ylabel('Y Position')
    axes[0, 1].grid(True)
    
    density3 = axes[0, 2].hexbin(resampled_particles_stratified[:, 0], resampled_particles_stratified[:, 2],gridsize=50, cmap='viridis')
    axes[0, 2].set_title('Stratified Resampling (x vs y)')
    axes[0, 2].set_xlabel('X Position')
    axes[0, 2].set_ylabel('Y Position')
    axes[0, 2].grid(True)

    scatter4 = axes[1, 0].scatter(z, df['p'].to_numpy(), c=weights, cmap='plasma', alpha=0.6, edgecolors='w', linewidth=0.5)
    axes[1, 0].set_title('Original p vs. z')
    axes[1, 0].set_xlabel('Z Position')
    axes[1, 0].set_ylabel('P')
    axes[1, 0].grid(True)
    cbar4 = fig.colorbar(scatter4, ax=axes[1, 0], orientation='vertical', label='Weight')

    # Systematically resampled p vs. z plot
    resampled_indexes_systematic = systematic_resample(number, np.arange(len(df['p'])), weights)
    resampled_p_systematic = df['p'].to_numpy()[resampled_indexes_systematic]
    resampled_z_systematic = z[resampled_indexes_systematic]
    scatter5 = axes[1, 1].scatter(resampled_z_systematic, resampled_p_systematic, alpha=0.6, edgecolors='w', linewidth=0.5)
    axes[1, 1].set_title('Systematic Resampling (p vs z)')
    axes[1, 1].set_xlabel('Z Position')
    axes[1, 1].set_ylabel('P')
    axes[1, 1].grid(True)

    # Stratified resampled p vs. z plot
    resampled_indexes_stratified = stratified_resample(number, np.arange(len(df['p'])), weights)
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
else:
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
    fig, axes = plt.subplots(2, 3, figsize=(21, 6))

    # Original particles plot
    #scatter1 = axes[0, 0].scatter(particles[:, 0], particles[:, 2], c=weights, cmap='viridis', alpha=0.6, edgecolors='w', linewidth=0.5)
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
    resampled_indexes_systematic = systematic_resample(number, np.arange(len(df['p'])), weights)
    resampled_p_systematic = df['p'].to_numpy()[resampled_indexes_systematic]
    resampled_z_systematic = z[resampled_indexes_systematic]
    scatter5 = axes[1, 1].scatter(resampled_z_systematic, resampled_p_systematic, alpha=0.6, edgecolors='w', linewidth=0.5)
    axes[1, 1].set_title('Systematic Resampling (p vs z)')
    axes[1, 1].set_xlabel('Z Position')
    axes[1, 1].set_ylabel('P')
    axes[1, 1].grid(True)

    # Stratified resampled p vs. z plot
    resampled_indexes_stratified = stratified_resample(number, np.arange(len(df['p'])), weights)
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


#The following generate SDDS files based on the resampled data, respectively named 00 and 01

data['x'] = resampled_particles_systematic[:, 0]  # Update x_m with the x-values of the resampled particles
data['xp'] = resampled_particles_systematic[:, 1]
data['y'] = resampled_particles_systematic[:, 2]
data['yp'] = resampled_particles_systematic[:, 3]
data['t'] = resampled_particles_systematic[:, 4]
data['p'] = resampled_particles_systematic[:, 5]
write_sdds('output00.sdds', data)
print(is_ascii('output00.sdds'))
issues = check_sdds_file('output00.sdds')
if issues:
    print("Issues found in SDDS file:")
    for issue in issues:
        print(issue)
else:
    print("No issues found. File appears to be well-formed.")

data['x'] = resampled_particles_stratified[:, 0]  # Update x_m with the x-values of the resampled particles
data['xp'] = resampled_particles_stratified[:, 1]
data['y'] = resampled_particles_stratified[:, 2]
data['yp'] = resampled_particles_stratified[:, 3]
data['t'] = resampled_particles_stratified[:, 4]
data['p'] = resampled_particles_stratified[:, 5]

write_sdds('output01.sdds', data)
print(is_ascii('output01.sdds'))
issues = check_sdds_file('output01.sdds')
if issues:
    print("Issues found in SDDS file:")
    for issue in issues:
        print(issue)
else:
    print("No issues found. File appears to be well-formed.")









