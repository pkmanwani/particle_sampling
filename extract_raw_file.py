import h5py as h5
import numpy as np
import csv
import os
import glob

c = 299792458 #m/s
def create_csv(data_file,kpn1,fraction,csv_folder):
    # Open the HDF5 file
    with h5.File(data_file, 'r') as f:
        # Extract necessary data
        time = int(round(f.attrs['TIME'][0]))
        distance = round(time * kpn1 * 1e-4, 1)  # cm
        z = np.array(f['x1']) - f.attrs['TIME']
        x = np.array(f['x2'])
        y = np.array(f['x3'])
        pz = np.array(f['p1'])
        px = np.array(f['p2'])
        py = np.array(f['p3'])
        q = np.array(f['q'])

        # Calculate additional parameters
        x_p = px / pz
        y_p = py / pz
        rel_q = point_factor * q * 20 / np.min(q)
        n0 = 1.0e22  # m^-3
        grid = 128 * 96 * 96  # grid dimensions
        vol = 12 * 2 * 2 * (kpn1 * 1e-6) ** 3 / (grid * fraction)  # m^3
        charge = np.sum(q) * vol * n0 * 1.6022e-19 * 1e12
        x_m = x * kpn1 * 1e-6
        y_m = y * kpn1 * 1e-6
        z_m = z * kpn1 * 1e-6

        # Print charge for verification
        print(f"Total charge: {charge:.3f} pC")

        #Print distance
        print(f"Distance: {distance:.3f} cm")

        # Prepare data for CSV
        data = zip(x_m, x_p, y_m, y_p, z_m/c, pz, q/np.sum(q))


        csv_filename = data_file.split('\\')[-1][:-3] + '.csv'

        # Write data to CSV
        with open(os.path.join(csv_folder,csv_filename), mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write the header with charge information
            writer.writerow([f"Charge: {charge:.3f} pC"])
            writer.writerow([f"Distance: {distance:.3f} cm"])
            writer.writerow(['x', 'xp', 'y', 'yp','t','p', 'q'])

            # Write the data rows
            for row in data:
                writer.writerow(row)

    print(f"Data successfully written to {csv_filename}")

#RAW folder
folder_name = 'RAW'
#CSV folder
csv_folder = 'raw_csv'
# Define CSV file name

# Define your constants
driver_files = list(glob.glob('RAW/RAW-driver-*.h5'))
driver_files.sort(key=lambda path: int(path[-9:-3]))
data_name = "RAW-driver-000000.h5"  # Replace with your HDF5 file name
kpn1 = 53  # um
point_factor = 1  # Define if it's used
fraction = 0.01  # Define if it's used

for driver_file in driver_files:
    create_csv(driver_file,kpn1,fraction,csv_folder)