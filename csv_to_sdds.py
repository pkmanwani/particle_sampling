import pandas as pd
import os

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
        num_rows = len(data['x'])  # Assuming 'x' is representative of all columns
        f.write(f"{num_rows:4d}\n")  # Adjust to 6 spaces as needed

        # Formatting parameters
        col_format = "{: 22.15e}"  # 22 spaces total, to accommodate space for sign and decimal
        spacing = " " * 1  # Space between columns

        # Write zero lines
        #zeros = spacing.join([col_format.format(0.0)] * 6)
        #f.write(f"{zeros}\n")

        # Write data rows
        for i in range(0, num_rows, 3):
            row_chunk = [data[h][i:i+3] for h in headers]
            formatted_rows = [
                spacing.join([col_format.format(v) for v in row])
                for row in zip(*row_chunk)
            ]
            f.write(f"{formatted_rows[0]} \n")
            if len(formatted_rows) > 1:
                f.write(f"{formatted_rows[1]} \n")
            if len(formatted_rows) > 2:
                f.write(f"{formatted_rows[2]} \n")

# Define the path for resampled CSV files and SDDS output
resampled_folder_path = 'resampled_csv'
sdds_folder_path = 'resampled_sdds'

# Create the SDDS output folder if it doesn't exist
os.makedirs(sdds_folder_path, exist_ok=True)

# Iterate over each CSV file in the resampled folder
for file_name in os.listdir(resampled_folder_path):
    if file_name.endswith('.csv'):
        csv_path = os.path.join(resampled_folder_path, file_name)
        sdds_path = os.path.join(sdds_folder_path, file_name.replace('.csv', '.sdds'))

        # Load CSV data
        df = pd.read_csv(csv_path)

        # Prepare data for SDDS
        data = {
            "x": df['x'].tolist(),
            "xp": df['xp'].tolist(),
            "y": df['y'].tolist(),
            "yp": df['yp'].tolist(),
            "t": df['t'].tolist(),
            "p": df['p'].tolist()
        }

        # Write to SDDS file
        write_sdds(sdds_path, data)

        print(f"Converted {file_name} to SDDS format and saved as {os.path.basename(sdds_path)}")
