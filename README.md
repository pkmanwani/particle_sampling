**Particle resampler to handle weighted particles**

This is manual for converter program

Functionality: This script takes in a json and RAW data file, perform resampling on particle based on systematic and stratified scheme, plot the outputted distribution, and return two SDDS files corresponding for each RAW for simulations. 

Json file should contain information of beam particle, namely, n0 and fraction. Use RETURN, '', to use default value(n0=1.0e22) and fraction=0.01.  
RAW data file should contain all information of momentum and position components

User need to input: A folder name of the data folder and number of particles wished to be resampled, and name of any particular invidual RAW data file if one needs to see the density plot of resampled distribution. 

IMPORTANT: Json files and Folder of RAW data files should be in the same directory i.e.the data folder which user supply the name. The name of the RAW data folder must be named 'RAW' and the json file must end with '.json'. 

The program was named 'Neuercon' because there was an older version named 'Newcon' for a RAW to CSV converter and this is a even newer version of such converter now with graphics. After several update, it is now coined the name 'Neuestecon'. 

The programs comes with methods that ensures file type checking at each step, look for the messages in terminal to better diagnose. Common errors could be Ascii-reading or incorrect file content. 

The program has an open mind, it will take a folder of any file type and only process those with ending '.h5' and will inform you that it has skipped those what's not of this form for you to reflect on why you are feeding it non-raw data file despite the fact that it says 'RAW data folder'

The program aumatically creates a folder named 'sdds_files' inside the given data data folder to store all output files. 

The program can help you graph a particular individual resampled RAW file one at a time if user supply the file name in the folder(e.g. 'RAW-driver-000000.h5') and terminates if user entered ''(return), as plotting all of them resulting in arbitatrily many pop-up window graphs will be excessive. 

The outputted SDDS files has names: 'output{j}{x}.sdds' with integer j is indexing from 0 to number of .h5 files in your folder and x is binary(1 or 0) with 0 corresponding systemaitcally resampled SDDS and 1 corresponding to stratified resampled SDDS. e.g. the first raw data file generates two SDDS: 'output00.sdds' and 'output01.sdds' the former is systematic and the latter is stratified. 

If any additional and mischellaneous issues occured, one debug it oneself as rome was not built in a day and one's own work can only be perfected upon by the striving of his peers. 

The hierarchy of the file should be 
Current_directory >
	Neuestecon.py
	DATA_FOLDER > 
		~Json_file 
		~RAW  > 
			~input1.h5 
		    ~input2.h5
		        ...


example implementation could be the following: 

Number of particles to resample: 40000
Name of your DATA files folder: data1
True
No issues found. File appears to be well-formed.
True
No issues found. File appears to be well-formed.
Skipping non-h5 file: .DS_Store
Skipping non-h5 file: realTest.csv
True
No issues found. File appears to be well-formed.
True
No issues found. File appears to be well-formed.
To see plot of a particular one, enter name here(hit return if not): RAW-driver-000107.h5
Original:
Mean: [-8.78657249e-14 -2.04079370e-05  1.24113019e-13  5.57946479e-05
 -1.06125630e-09  4.94786216e+01]
Standard Deviation: [1.94400576e-11 2.52450956e-03 2.08410262e-11 2.80468748e-03
 5.33366403e-09 2.37524863e+01]
Systematic Resampling:
Mean: [-8.21008308e-14 -1.94933927e-05  1.25627165e-13  5.59863099e-05
 -1.05879505e-09  4.94888671e+01]
Standard Deviation: [1.94429119e-11 2.52735777e-03 2.08397812e-11 2.79812815e-03
 5.33846824e-09 2.37512250e+01]

Stratified Resampling:
Mean: [-8.17808731e-14 -1.53305963e-05  1.26530079e-13  5.94966948e-05
 -1.05841847e-09  4.94764658e+01]
Standard Deviation: [1.94325416e-11 2.52868466e-03 2.08336104e-11 2.80578275e-03
 5.33703264e-09 2.37509511e+01]
Shape of particles: (43797, 6)
To see plot of a particular one, enter name here(hit return if not): 
