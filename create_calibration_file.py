#AKB started to convert... need internet. need to add pandas to conda env
import numpy as np
import itertools
import pandas as pd
from pathlib import Path
# This script creates combinations of values to be tested in calibration step and loads them into an excel file

#Variables for excel file
data = {
    'camera': ['cam1'],
    'image': ['20190731-201250.jpg'],
    # 'imagefolder': ['/hdd3/opensource/iceberg_tracking/data/sample_data/cam1'],
    'imagefolder': Path('G:/Glacier/GD_ICEH_iceHabitat/data'),
    'start_day': [20190724],
    'end_day': [20190726],
    'start_time': ['13:00'],
    'tracking_duration': [16],
    'tracking_interval': [60],
    'sensor_width': [22.3],
    'easting': [377280.39],
    'northing': [6525846.97],
    'elevation': [261.3],
    'antenna_height': [0],
    'image_width': [3456],
    'image_height': [2304],
    'crop_left': [0],
    'crop_right': [0],
    'crop_top': [0],
    'crop_bottom': [0],
    'mask': ['mask_20190731-201250.shp']
}
df1 = pd.DataFrame(data)


#Create combinations
sigma_min = np.arange(17,19,1)
theta_min = np.arange(295,315,5)
phi_min = np.arange(0,10,1)
psi_min = np.arange(-3,0,1)

# result contains all possible combinations.
combinations = list(itertools.product(sigma_min,theta_min,phi_min,psi_min))

#Create dataframe
df = pd.DataFrame(combinations, columns=['sigma_min', 'theta_min', 'phi_min', 'psi_min'])
#AKB note: what are these edits for?
df ['sigma_max'] = df['sigma_min'] + 4
df ['theta_max'] = df['theta_min'] + 40
df ['phi_max'] = df['phi_min'] + 4
df ['psi_max'] = df['psi_min'] + 4

#combine dataframes
df1_repeated = pd.concat([df1]*len(df), ignore_index=True)
df_combined = pd.concat([df1_repeated, df], axis=1)

#Save to excel
# df_combined.to_excel('data/sample_data/calibration_combinations_all.xlsx', index=False)
df_combined.to_excel(Path('G:/Glacier/GD_ICEH_iceHabitat/data/sample_data/calibration_combinations_all.xlsx', index=False)
