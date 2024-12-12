# This script creates combinations of values to be tested in calibration step and loads them into an excel file
import numpy as np
import itertools
import pandas as pd
from pathlib import Path

#Variables for excel file
data = {
    'camera': ['cam1'],
    'image': ['20190731-201250.jpg'],
    # 'imagefolder': ['/hdd3/opensource/iceberg_tracking/data/sample_data/cam1'],
    'imagefolder': Path('G:/Glacier/GD_ICEH_iceHabitat/data/cam1'),
    'mask': ['mask_20190731-201250.shp'],
    'start_day': [20190724],
    'end_day': [20190726],
    'start_time': ['13:00'],
    'tracking_duration': [16],
    'tracking_interval': [60],
    'easting': [377280.39],
    'northing': [6525846.97],
    'elevation': [261.3],
    'antenna_height': [0],
    'sensor_width': [22.3],
    'image_width': [3456],
    'image_height': [2304],
    'crop_left': [0],
    'crop_right': [0],
    'crop_top': [0],
    'crop_bottom': [0],
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
#set max values based on min values
df ['sigma_max'] = df['sigma_min'] + 4
df ['theta_max'] = df['theta_min'] + 40
df ['phi_max'] = df['phi_min'] + 4
df ['psi_max'] = df['psi_min'] + 4

#combine dataframes
df1_repeated = pd.concat([df1]*len(df), ignore_index=True)
df_combined = pd.concat([df1_repeated, df], axis=1)
df_combined.dtypes

#TODO: Side quest - create calibration_input_2019 from code:
#create df_single
df_single=df_combined.head(1)
df_single.to_excel(Path('G:/Glacier/GD_ICEH_iceHabitat/data/calibration_input_2019_TEST.xlsx'), index=False)

#Save to excel
# df_combined.to_excel('data/calibration_combinations_all.xlsx', index=False)
df_combined.to_excel(Path('G:/Glacier/GD_ICEH_iceHabitat/data/calibration_combinations_all.xlsx'), index=False)
# df_combined.to_excel(Path(r'D:\U\Glacier\GD_ICEH_iceHabitat/JUNK_calibration_combinations_all.xlsx'), index=False)

