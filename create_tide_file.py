import requests
import pickle
import pandas as pd
from pathlib import Path

''' Downloads tide data from NOAA at Tarr Inlet, you can change station number 
to retrieve data from other places. This code uses GMT for time zone. '''

# specify the pickle file path
pickle_file_path = Path('G:/Glacier/GD_ICEH_iceHabitat/data') #Path would take care of trailing slash
# pickle_file_path = Path(r'C:\Users\andyb\Documents\U/Glacier/GD_ICEH_iceHabitat/data') #Path would take care of trailing slash
# pickle_file_path = Path('/hdd3/opensource/iceberg_tracking/data')
output_filename = 'tide_2019.pickle'

# specify the base URL
base_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

# specify the parameters
params = {
    "product": "predictions",
    "application": "NOS.COOPS.TAC.WL",
    "begin_date": "20190724",  # replace with your start date
    "end_date": "20190731",  # replace with your end date. AKB adjusted to catch photos used for shapefiles
    "datum": "MLLW",
    #"station": "", replace with your station number
    "station": "9452749",  #Tarr Inlet, Glacier Bay
    # "station": "9452634",  #Elfin Cove (with real-time data)
    # "station": "9453220",  #Yakutat Bay (with real-time data)
    # "station": "9452584",  #Muir Inlet, Glacier Bay. Muir subordinate - different data output?
    # "station": "9452534",  #Bartlett Cove (and note that there are a few more in GLBA)
    # "station": "9453210",  #Point Latouche, Yakutat Bay (south of Haenke Island)
    "time_zone": "GMT",
    "units": "metric",
    "interval": "1",
    "format": "json"
}

# send the GET request
response = requests.get(base_url, params=params)

# check the status of the request
if response.status_code == 200:
    # parse the data from the response
    data = response.json()
else:
    print(f"Request failed with status code {response.status_code}")

# extract the 'predictions' list from the data
predictions = data['predictions']

# convert the list of dictionaries to a DataFrame
df = pd.DataFrame(predictions)

# rename the columns
df = df.rename(columns={'t': 'date', 'v': 'depth_tide_ellipsoid'})

df['date'] = pd.to_datetime(df['date'])

# save the DataFrame to the pickle file
df.to_pickle(pickle_file_path/output_filename) #pathlib overloads division '/'

#AKB NOTE: this saves depth_tide_ellipsoid as text rather than numeric. 
# s0_2_camera_calibration expects it as numeric. For now, leave this here, convert there.

##########
#plot the tide levels (AKB added)
##########
# df = pd.read_pickle(pickle_file_path/output_filename)
df['depth_tide_ellipsoid']=pd.to_numeric(df['depth_tide_ellipsoid'])
#or:
# df.loc[:,'depth_tide_ellipsoid']=pd.to_numeric(df['depth_tide_ellipsoid'])

import matplotlib.pyplot as plt
# df2=df.copy()
# df2.set_index('date', inplace=True) # Set 'date' as the index
# plt.figure(figsize=(10, 6))
# df2.plot(df2['depth_tide_ellipsoid'])
# plt.show()
# 
# df.plot(x=df['date'],y=df['depth_tide_ellipsoid'])
# plt.show()
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['depth_tide_ellipsoid'])
plt.xticks(rotation=45, ha='right')
plt.ylabel('Tide (m)')
plt.title(f'Tide at {params['station']}')
plt.grid(False)
plt.tight_layout() #helps, but not enough
plt.subplots_adjust(bottom=0.2,left=.15)
plt.show()

