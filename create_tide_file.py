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
    "end_date": "20190726",  # replace with your end date
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

########
#AKB added
#########
dfp = pd.read_pickle(pickle_file_path/output_filename)
df=dfp
# was having trouble plotting due to type:
type(df['depth_tide_ellipsoid'])
# <class 'pandas.core.series.Series'>
type(df['depth_tide_ellipsoid'][1])
# <class 'str'>
df.dtypes
df['depth_tide_ellipsoid'].dtypes

#astype(str)
#to_numeric
df['depth_tide_ellipsoidNum']=pd.to_numeric(df['depth_tide_ellipsoid'])
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead
df[:,'depth_tide_ellipsoidNum']=pd.to_numeric(df['depth_tide_ellipsoid'])
#raise InvalidIndexError(key)
df.loc[:,'depth_tide_ellipsoidNum']=pd.to_numeric(df['depth_tide_ellipsoid'])
#worked
df.loc[:,'depth_tide_ellipsoidAlt']=df['depth_tide_ellipsoid']
# df['depth_tide_ellipsoidAlt']=pd.to_numeric(df['depth_tide_ellipsoid'])
df.loc[:,'depth_tide_ellipsoid']=pd.to_numeric(df['depth_tide_ellipsoid'])

import matplotlib.pyplot as plt
# # plot the tide levels
# df.set_index('date', inplace=True) # Set 'date' as the index
# df.plot(df.index,df['depth_tide_ellipsoid'])
# 
# df.plot(x=df['date'],y=df['depth_tide_ellipsoid'])
# plt.show()
# 
# df.plot()
# plt.show()
df2=df
df=df2
df=df[0:50]
plt.figure(figsize=(10, 6))
# plt.plot(df.index, df['depth_tide_ellipsoid'])
# plt.scatter(df.index, df['depth_tide_ellipsoid'])
plt.plot(df['date'], df['depth_tide_ellipsoid'])
# plt.scatter(df['date'], df['depth_tide_ellipsoid'])
plt.xlabel('Date')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Value')
plt.title('Time Series Plot')
plt.grid(False)
plt.tight_layout() #helps, but not enough
plt.subplots_adjust(bottom=0.2,left=.15)
plt.show()

