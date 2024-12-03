#AKB started to convert... need internet. need to add pandas to conda env
import requests
import pickle
import pandas as pd
from pathlib import Path

''' Downloads tide data from NOAA at Tarr Inlet, you can change station number to retrieve data from other places'''

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
df.to_pickle(pickle_file_path/output_filename)

#AKB added
# plot the tide levels
df.set_index('date', inplace=True) # Set 'date' as the index
df.plot(df.index,df['depth_tide_ellipsoid'])

df.plot(x=df['date'],y=df['depth_tide_ellipsoid'])
plt.show()

df.plot()
plt.show()
