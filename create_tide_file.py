import requests
import pickle
import pandas as pd

''' Downloads tide data from Noaa at Tar inlet, you can change station number to retrieve data from other places'''

# specify the pickle file path
pickle_file_path = '/hdd3/opensource/iceberg_tracking/data/'
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
    "station": "9452749",  # replace with your station number
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
df.to_pickle(pickle_file_path + output_filename)

