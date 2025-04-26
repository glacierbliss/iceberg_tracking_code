# config_EXAMPLE.py User should edit paths and rename this file to config.py.
#config.py file lives in the code folder (/iceberg_tracking_code/) but should be ignored by Git.
#config_EXAMPLE.py shows how it should be set up.
#The code should copy the config file to the output directory for safe keeping.
import os

# Define base directory if needed (e.g., the directory where your scripts are)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths for your files or directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Add more paths as needed

#Copy config file to output directory for safe-keeping.
#...
