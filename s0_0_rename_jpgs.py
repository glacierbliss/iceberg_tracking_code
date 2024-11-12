#!/usr/bin/env python
 
import os
import os.path as osp
import glob 

import datetime as dt
from PIL import Image

def get_date_taken(path):
    return Image.open(path)._getexif()[36867]
    
def rename_jpgs():
    
    '''creates time-stamped .jpgs ('%Y%m%d-%H%M%S.jpg') and moves them into daily folders. 
    in our case, camera clocks were set to UTC to match other deployed instruments. 
    because folder structure needs to match local time, a variable 'time_difference' 
    is added to account for time difference (e.g., -9 to get from UTC to Alaska
    Standard Time)''' 
     
    workspace = '/hdd3/johns_hopkins/mcbride/DCIM_rename/100CANON' # workspace containing the original photos
    time_difference = -9 # time difference to get from UTC (image timing) to local time
    file_extension = '.JPG' # file extension used to collect photos. case sensitive on linux   
    
    # collects .jpgs on the workspace level and in subfolders
    # such as 101CANON, 102CANON
    jpegs = glob.glob('{}/*{}'.format(workspace, file_extension)) + glob.glob('{}/*/*{}'.format(workspace, file_extension))
    
    for jpg in jpegs:
        
        # extract the timestring from the .exif data
        date_taken = get_date_taken(jpg) 
        
        # convert into datetime and create name of new .jpg
        datetime_utc = dt.datetime.strptime(date_taken, '%Y:%m:%d %H:%M:%S')
        newjpg = datetime_utc.strftime('%Y%m%d-%H%M%S.jpg')
        
        # determine time of image in alaska standard time
        datetime_akst = datetime_utc + dt.timedelta(hours = time_difference)
        folder = datetime_akst.strftime('%Y%m%d')
        
        # create dir if it does not yet exist
        try:
            os.mkdir(osp.join(workspace, folder))
            print(folder + ' created')
        except:
            pass
        
        # rename .jpg and move to the new folder
        os.rename(jpg, osp.join(workspace, folder, newjpg))
    
    # list old folders    
    folders = os.listdir(workspace)
    
    # loop over old folders and remove the empty ones
    # rmdir throws an error for non-empty folders, 
    # hence the try/except structure
    for folder in folders:
        try:
            os.rmdir(osp.join(workspace, folder))
            print(folder + ' removed')
        except:
            pass  
       
if __name__ == '__main__':
    
    rename_jpgs()