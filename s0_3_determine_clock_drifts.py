#!/usr/bin/env python

import os.path as osp

import pandas as pd
import datetime as dt
                              
def prepare_correction_cam_drift(workspace):
    
    '''reads an excel file that contains camera clock time in mm:ss and corresponding GPS clock time 
    calculates time drifts at the beginning and end of the deployment  
    also calculates time drift change in seconds per day'''
        
    df = pd.read_excel(osp.join(workspace, 'data/camera_time_drifts_input.xlsx'))
    
    # add new empty columns
    df = df.reindex(columns = df.columns.tolist() + ['day_diff', 'drift_start_sec', 
    'drift_end_sec', 'drift_pday_sec'])

    # loop over rows and conduct calculations
    for index, row in df.iterrows():
        
        start_date = dt.datetime.strptime(str(row['start_date']), '%Y%m%d').date() 
        end_date = dt.datetime.strptime(str(row['end_date']), '%Y%m%d').date()  
        
        # day difference between start and end of deployment
        day_diff = (end_date - start_date).days 
        
        # convert mm:ss to [mm, ss]
        start_time_cam = row['start_time_cam_mmss'].split(':')
        start_time_gps = row['start_time_gps_mmss'].split(':')
        end_time_cam = row['end_time_cam_mmss'].split(':')
        end_time_gps = row['end_time_gps_mmss'].split(':')
        
        # convert [mm, ss] to seconds
        start_time_cam = int(start_time_cam[0]) * 60.0 + int(start_time_cam[1])
        start_time_gps = int(start_time_gps[0]) * 60.0 + int(start_time_gps[1])
        end_time_cam = int(end_time_cam[0]) * 60.0 + int(end_time_cam[1])
        end_time_gps = int(end_time_gps[0]) * 60.0 + int(end_time_gps[1])
        
        # determine drifts at beginning and end of deployment
        drift_start_sec = start_time_gps - start_time_cam
        drift_end_sec = end_time_gps - end_time_cam
        
        # determine drift change from start to end of campaign
        drift_change = drift_end_sec - drift_start_sec   
        
        df.set_value(index, 'day_diff', day_diff)
        df.set_value(index, 'drift_start_sec', drift_start_sec)
        df.set_value(index, 'drift_end_sec', drift_end_sec)
        df.set_value(index, 'drift_pday_sec', float(drift_change) / day_diff)
          
    df.to_excel(osp.join(workspace, 'data/camera_time_drifts.xlsx'), 
                   index = 0)
                   
if __name__ == '__main__':
    
    # workspace = '.../.../iceberg_tracking'
    
    # determine directory of current script (does not work in interactive mode)
    workspace = osp.dirname(osp.realpath(__file__))
    
    prepare_correction_cam_drift(workspace)