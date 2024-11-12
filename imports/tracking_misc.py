#!/usr/bin/env python

import glob
import os.path as osp

import math 
import numpy as np
import datetime as dt
import pandas as pd

import matplotlib.path as mplPath 
  
#%%
      
def create_squares(origin, width, height):
    
    x = origin[0]
    y = origin[1]
    
    poly = [(x, y), (x + width, y), (x + width, y - width), (x, y - width)] #(x, y)
    point = [x + 0.5 * width, y - 0.5 * height]
    
    return [poly, point]
    
def create_grid_across_fjord(fjord, spacing):
    
    polygons = []
    centerpoints = []
    indices = []
    
    topleft = [min(fjord['x']), max(fjord['y'])] 
    
    topleft_px_center = [min(fjord['x']) + 0.5 * spacing, max(fjord['y']) - 0.5 * spacing]  
    
    # points_fjord = np.vstack((fjord['x'][fjord['id']<2], fjord['y'][fjord['id']<2])).T
    points_fjord = np.vstack((fjord['x'], fjord['y'])).T

    fjord_poly = mplPath.Path(points_fjord)
    
    cols = int(math.ceil((max(fjord['x']) - min(fjord['x'])) / spacing)) 
    rows = int(math.ceil((max(fjord['y']) - min(fjord['y'])) / spacing)) 

    for i in range(0, cols):
        
        for j in range(0, rows):
            
            origin = [topleft[0] + i * spacing, topleft[1] - j * spacing]
            
            [poly, point] = create_squares(origin, spacing, spacing) 
            
            # check whether center point is within the fjord
            if fjord_poly.contains_point(point) == 1:
                
                polygons.append(poly)
                centerpoints.append(point)
                indices.append([i, j])
                           
    return [polygons, centerpoints, indices, topleft_px_center, rows, cols]
    

def scale_arrows(u, v, exponent = 0.5, factor = 250):
    
    """returns scaled u and v vectors, used for plotting"""
    
    angles = np.arctan2(v, u) 
    
    speed = np.hypot(u, v)
    
    speed_scaled = (speed ** 0.5) * factor
    
    u_scaled = np.cos(angles) * speed_scaled
    v_scaled = np.sin(angles) * speed_scaled
    
    return [u_scaled, v_scaled] 
          
def azimuth_transect(p1, p2):
    
    u = p2[0] - p1[0]
    v = p2[1] - p1[1]
    
    azimuth = np.arctan2(v, u)
    
    return azimuth
    
def locate_points_along_transect(p1, p2, dist):
    
    u = p2[0] - p1[0]
    v = p2[1] - p1[1]
    
    profile_len = np.hypot(u, v)
    
    azimuth = np.arctan2(v, u)
    
    u_scaled = np.cos(azimuth) * dist
    v_scaled = np.sin(azimuth) * dist
       
    pointlist = []
    distancelist = []
        
    counter = np.arange(0, np.ceil((profile_len + 0.3 * dist) / dist))
    
    for c in counter:
        
        pointlist.append([p1[0] + c * u_scaled, p1[1] + c * v_scaled])
        distancelist.append(c * dist)
        
    return [pointlist, distancelist] 

def create_squares_rot(center, height, width, rotation):
    
    x = center[0]
    y = center[1]
    
    c = np.cos(rotation)
    s = np.sin(rotation)
   
    R = np.array([[c, s], [-s, c]])
        
    [u1, v1] = np.dot(np.array([0.5 * height, 0.5 * width]), R)
    [u2, v2] = np.dot(np.array([0.5 * height, -0.5 * width]), R)
    [u3, v3] = np.dot(np.array([-0.5 * height, -0.5 * width]), R)
    [u4, v4] = np.dot(np.array([-0.5 * height, 0.5 * width]), R)
    
    p1 = (x + u1, y + v1)
    p2 = (x + u2, y + v2)
    p3 = (x + u3, y + v3)
    p4 = (x + u4, y + v4)
    
    poly = [p1, p2, p3, p4]
    
    return poly 
  
def create_squares_along_transect(profile, pointspacing, width):
    
    p1 = np.array((profile['x'][0], profile['y'][0]))
    p2 = np.array((profile['x'][-1], profile['y'][-1]))
    
    [points, distances] = locate_points_along_transect(p1, p2, pointspacing)   
    azimuth = azimuth_transect(p1, p2)
    
    polylist = []
       
    for point in points:
        
        poly = create_squares_rot([point[0], point[1]], pointspacing, width, azimuth)
        
        polylist.append(poly)
        
    return [polylist, points, distances]
    
def create_squares_around_mooring(coord_mooring, azimuth = -45, width = 100, nr = 7):
    
    polygons = []
    centerpoints = []
    distancelist = []
    
    nr_side = np.floor(nr / 2.0)
           
    distances1 = np.arange(-1 * nr_side * width, nr_side * width + 1, width)
    distances2 = np.arange(-1 * nr_side * width, nr_side * width + 1, width)
      
    azimuth = np.radians(azimuth)
        
    for dist1 in distances1:
        
        for dist2 in distances2:
    
            u_scaled_1 = np.cos(azimuth) * dist1
            v_scaled_1 = np.sin(azimuth) * dist1
            
            u_scaled_2 = np.cos(azimuth + np.radians(90)) * dist2
            v_scaled_2 = np.sin(azimuth + np.radians(90)) * dist2
            
            point = [coord_mooring[0] + u_scaled_1 + u_scaled_2, 
                              coord_mooring[1] + v_scaled_1 + v_scaled_2] 
                        
            centerpoints.append(point)
                              
            poly = create_squares_rot([point[0], point[1]], width, width, azimuth)
            
            polygons.append(poly)
        
            distancelist.append([dist1, dist2])
                                                          
    return [polygons, centerpoints, distancelist]   
        
def vector_magn(v):
    '''returns vector magnitude'''
    magnitude = np.hypot(v[0], v[1])
    return magnitude

def vector_angle(v1, v2):
    '''returns the angle between two vectors in degrees'''
    ang = (np.arccos(np.dot(v1, v2) / (vector_magn(v1) * vector_magn(v2)))) * 180 / np.pi
    return ang
  
def vector_projection(v1, v2):
    '''returns projection of vector v1 onto v2'''
    v1_proj  = ((np.dot(v1, v2)) / (vector_magn(v2) ** 2)) * v2
    return v1_proj
    
def calc_velocity_across_transect(flow_vec, transect_vec):
    
    flow_vec_mag = vector_magn(flow_vec)
    
    # add small number to avoid nans
    flow_vec[0] = flow_vec[0] + 0.000001
    
    # in case of 180 degrees (vectors have opposite directions, hence *-1)   
    if round(vector_angle(flow_vec, transect_vec), 0) == 180:
        flow_vec_mag = flow_vec_mag * -1
        
    elif round(vector_angle(flow_vec, transect_vec), 0) == 0: 
        flow_vec_mag = flow_vec_mag
        
    else:
        flow_vec_mag = np.nan 
        
    return flow_vec_mag

#%%

def round_time(time = None, round_to = 60):
   """round a datetime object to any time laps in seconds
   time: datetime.datetime object, default now
   round_to: closest number of seconds to round to, default 1 minute
   """
   if time == None: 
       time = dt.datetime.now()
   
   seconds = (time.replace(tzinfo=None) - time.min).seconds
   
   rounding = (seconds + round_to / 2) // round_to * round_to
   
   return time + dt.timedelta(0, rounding-seconds, -time.microsecond)
    
def datetime_to_epoch(dt_stamp):
    
    return int((dt_stamp - dt.datetime(1970, 1, 1)).total_seconds())
    
def epoch_to_datetime(epoch):
    
    return dt.timedelta(seconds = epoch) + dt.datetime(1970, 1, 1)

def return_velocities_by_time(workspace, start_time, end_time):
    
    '''function selects velocities within time window'''
    
    start_time_epoch = datetime_to_epoch(start_time) 
    end_time_epoch = datetime_to_epoch(end_time)
    
    start_time_trunc = start_time.replace(minute = 0, second = 0) 
    end_time_trunc = end_time.replace(minute = 0, second = 0)
    
    hours = pd.date_range(start_time_trunc, end_time_trunc, freq = 'H').tolist()
    
    x_sel = np.array([])
    y_sel = np.array([])
    u_sel = np.array([])
    v_sel = np.array([])
    speed_sel = np.array([])
    time_sel = np.array([]) 
    
    # loop over hourly periods and load corresponding .npz files 
    for hour in hours:
        
        try:

            npzfile = glob.glob(osp.join(workspace, hour.strftime('%Y%m%d_%H00') + '*.npz'))[0]
            
            npz = np.load(npzfile)
            
            x = npz['x']
            y = npz['y']
            u = npz['u']
            v = npz['v']
            speed = npz['speed']
            time = npz['time']
            
            mask = (time >= start_time_epoch) & (time < end_time_epoch)
            
            x_sel = np.concatenate([x_sel, x[mask]])
            y_sel = np.concatenate([y_sel, y[mask]])
            u_sel = np.concatenate([u_sel, u[mask]])
            v_sel = np.concatenate([v_sel, v[mask]])
            speed_sel = np.concatenate([speed_sel, speed[mask]])
            time_sel = np.concatenate([time_sel, time[mask]]) 
        
        # pass in case there is no .npz file
        except:
            pass
            
    return [x_sel, y_sel, u_sel, v_sel, speed_sel, time_sel]
   
def return_closest_image(workspace, target_time, max_timediff = 300):
    
    # identify folder of pics
    # subtract 8 hours to get from UTC to AK time
    dt_local = target_time - dt.timedelta(hours = 8)
    foldername = dt_local.strftime('%Y%m%d')
    
    # list images and create time list
    imagenames = glob.glob(osp.join(workspace, foldername, '*.jpg'))
    image_times = [dt.datetime.strptime(osp.basename(name), '%Y%m%d-%H%M%S.jpg') for name in imagenames]
    
    # find the value closest in time
    closest_time = min(image_times, key = lambda x: abs(x - target_time))
    
    closest_image = osp.join(workspace, foldername, closest_time.strftime('%Y%m%d-%H%M%S.jpg'))
    
    # return image only if within a certain time difference of target_time
    if abs((target_time - closest_time).total_seconds()) < max_timediff:  
        return closest_image    
    else:
        return -99 
       
def nearest_date(target_time, df, time_window):
    
    '''returns the time closest to the target_time, time_window is in minutes'''
           
    # preselect values close to the searched value 
    items = df[(df['date'] > target_time - dt.timedelta(minutes = time_window)) & 
            (df['date'] < target_time + dt.timedelta(minutes = time_window))]
    
    # find the value closest in time
    closest = min(items['date'], key = lambda x: abs(x - target_time))

    selection = df[(df['date'] == closest)]

    return [selection['date'], list(selection['npz'])] 
                     
def correct_time_drift(camnr, date, time_drift_file): 
    
    '''returns a time correction in seconds that needs to be added to the photo timestamps to arrive at corrected UTC times'''
    
    # select correct row in the file
    time_drifts_sel = time_drift_file[(time_drift_file['cam'] == camnr) & 
                     (time_drift_file['start_date'] < int(date)) & 
                     (time_drift_file['end_date'] >= int(date))] 
    
    # convert to datetimes and determine day difference
    date_dt = dt.datetime.strptime(date, '%Y%m%d')
    date_dt_start = dt.datetime.strptime(str(time_drifts_sel['start_date'].iloc[0]), '%Y%m%d')
    difference = date_dt - date_dt_start
    
    # correction is composed of clock drift at the beginning plus the drift accumulated over time    
    correction_seconds = time_drifts_sel['drift_start_sec'].iloc[0] + difference.days * time_drifts_sel['drift_pday_sec'].iloc[0]
    
    return round(correction_seconds, 1) 