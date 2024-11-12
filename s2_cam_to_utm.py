#!/usr/bin/env python
 
import os
import os.path as osp
import glob 
import multiprocessing

import numpy as np
import datetime as dt

from imports.stop_watch import stop_watch 
import imports.tracking_misc as trm
import imports.camtools as ct

#%%

''' 
script to project tracks from camera coordinates to map coordinates

VARIABLES:
   
workspace:
    path to workspace with .npz input files
    input .npz files will be searched under workspace/camname/oblique/%Y%m%d/
    e.g., '/raid/image_processing/UAS7/20180902/'
    
camnames: 
    camera(s) to be processed
    single name or list of names, e.g. 'UAS7' or ['UAS7', 'UAS8'] 
    camera must match a folder name in the 'workspace' directory
    it also has to match the camera name in the parameter excel file 'paramfile_name' 
    
target_folder:
    name of target folder, to be created under workspace/camname/
    will contain the projected velocity fields in hourly .npz files
      
paramfile_name:
    excel file featuring processing parameters, e.g., from camera calibration
    must be placed under ./ (in the same directory as this script)
    e.g., 'parameter_file_2018.xlsx' 
    
tide_file_name:
    name of .pickle file containing the tide time-series
    file must be placed under ./data/ 
    e.g., 'tide_prediction_2018.pickle' 
    
min_date: 
    start date of period of interest, e.g. 20180904 
max_date: 
    end date of period of interest, e.g. 20180905
    
max_speed:
    maximum iceberg speed allowed, in m/s, e.g., 1.7
min_speed:
    minimum iceberg speed allowed, in m/s, e.g., 0.03
max_speedfactor:
    maximum speed ratio allowed between consecutive vectors
    e.g., 2.5 
max_angle:
    maximum direction change allowed between consecutive vectors 
    in degrees, e.g., 60  
speed_threshold:
    speed threshold, in m/s, e.g., 0.1
    above which to apply the max_speedfactor and max_angle criteria

n_proc: 
    number of processors used during processing, e.g. 8  
''' 
    
def main():  
    
    # parameters
    workspace = '/hdd3/opensource/iceberg_tracking/output/'
    camnames = ['cam4'] 
    target_folder = 'utm'
        
    paramfile_name = 'parameter_file_2019.xlsx'
    tide_file_name = 'tide_2019.pickle'
    
    min_date = 20190724
    max_date = 20190726
    
    max_speed = 1.7
    min_speed = 0.0
    max_speedfactor = 2.5
    max_angle = 60
    speed_threshold = 0.1 
    
    n_proc = 8
    
    #--------------------------------------------------------------------------
    
    # create the stop watch object
    sw = stop_watch()
    sw.start()
    
    # check whether camnames is a list, and create list if not
    if type(camnames) != list:
        camnames = [camnames] 
    
    # loop over camera names, locate daily folders, and launch script for 
    # subfolders in parallel
    # paralellization is set up so that cams are processed consecutively    
    for camname in camnames:
        
        n_proc_orig = n_proc 
                   
        print('processing: ' + camname)
        
        # list the daily folders
        source_workspaces = sorted(glob.glob(osp.join(workspace, camname, 'oblique', '20??????')))
        
        # keep only folders within the defined time range
        for swsp in source_workspaces[:]:
            
            if int(osp.basename(swsp)) < min_date or int(osp.basename(swsp)) > max_date:
                
                source_workspaces.remove(swsp)
                    
        if len(source_workspaces) < n_proc_orig:
            n_cpus_checked = len(source_workspaces)
        else:
            n_cpus_checked = n_proc_orig 
            
        target_workspace = osp.join(workspace, camname, target_folder)
            
        if osp.isdir(target_workspace) == 0:
            os.mkdir(target_workspace)
            
        # create list of tasks to be executed
        arguments = [(source_workspace, target_workspace, camname, max_speed, min_speed, 
                      max_speedfactor, max_angle, speed_threshold, paramfile_name, tide_file_name) 
                      for source_workspace in source_workspaces]
        
        if n_proc > 1:
            pool = multiprocessing.Pool(processes = n_cpus_checked)
            pool.map(cam_to_utm, arguments)
            pool.close()
        else:
            for argument in arguments:
                cam_to_utm(argument)
                
        sw.print_intermediate_time()
                
    sw.print_elapsed_time()
    sw.stop()
    
#%%

def cam_to_utm(arguments):
    
    (source_workspace, target_workspace, camname, max_speed, min_speed, 
     max_speedfactor, max_angle, speed_threshold, paramfile_name, tide_file) =  arguments
    
    # determine directory of current script (does not work in interactive mode)
    file_path = osp.dirname(osp.realpath(__file__))
    
    # concatenate path to camera calibration file      
    paramfile_path = osp.join(file_path, paramfile_name) 
    
    # obtain daytring from foldername
    datestring = osp.basename(source_workspace)
    
    # list npz files in the folder   
    npzs = sorted(glob.glob(source_workspace + '/*.npz'))
    
    if len(npzs) > 0:
    
        # derive the image spacing (in seconds) from the first .npz file
        tracking_interval = int(osp.basename(npzs[0]).split('_')[-2].split('sec')[0])
            
        # empty lists to store one hour's worth of tracks           
        xlist = []
        ylist = []
        ulist = []
        vlist = []
        speedlist = []
        timelist = []
        
        # empty lists for tracks of the following hour
        xlist_n = []
        ylist_n = []
        ulist_n = []
        vlist_n = []
        speedlist_n = []
        timelist_n = []
        
        for c, npz in enumerate(npzs):
                                  
            npz_time = osp.basename(npz).split('_')[0]
            npz_time_dt = dt.datetime.strptime(npz_time, '%Y%m%d-%H%M%S')
            
            current_hour = npz_time_dt.hour
            
            if c == 0:
                next_hour = (npz_time_dt + dt.timedelta(hours = 1)).hour 
            
            # save to .npz file once a new hour starts
            # hourly instead of daily .npz files circumvent large .npz files
            if current_hour == next_hour:
                
                npz_time_dt_label = npz_time_dt - dt.timedelta(hours = 1) 
                                
                # save projected velocities to .npz file (one file per hour)    
                np.savez(osp.join(target_workspace, '{}_{}00_{}s_utm.npz'.format(npz_time_dt_label.strftime('%Y%m%d'), 
                                  npz_time_dt_label.strftime('%H'), tracking_interval)), x = xlist, y = ylist, 
                                  u = ulist, v = vlist, speed = speedlist, time = timelist)
                
                xlist = xlist_n
                ylist = ylist_n
                ulist = ulist_n
                vlist = vlist_n
                speedlist = speedlist_n
                timelist = timelist_n
                
                xlist_n = []
                ylist_n = []
                ulist_n = []
                vlist_n = []
                speedlist_n = []
                timelist_n = []
                
                next_hour = (npz_time_dt + dt.timedelta(hours = 1)).hour
                
            # load the npz file and convert from array to list
            tracks = np.load(npz)
            tracks = tracks['tracks'].tolist()
            
            # create camera object
            cam = ct.Camera(camname = camname, date = datestring, paramfile_path = paramfile_path, 
                            tide_corr = 1, tide_file = tide_file, datetime = npz_time)
            
            # 1) loop through each track (contains x and y in image coordinates), 
            # 2) convert to utm coordinates, 3) calculate u and v in m, 
            # 4) remove implausible vectors 
            for track in tracks:
            
                track_utm = []    
                
                for (x, y) in track:
                    
                    # move coordinates to where they would be without cropping
                    [photocordx, photocordy] = cam.photocords_cropped_to_uncropped(x, y)
                    
                    vertex_utm = cam.photo_to_utm(photocordx, photocordy)
                    
                    track_utm.append(vertex_utm)
                
                # lists for each track
                #                 
                usublist = []
                vsublist = []
                speedsublist = []
                
                # list for tracks within current hour
                xsublist_p = []
                ysublist_p = []
                usublist_p = []
                vsublist_p = []
                speedsublist_p = []
                timesublist_p = []
                
                # list for tracks within next hour
                xsublist_n = []
                ysublist_n = []
                usublist_n = []
                vsublist_n = []
                speedsublist_n = []
                timesublist_n = []
                 
                # loop over vertices, calculate speeds in m/s, and allocate times 
                for i in range(1, len(track_utm)):
                    
                    time = npz_time_dt + dt.timedelta(seconds = (i - 1) * tracking_interval)
                    
                    # divide speed by tracking interval (in seconds) to obtain m/s 
                    xmovement = (track_utm[i][0] - track_utm[i - 1][0]) / float(tracking_interval) 
                    ymovement = (track_utm[i][1] - track_utm[i - 1][1]) / float(tracking_interval) 
                    speed = np.hypot(xmovement, ymovement)
                    
                    usublist.append(xmovement)   
                    vsublist.append(ymovement)
                    speedsublist.append(speed)
                    
                    if time.hour == current_hour:
                        
                        xsublist_p.append(track_utm[i-1][0])
                        ysublist_p.append(track_utm[i-1][1])
                        usublist_p.append(xmovement)
                        vsublist_p.append(ymovement)
                        speedsublist_p.append(speed)
                        timesublist_p.append(trm.datetime_to_epoch(time))
                        
                    else:
                        xsublist_n.append(track_utm[i-1][0])
                        ysublist_n.append(track_utm[i-1][1])
                        usublist_n.append(xmovement)
                        vsublist_n.append(ymovement)
                        speedsublist_n.append(speed)
                        timesublist_n.append(trm.datetime_to_epoch(time))
                        
                # criterion 1 - remove implausibly low or high speeds                
                if (np.mean(speedsublist) < min_speed) or (max(speedsublist) > max_speed): 
                    continue
                
                # check only if minimal speed_threshold is exceeded
                if max(speedsublist) > speed_threshold: 
                
                    # loop over consecutive vectors and determine angle and 
                    # speed ratios between them
                    anglediffs = []
                    speedratios = []
                    
                    for c1, c2 in zip(range(0, len(usublist) - 1), range(1, len(usublist))): 
                        
                        dotprod = usublist[c1] * usublist[c2] + vsublist[c1] * vsublist[c2] 
                        mag1 = np.hypot(usublist[c1], vsublist[c1])
                        mag2 = np.hypot(usublist[c2], vsublist[c2])
                        
                        anglediff = np.degrees(np.arccos(dotprod / (mag1 * mag2)))
                        
                        anglediffs.append(abs(anglediff))
                        
                        speedratios.append(max([speedsublist[c1], speedsublist[c2]]) / 
                                           min([speedsublist[c1], speedsublist[c2]]))
                                           
                                    
                    # criterion 2 - remove if consecutive vectors have speed ratios
                    # above the threshold
                    if max(speedratios) > max_speedfactor:
                        continue
                    
                    # criterion 3 - remove if consecutive vectors have direction 
                    # changes above the threshold
                    if max(anglediffs) > max_angle:
                        continue
                
                # extend lists only if all criteria are met
                xlist.extend(xsublist_p)
                ylist.extend(ysublist_p)
                ulist.extend(usublist_p)
                vlist.extend(vsublist_p)
                speedlist.extend(speedsublist_p)
                timelist.extend(timesublist_p)
                
                xlist_n.extend(xsublist_n)
                ylist_n.extend(ysublist_n)
                ulist_n.extend(usublist_n)
                vlist_n.extend(vsublist_n)
                speedlist_n.extend(speedsublist_n)
                timelist_n.extend(timesublist_n)
                    
        # save projected velocities to .npz file (one file per hour)    
        np.savez(osp.join(target_workspace, '{}_{}00_{}s_utm.npz'.format(npz_time_dt.strftime('%Y%m%d'), 
                          npz_time_dt.strftime('%H'), tracking_interval)), x = xlist, y = ylist, 
                          u = ulist, v = vlist, speed = speedlist, time = timelist)
        
        print('folder {} done: {} files'.format(datestring, len(npzs)))
        
    else:
        print('folder {}: no files'.format(datestring))    
        
#%%
                       
if __name__ == '__main__':
    
    main()