#!/usr/bin/env python

import os
import os.path as osp
import multiprocessing
import subprocess
import glob
import warnings

import pandas as pd
import datetime as dt
import numpy as np

from PIL import Image
from PIL import ImageFile

import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.path as mplPath 
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import cmocean

from imports.stop_watch import stop_watch 
import imports.utilities as util
import imports.tracking_misc as trm

try:
    warnings.filterwarnings('ignore')
    mpl.rc("font", **{"sans-serif": ["Arial"]})
except:
    pass

''' 
script to combine results from multiple cameras, to average velocity fields 
spatially and temporally, and to plot the results

VARIABLES:
   
source_path_head:
    path head portion to the directory with the .npz input files (projected velocity fields)
    files will be searched under source_path_head/camname/source_path_tail
    
camnames: 
    camera(s) to be processed
    single name or list of names, e.g. 'UAS7' or ['UAS7', 'UAS8'] 
    camera name must match a folder name in the 'source_path_head' directory
    it also has to match the camera name in the parameter excel file 'paramfile'
    files will be searched under source_path_head/camname/source_path_tail
    
source_path_tail:
    path tail portion to the directory with the .npz input files (projected velocity fields)
    files will be searched under source_path_head/camname/source_path_tail
    
source_path_photos:
    path to directory with original .jpg photos (used for plotting)
    e.g. '/hdd2/leconte_2018/cameras'
    photos will be searched in daily folders
    source_path_photos/camname/%Y%m%d
    
target_path:
    path to directory that will contain the integrated velocity fields (.npz files)
    and plots
      
paramfile:
    excel file featuring processing parameters, e.g., from camera calibration
    must be placed under ./ (in the same directory as this script)
    e.g., 'parameter_file_2018.xlsx'
    
clockdriftfile:
    excel file featuring clockdrift related info for each camera
    must be placed under ./data/

fjord_outline:
    .npz file with fjord outline coordinates 
    required to create grid across fjord and for plotting
    must be placed under ./data/
    must contain the three arrays 'x, 'y', and 'id', 'id' must be set to 1
    
min_date: 
    start date of period of interest, e.g. 20180904 
max_date: 
    end date of period of interest, e.g. 20180905
    
time_window:
    time window (in hours) for temporal aggregation of trajectories, e.g. 1.0
    24 aggregates over full day
grid_size:
    grid size (in m) used for spatial aggregation of trajectories, e.g. 50
observation_threshold:
    minimum number of trajectories per grid cell, e.g. 10
    
plot_switch:
    switch [0, 1, 2] to control plotting
    0 = create no plots (i.e., create .npz files only)
    1 = create plot with one map showing aggregated velocities 
    2 = create plot with two maps, showing original and aggregated velocities        
        (avoid '2' for time windows > 30 minutes as plotting of original velocities 
        may use too much memory) 

speedthreshold_cbar:
    threshold (m/s) limiting upper speed limit of colorbar
    
movie_switch:
    switch [1 or 0] to control whether plots are animated into a movie

n_proc: 
    number of processors used during processing, e.g. 8 

NOTE: 
    adapt plotting variables in the two plotting functions directly    
    
''' 
  
def main():
    
    # parameters
    source_path_head = '/hdd3/opensource/iceberg_tracking/output/'
    camnames = ['cam1','cam2','cam3','cam4'] 
    source_path_tail = 'utm'
    source_path_photos = '/hdd3/opensource/iceberg_tracking/data/'
    target_path = '/hdd3/opensource/iceberg_tracking/output/run1/' 
    
    paramfile = 'parameter_file_2019.xlsx'
    clockdriftfile = 'camera_time_drifts.xlsx' 
    fjord_outline = 'fjord_outline.npz'
    
    min_date = 20190724
    max_date = 20190726
    
    time_window = 30 / 60.0  
    grid_size = 200
    observation_threshold = 10
    
    plot_switch = 2
    speedthreshold_cbar = 0.5
    movie_switch = 1
    
    n_proc = 10
    

    #Changes to make:
    #1. Add an option to add in a region of interest instead of grabbing coordinates from fjord outline

    #--------------------------------------------------------------------------
    
    # create the stop watch object
    sw = stop_watch()
    sw.start()
           
    if osp.isdir(target_path) == 0:
        os.makedirs(target_path)
    
    dayrange = pd.date_range(dt.datetime.strptime(str(min_date), '%Y%m%d'), 
                             dt.datetime.strptime(str(max_date), '%Y%m%d'))
    
    if len(dayrange) < n_proc:
        n_proc = len(dayrange)
        
    # check whether camnames is a list, and create list if not
    if type(camnames) != list:
        camnames = [camnames] 
    
    arguments = [(camnames, source_path_head, source_path_tail, target_path, 
                  source_path_photos, paramfile, clockdriftfile, fjord_outline, day, 
                  time_window, grid_size, speedthreshold_cbar, observation_threshold, 
                  plot_switch) for day in dayrange]
    
    if n_proc > 1:
        pool = multiprocessing.Pool(processes = n_proc)
        pool.map(utm_to_gridded_utm, arguments)
        pool.close()
    else:
        for argument in arguments:
            utm_to_gridded_utm(argument)
            
        sw.print_intermediate_time()
        
    if plot_switch != 0 and movie_switch == 1:
        
        print('create movie...')
        
        # determine directory of current script (does not work in interactive mode)
        file_path = osp.dirname(osp.realpath(__file__))
        print(file_path)             
        shell_script = osp.join(file_path, 'imports', 'timelapse.sh') 
        dim1 = 2000
        dim2 = -10
        moviename = 'velocities_utm_{}min.avi'.format(int(time_window * 60.0))
        searchpattern = '*.png'
        
        subprocess.call([shell_script, target_path + '/', str(int(dim1)), str(int(dim2)), 
                         moviename, '8', searchpattern])  
        
    sw.print_elapsed_time()
    sw.stop()
    
#%%
        
def utm_to_gridded_utm(arguments):
    
    (camnames, source_path_head, source_path_tail, target_path, source_path_photos, 
     paramfile_name, clockdriftfile_name, fjord_outline, day, time_window, grid_size, 
     speedthreshold_cbar, observation_threshold, plot_switch) = arguments
         
    day_str = day.strftime('%Y%m%d')
    
    print('working on ' + day_str)
    
    # determine directory of current script (does not work in interactive mode)
    file_path = osp.dirname(osp.realpath(__file__))
    
    # concatenate path to parameter file and read it     
    paramfile_path = osp.join(file_path, paramfile_name)
    paramfile = pd.read_excel(paramfile_path)
    
    startlist = []
    endlist = []
    camnames_filtered = [] 

    # determine at what hour cameras started and stopped taking photos 
    # information is taken from the parameter file   
    for camname in camnames:
        
        # obtain the parameters for specified day 
        parameters = paramfile.loc[(paramfile['camera'] == camname) & 
        (paramfile['start_day'] <= int(day_str)) & 
        (paramfile['end_day'] >= int(day_str))]
        
        if len(parameters.index) == 1: 
        
            timestring = parameters['start_time'].iloc[0]
            timeobj = dt.datetime.strptime(timestring, '%H:%M').time()
            
            start = timeobj.hour + timeobj.minute / 60.0 
            end = start + parameters['tracking_duration'].iloc[0]
            
            startlist.append(start)
            endlist.append(end)
            camnames_filtered.append(camname)
        
    
    # continue only if at least one camera took photos                    
    if len(camnames_filtered) > 0:
        
        # divide time period between start and stop time into subperiod of lenght time_window
        start_hours = list(np.arange(min(startlist), max(endlist) + 0.001, time_window))[0:-1]
        end_hours = list(np.arange(min(startlist), max(endlist) + 0.001, time_window))[1:]    
      
        if time_window == 24.0:
            start_hours = [min(startlist)]
            end_hours = [max(endlist)]
        
        # read file containing time drifts
        time_drift_file = pd.read_excel(osp.join(file_path, 'data', clockdriftfile_name))
        
        # load fjord coordinates
        fjord = np.load(osp.join(file_path, 'data', fjord_outline))
        
        # loop over consecutive time periods, correct cameras for time drifts and 
        # extract velocities within time window for each camera    
        for start_hour, end_hour in zip(start_hours, end_hours):
            
            # lists to keep track which cameras have velocities
            cam_with_tracks = []
    
            start_datetime = day + dt.timedelta(hours = start_hour)
            end_datetime = day + dt.timedelta(hours = end_hour) 
            
            # calculate length of time window in minutes
            time_diff = end_datetime - start_datetime
            time_diff = int(time_diff.total_seconds() / 60.0)
            
            # switch to determine whether camera is the first one with data
            c_switch = 0
            
            mintimelist = []
            maxtimelist = []
            
            for camname in camnames_filtered:
                
                try:
                    time_correction = trm.correct_time_drift(camname, day_str, time_drift_file)
                except:
                    print(camname + ': no time drift correction available')
                    time_correction = 0
                            
                # convert utc times to camera times (reverse engineer time shift, thus minus) 
                start_datetime_corr = start_datetime - dt.timedelta(seconds = time_correction)
                end_datetime_corr = end_datetime - dt.timedelta(seconds = time_correction) 
                
                # concatenate workspace and list .npz files
                workspace = osp.join(source_path_head, camname, source_path_tail)  
                npzs = sorted(glob.glob(osp.join(workspace, day_str + '*utm.npz')))
                
                # continue if there are no velocities from selected camera for selected day
                if len(npzs) == 0:
                    continue
                
                # call function that returns velocities within selected time window
                [x_sel, y_sel, u_sel, v_sel, speed_sel, time_sel] = trm.return_velocities_by_time(workspace,
                 start_datetime_corr, end_datetime_corr)
                
                if len(u_sel) > 0:
                    
                    cam_with_tracks.append(camname) 
                    
                    # obtain min and max times and bring back to correct UTC time
                    mintimelist.append(trm.epoch_to_datetime(min(time_sel)) + 
                                       dt.timedelta(seconds = time_correction))
                    maxtimelist.append(trm.epoch_to_datetime(max(time_sel)) + 
                                       dt.timedelta(seconds = time_correction))
                                    
                    # obtain the tracking interval from the .npz file and repeat as 
                    # many times as we have velocities
                    tracking_interval = osp.basename(npzs[0]).split('_')[2].split('s')[0]      
                    tracking_interval = np.repeat(float(tracking_interval), len(u_sel)) 
                                    
                    if c_switch == 0:
                        
                        x_all = x_sel
                        y_all = y_sel
                        u_all = u_sel
                        v_all = v_sel
                        speed_all = speed_sel
                        tracking_interval_all = tracking_interval
                        
                        c_switch = 1
                    
                    else:
                        x_all = np.concatenate([x_all, x_sel]) 
                        y_all = np.concatenate([y_all, y_sel]) 
                        u_all = np.concatenate([u_all, u_sel]) 
                        v_all = np.concatenate([v_all, v_sel])
                        speed_all = np.concatenate([speed_all, speed_sel])
                        tracking_interval_all = np.concatenate([tracking_interval_all, tracking_interval]) 
                    
            #--------------------------------------------------------------------------
            
            # stop if there is no .npz file for selected day (len(cam_with_tracks) == 0)
            # or if there are no detected velocities for selected time window (len(x_all) == 0)
            if len(cam_with_tracks) == 0 or len(x_all) == 0:
                continue
                        
            # create coarse grid across the fjord    
            [polygons_coarse, centerpoints_coarse, indices, topleft, rows, cols] = trm.create_grid_across_fjord(fjord, grid_size)
            
            # adapt velocity shape
            points = np.vstack((x_all, y_all)).T
            point_displacements = np.vstack((u_all, v_all)).T
        
            # prep empty lists to store results for each grid cell 
            grid_id = []
            coarsegrid_i = []
            coarsegrid_j = []        
            coarsegrid_x = []
            coarsegrid_y = []
            coarsegrid_u = []
            coarsegrid_v = []
            coarsegrid_speed = []
            count = []
            
            polygons_measured = []
            polygons_not_measured = []
            
            # loop over grid and detect velocities within each grid cell    
            for counter, (poly, centerpoint, index) in enumerate(zip(polygons_coarse, centerpoints_coarse, indices)): 
                
                # check which velocities are contained in grid cell
                grid = mplPath.Path(poly).contains_points(points)
                points_selected = points[grid==1]
                displacements_selected = point_displacements[grid==1] 
                
                nr_observations = len(points_selected) 
                
                if nr_observations > observation_threshold:
                    
                    coarsegrid_i.append(index[0])
                    coarsegrid_j.append(index[1])
                    coarsegrid_x.append(centerpoint[0]) 
                    coarsegrid_y.append(centerpoint[1]) 
                    
                    # divide by # of observations to obtain average displacements in m/s
                    mean_u = np.sum(displacements_selected[:, 0]) / nr_observations
                    mean_v = np.sum(displacements_selected[:, 1]) / nr_observations
                    
                    coarsegrid_u.append(mean_u)
                    coarsegrid_v.append(mean_v) 
                    coarsegrid_speed.append(np.hypot(mean_u, mean_v))
                    
                    count.append(nr_observations)
                    grid_id.append(counter)
                    
                    polygons_measured.append(poly)
                    
                else:
                    polygons_not_measured.append(poly)
                    
            # round min and max time to the closest 30 minutes (used to annotate the .npz files)
            min_time = trm.round_time(min(mintimelist), 30 * 60)
            max_time = trm.round_time(max(maxtimelist), 30 * 60)          
            
            # save to .npz file        
            if time_window == 24.0:
                npz_name = '{}/{}-{}_full_day_{}m.npz'.format(target_path, min_time.strftime('%Y%m%d_%H%M'), 
                            max_time.strftime('%H%M'), grid_size)
                         
            else:
                npz_name = '{}/{}-{}_{}min_{}m.npz'.format(target_path, start_datetime.strftime('%Y%m%d_%H%M'), 
                             end_datetime.strftime('%H%M'), time_diff, grid_size) 
            
            np.savez(npz_name, grid_size = grid_size, topleft = topleft, rows = rows, cols = cols,
                     grid_id = grid_id, i = coarsegrid_i, j = coarsegrid_j, 
                     x = coarsegrid_x, y = coarsegrid_y, u = coarsegrid_u, v = coarsegrid_v, 
                     speed = coarsegrid_speed, count = count, measured = polygons_measured, 
                     not_measured = polygons_not_measured)
             
            #--------------------------------------------------------------------------
            # plotting
             
            if plot_switch == 1:
                
                plot_velocities_one_map(camnames_filtered, target_path, source_path_photos, day, day_str, start_datetime, 
                                        start_datetime_corr, end_datetime, min_time, max_time, time_window, 
                                        fjord, grid_size, cam_with_tracks, polygons_measured, polygons_not_measured, 
                                        coarsegrid_x, coarsegrid_y, coarsegrid_u, coarsegrid_v, coarsegrid_speed, 
                                        speedthreshold_cbar, paramfile)
                     
                                
            if plot_switch == 2: 
            
                plot_velocities_two_maps(camnames_filtered, target_path, source_path_photos, day, day_str, start_datetime, 
                                         start_datetime_corr, end_datetime, min_time, max_time, time_window, 
                                         fjord, grid_size, cam_with_tracks, polygons_measured, polygons_not_measured, 
                                         coarsegrid_x, coarsegrid_y, coarsegrid_u, coarsegrid_v, coarsegrid_speed, 
                                         speedthreshold_cbar, x_all, y_all, u_all, v_all, tracking_interval_all, 
                                         speed_all, paramfile)
                                 
        print(day_str + ' done...')

#%%
    
def plot_velocities_one_map(camnames, target_path, source_path_photos, day, day_str, start_datetime, 
                             start_datetime_corr, end_datetime, min_time, max_time, time_window, 
                             fjord, grid_size, cam_with_tracks, polygons_measured, polygons_not_measured, 
                             coarsegrid_x, coarsegrid_y, coarsegrid_u, coarsegrid_v, coarsegrid_speed, 
                             speedthreshold_cbar, paramfile):
    
        # set interactive plotting to off
        plt.ioff() 
      
        # figsize, x and y extent
        figsize = (14, 10.75)
        
        # Load the fjord npz file and grab bounding coordinates
        x_coords = fjord['x']
        y_coords = fjord['y']

        # Calculate the bounding box 300 m outside of the fjord outline (500m on the left side to have room for camera picture)
        min_x, max_x = int(np.min(x_coords)-500), int(np.max(x_coords)+300)
        min_y, max_y = int(np.min(y_coords)-300), int(np.max(y_coords)+300)

        # Set the x and y limits
        xlim = [min_x, max_x] 
        ylim = [min_y, max_y]
        
        # ticks on x and y axes 
        xticks = range(min_x, max_x, 500) 
        yticks = range(min_y, max_y, 500)
        

        
        # create figure and axes        
        fig, ax = plt.subplots(1, 1, figsize = figsize, facecolor = 'w')
                                       
        # add coarse grid to plot
        ax.add_collection(PolyCollection(polygons_measured, color = 'none', 
                                          linewidths = 0.5, edgecolor = 'darkgray'))
        
        ax.add_collection(PolyCollection(polygons_not_measured, color = 'lightgray', 
                                          linewidths = 0.5, edgecolor = 'darkgray'))
        
        # call the scaling function to scale the velocities
        [coarsegrid_u_adapted, coarsegrid_v_adapted] = trm.scale_arrows(coarsegrid_u, coarsegrid_v, exponent = 0.2, factor = 100)
        
        # plot the velocities
        quivers = ax.quiver(coarsegrid_x, coarsegrid_y, coarsegrid_u_adapted, 
                             coarsegrid_v_adapted, coarsegrid_speed,
                             clim = [0.0, speedthreshold_cbar], pivot = 'mid', 
                             cmap = 'gist_rainbow', units = 'x', 
                             scale = 1, width = 4, alpha = 1, zorder = 1000)
        
        # create and annotate strings        
        datestring = 'Date: {}'.format(day.strftime('%Y-%m-%d'))

        if time_window == 24.0:
            timestring = 'Time: ' + min_time.strftime('%H:%M') + util.minus_formatter('-') +\
                        max_time.strftime('%H:%M') + ' UTC'             
        else:
            timestring = 'Time: ' + start_datetime.strftime('%H:%M') + util.minus_formatter('-') +\
                end_datetime.strftime('%H:%M') + ' UTC'

        gridstring = 'Grid spacing: {} m'.format(grid_size)

        if len(cam_with_tracks) == 1:
            camstring = 'Camera: {}'.format(str(cam_with_tracks)[1:-1].replace("'", ''))
        else:
            camstring = 'Cameras: {}'.format(str(cam_with_tracks)[1:-1].replace("'", ''))

        for camname in camnames:

            # obtain parameters for specified day from parameter file 
            parameters = paramfile.loc[(paramfile['camera'] == camname) & 
            (paramfile['start_day'] <= int(day_str)) & 
            (paramfile['end_day'] >= int(day_str))]

            if len(parameters.index) == 1:

                # plot the camera location
                xcord = parameters['easting'].iloc[0]
                ycord = parameters['northing'].iloc[0]

                ax.plot(xcord, ycord, 'o', ms = 6, color = 'red')
                
                # annotate only one cam
                if camname == camnames[0]: 

                    if len(camnames) > 1:
                        ax.text(xcord - 120, ycord + 50, 'Cameras', fontsize = 18)
                    else:
                        ax.text(xcord - 120, ycord + 50, 'Camera', fontsize = 18)

                # check if camera location is close to left corner
                if xcord- min_x <= 500 and max_x - ycord  <= 500:
                    # if it is, plot details in right corner
                    util.annotatefun(ax, [datestring, timestring, camstring, gridstring], 0.98, 0.98, 
                                    ydiff = 0.03, fonts = 18, col = 'k', ha='right')
                else:
                    # otherwise, plot details in left corner
                    util.annotatefun(ax, [datestring, timestring, camstring, gridstring], 0.02, 0.8, 
                                    ydiff = 0.03, fonts = 18, col = 'k')
        
        # plot fjord lines 
        for idnr in [0, 1, 2]:
            ax.plot(fjord['x'], fjord['y'], 
                    '-', lw = 0.6, color = 'k')
        
        # create color bar    
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size = '2%', pad = 0.1)              
        cb = fig.colorbar(quivers, cax = cax)
        cb.set_label(label = 'Speed (m/s)', labelpad = 10, size = 15)
        cb.set_ticks([0.0, speedthreshold_cbar])
        
        # set figure extents and ticks               
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
                
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        #------------------------
        # plotting of original photos on inset axes 
        
        # allow for loading truncated images
        ImageFile.LOAD_TRUNCATED_IMAGES = True
                
        for cam_with_track in cam_with_tracks:
            
            # select image in the middle of the time window 
            if time_window == 24.0:
                phototime = min_time + (max_time - min_time) / 2  
            else:
                phototime = start_datetime_corr + dt.timedelta(hours = time_window / 2.0)
            
            image = trm.return_closest_image(osp.join(source_path_photos, cam_with_track), 
                                             phototime, 300)
                        
            # no image available within 300 seconds of target time
            if image == -99:
                continue
            
            # add inset axes
            axins = fig.add_axes([0.06, 0.06, 0.2, 0.2], anchor = 'SW', zorder = 10) 
          
                
            util.annotatefun(axins, [cam_with_track], 0.05, 0.85, ydiff = 0.05, 
                                 fonts = 18, col = 'k')
            
            # load and plot photo
            frame = np.array(Image.open(image))
            axins.imshow(frame)            
                            
            axins.set_xlim([0, frame.shape[1]])
            axins.set_ylim([frame.shape[0], 0])
            axins.axis('off')
            
        fig.tight_layout()
    
        if time_window == 24.0:
            
            plotname = osp.join(target_path, '{}-{}.png'.format(min_time.strftime('%Y%m%d_%H%M'), 
                                max_time.strftime('%H%M')))                
        else:
            
            plotname = osp.join(target_path, '{}-{}.png'.format(start_datetime.strftime('%Y%m%d_%H%M'), 
                    end_datetime.strftime('%H%M')))
                         
        plt.savefig(plotname, format = 'png', dpi = 100)
        
        plt.close(fig)
        
        
def plot_velocities_two_maps(camnames, target_path, source_path_photos, day, day_str, start_datetime, 
                             start_datetime_corr, end_datetime, min_time, max_time, time_window, 
                             fjord, grid_size, cam_with_tracks, polygons_measured, polygons_not_measured, 
                             coarsegrid_x, coarsegrid_y, coarsegrid_u, coarsegrid_v, coarsegrid_speed, 
                             speedthreshold_cbar, x_all, y_all, u_all, v_all, tracking_interval_all, 
                             speed_all, paramfile):
      
        # set interactive plotting to off
        plt.ioff()       

        # figsize, x and y extent
        figsize = (28, 10.75)

        # Load the fjord npz file and grab bounding coordinates
        x_coords = fjord['x']
        y_coords = fjord['y']

        # Calculate the bounding box 300 m outside of the fjord outline (1000m on the left side to have room for camera picture)
        min_x, max_x = int(np.min(x_coords)-3000), int(np.max(x_coords)+300)
        min_y, max_y = int(np.min(y_coords)-300), int(np.max(y_coords)+300)
        
        # Set the x and y limits
        xlim = [min_x, max_x] 
        ylim = [min_y, max_y]
        
        # ticks on x and y axes 
        xticks = range(min_x, max_x, 500) 
        yticks = range(min_y, max_y, 500)
        
        # create figure and axes        
        fig, axes = plt.subplots(1, 2, figsize = figsize, facecolor = 'w')
        [ax2, ax1] = axes
        
        #------------------------
        # items on ax1 only        
                               
        # add coarse grid to plot
        ax1.add_collection(PolyCollection(polygons_measured, color = 'none', 
                                          linewidths = 0.5, edgecolor = 'darkgray'))
        
        ax1.add_collection(PolyCollection(polygons_not_measured, color = 'lightgray', 
                                          linewidths = 0.5, edgecolor = 'darkgray'))
        
        # call the scaling function to scale the velocities
        [coarsegrid_u_adapted, coarsegrid_v_adapted] = trm.scale_arrows(coarsegrid_u, coarsegrid_v)
        
        # plot the velocities
        quivers = ax1.quiver(coarsegrid_x, coarsegrid_y, coarsegrid_u_adapted, 
                             coarsegrid_v_adapted, coarsegrid_speed,
                             clim = [0.0, speedthreshold_cbar], pivot = 'mid', 
                             cmap = 'gist_rainbow', units = 'x', 
                             scale = 1, width = 8, alpha = 1, zorder = 1000)
        
        # create and annotate strings        
        datestring = 'Date: {}'.format(day.strftime('%Y-%m-%d'))
        
        if time_window == 24.0:
            timestring = 'Time: ' + min_time.strftime('%H:%M') + util.minus_formatter('-') +\
                         max_time.strftime('%H:%M') + ' UTC'             
        else:
            timestring = 'Time: ' + start_datetime.strftime('%H:%M') + util.minus_formatter('-') +\
                 end_datetime.strftime('%H:%M') + ' UTC'
        
        gridstring = 'Grid spacing: {} m'.format(grid_size)

        # util.annotatefun(ax1, [datestring, timestring, gridstring], 0.03, 0.95, 
        #                  ydiff = 0.05, fonts = 18, col = 'k')
        
        #------------------------
        # items on ax2 only 
        
        # plot the velocities            
        quivers = ax2.quiver(x_all, y_all, u_all * tracking_interval_all, v_all * tracking_interval_all, speed_all, 
                            clim = [0.0, speedthreshold_cbar], cmap = 'gist_rainbow', units = 'x', 
                            scale = 1.0, width = 3.5, alpha = 0.75)
                            
        if len(cam_with_tracks) == 1:
            camstring = 'Camera: {}'.format(str(cam_with_tracks)[1:-1].replace("'", ''))
        else:
            camstring = 'Cameras: {}'.format(str(cam_with_tracks)[1:-1].replace("'", ''))
                    
        # util.annotatefun(ax2, [datestring, timestring, camstring], 0.03, 0.95, 
        #                  ydiff = 0.05, fonts = 18, col = 'k')
                         
        #------------------------
        # items on both axes 
                         
        for camname in camnames:
            
            # obtain parameters for specified day from parameter file 
            parameters = paramfile.loc[(paramfile['camera'] == camname) & 
            (paramfile['start_day'] <= int(day_str)) & 
            (paramfile['end_day'] >= int(day_str))]
        
            if len(parameters.index) == 1:
            
                # plot the camera location
                xcord = parameters['easting'].iloc[0]
                ycord = parameters['northing'].iloc[0]
                
                for ax in axes:
                    ax.plot(xcord, ycord, 'o', ms = 6, color = 'red')
                    
                    # annotate only one cam
                    if camname == camnames[0]: 
                    
                        if len(camnames) > 1:
                            ax.text(xcord - 120, ycord + 50, 'Cameras', fontsize = 18)
                        else:
                            ax.text(xcord - 120, ycord + 50, 'Camera', fontsize = 18)

        for ax in axes: 
        
            # plot fjord lines 
            for idnr in [0, 1, 2]:
                ax.plot(fjord['x'], fjord['y'], 
                        '-', lw = 0.6, color = 'k')
            
            # create color bar    
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size = '2%', pad = 0.1)              
            cb = fig.colorbar(quivers, cax = cax)
            cb.set_label(label = 'Speed (m/s)', labelpad = 10, size = 15)
            
            # set figure extents and ticks               
            ax.get_yaxis().get_major_formatter().set_useOffset(False)
            
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
                    
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            #check if camera location is close to left corner
            if xcord- min_x <= 500 and max_x - ycord  <= 500:
                # if it is, plot details in right corner
                util.annotatefun(ax, [datestring, timestring, camstring, gridstring], 0.98, 0.98, 
                                ydiff = 0.03, fonts = 18, col = 'k', ha='right')
            else:

                #plot details in left corner
                util.annotatefun(ax, [datestring, timestring, camstring, gridstring], 0.02, 0.98, 
                                ydiff = 0.03, fonts = 18, col = 'k')
                    
        #------------------------
        # plotting of original photos on inset axes 
        
        # Define positions for the inset axes
        inset_positions = [
            [0.03, 0.06, 0.18, 0.18],  # Position for the first camera image
            [0.03, 0.26, 0.18, 0.18],  # Position for the second camera image
            [0.03, 0.46, 0.18, 0.18],  # Position for the third camera image
            [0.03, 0.66, 0.18, 0.18]   # Position for the fourth camera image
        ]
        # Ensure the number of positions matches the number of cameras
        assert len(inset_positions) >= len(cam_with_tracks), "Not enough positions for all camera images"

        # allow for loading truncated images
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        for idx, cam_with_track in enumerate(cam_with_tracks):
            
            # select image in the middle of the time window 
            phototime = start_datetime_corr + dt.timedelta(hours = time_window / 2.0)            
            image = trm.return_closest_image(osp.join(source_path_photos, cam_with_track), 
                                             phototime, 300)
            
            # no image available within 300 seconds of target time
            if image == -99:
                continue
            
#            
            # Add inset axes at the specified position
            axins = fig.add_axes(inset_positions[idx], anchor='SW', zorder=10) 
            
            
            util.annotatefun(axins, [cam_with_track], 0.05, 0.85, ydiff = 0.05, 
                                 fonts = 18, col = 'black')
            
            # load and plot photo
            frame = np.array(Image.open(image))
            axins.imshow(frame)            
                            
            axins.set_xlim([0, frame.shape[1]])
            axins.set_ylim([frame.shape[0], 0])
            axins.axis('off')
            
        fig.tight_layout() 
    
        if time_window == 24.0:
            
            plotname = osp.join(target_path, '{}-{}.png'.format(min_time.strftime('%Y%m%d_%H%M'), 
                                max_time.strftime('%H%M')))                
        else:
            
            plotname = osp.join(target_path, '{}-{}.png'.format(start_datetime.strftime('%Y%m%d_%H%M'), 
                    end_datetime.strftime('%H%M')))
                         
        plt.savefig(plotname, format = 'png', dpi = 100)
        
        plt.close(fig)
        
#%%
    
if __name__ == '__main__':
    
    main()