#!/usr/bin/env python

import os
import os.path as osp
import glob
import warnings
import scipy

import pandas as pd
import datetime as dt
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

#from imports.stop_watch import stop_watch 
import imports.utilities as util
import imports.tracking_misc as trm

from pathlib import Path

try:
    warnings.filterwarnings('ignore')
    mpl.rc("font", **{"sans-serif": ["Arial"]})
except:
    pass

#%% stacks .npz files (many 1D files to one 3D file) and rearranges folder structure

'''
functions to stack .npz files (many 1D files to one 3D file) and to tidy up folder structure  
'''

def main():
    
    # parameters and switches
    # workspace = '/hdd3/opensource/iceberg_tracking2/output/run1/'#*
    workspace = Path('G:/Glacier/GD_ICEH_iceHabitat/output/run1')
    npz_combined_name = 'cam1_2_3_4_utm_200m_30min.npz'# *

    cleanup = 0 # delete original .npz files (1) or not (0)
    #TODO: FIX: if cleanup==1, can only run s4_postprocess_gridded_utm once or get an error
    #actually, other aspects of this code are too fragile to run a second time too (after cleanup = 0)
    #MEANWHILE: can just delete /run1/ and run s3_utm_to_gridded_utm again, then this again.
    #CONCLUSION: just leave cleanup=0 and fix all issues later
    
    # create dedicated directories for individual products
    movie_workspace = osp.join(workspace, 'movie')
    figure_workspace = osp.join(workspace, 'figures')
    npz_workspace = osp.join(workspace, 'npz')
    mat_workspace = osp.join(workspace, 'mat') 
    
    # Create directories
    create_directory(movie_workspace)
    create_directory(figure_workspace)
    create_directory(npz_workspace)
    create_directory(mat_workspace)

    
    # move figures into new folder
    figures = glob.glob(osp.join(workspace, '*.png'))
    
    for figure in figures:
        os.rename(figure, osp.join(figure_workspace, osp.basename(figure)))
    
    # move movie into new folder    
    movies = glob.glob(osp.join(workspace, '*.avi'))
    if movies:
        movie = movies[0]
        os.rename(movie, osp.join(movie_workspace, osp.basename(movie)))
    else:
        print("No .avi files found in the workspace.")
    
    
    # stack the many 1d .npz files into one 3d .npz file
    combine_npzs(workspace, npz_workspace, npz_combined_name)
    
    npz_combined = osp.join(npz_workspace, npz_combined_name) 
    
    # converts .npz file to .mat filesfor use in matlab              
    npz_to_mat(npz_combined, mat_workspace)
    
    # deletes the 1d .npz files that were previously converted
    # to one 3d .npz file
    if cleanup == 1:
    
        npzs = glob.glob(osp.join(workspace, '*.npz'))
        
        for npz in npzs:
            os.remove(npz) 
    

def create_nan_array(dim1=0, dim2=0, dim3=0):
    
    if dim3 == 0:
        if dim2 == 0:
            ar = np.empty((dim1))
        else:
            ar = np.empty((dim1, dim2))
    else:
        ar = np.empty((dim1, dim2, dim3))
          
    ar[:] = np.nan
    
    return ar

# Function to create directory if it doesn't exist
def create_directory(directory):
    try:
        if not osp.exists(directory):
            os.mkdir(directory)
            print(f"Directory created: {directory}")
        else:
            print(f"Directory already exists: {directory}")
    except OSError as e:
        print(f"Error creating directory {directory}: {e}")

def velocities_to_regular_grid(npz):
    
    '''converts 1d .npz files to regular 2d .npz files'''
    
    npz = np.load(npz)
    
    # load individual 1d arrays
    i = npz['i'] # colindex
    j = npz['j'] # rowindex
    x = npz['x'] # xcord in utm
    y = npz['y'] # ycord in utm
    u = npz['u'] 
    v = npz['v']
    speed = npz['speed']
    count = npz['count']
    grid_size = npz['grid_size']
    topleft = npz['topleft']
    cols = npz['cols'] # colnumber 
    rows = npz['rows'] # rownumber
    
    # create empty 2d arrays
    u_ras = create_nan_array(rows, cols)
    v_ras = create_nan_array(rows, cols)
    speed_ras = create_nan_array(rows, cols)
    x_ras = create_nan_array(rows, cols)
    y_ras = create_nan_array(rows, cols)
    count_ras = create_nan_array(rows, cols)
    
    # loop over 1d arrays and fill in 2d arrays     
    for p in range(0, len(npz['i'])):
                
        u_ras[j[p], i[p]] = u[p]
        v_ras[j[p], i[p]] = v[p]
        speed_ras[j[p], i[p]] = speed[p]
        x_ras[j[p], i[p]] = x[p]
        y_ras[j[p], i[p]] = y[p]
        count_ras[j[p], i[p]] = count[p]
    
    # create meshgrids in utm coordinates    
    x = np.arange(topleft[0], topleft[0] + cols * grid_size, grid_size)
    y = np.arange(topleft[1] - (rows - 1) * grid_size, topleft[1] + 1 * grid_size, grid_size) 

    xx, yy = np.meshgrid(x, y, indexing = 'xy') 
    yy = np.flipud(yy)
    
    # create meshgrids with row and col indices
    ii, jj = np.meshgrid(range(0, rows), range(0, cols), indexing = 'ij') 
        
    return [x_ras, y_ras, u_ras, v_ras, speed_ras, count_ras, xx, yy, ii, jj, cols, rows] 
    
def combine_npzs(folder, npz_workspace, npz_combined_name):
    
    '''stacks 1d .npz files into one 3d .npz file'''
    
    # list all the .npz files
    npzs = sorted(glob.glob(osp.join(folder, '*.npz')))
    
    # process the first file to derive key parameters
    [x_ras, y_ras, u_ras, v_ras, speed_ras, count_ras, xx, yy, ii, jj, cols, rows] = velocities_to_regular_grid(npzs[0])
    
    npz_nr = len(npzs) 
    
    # create empty arrays that have the correct length
    u = create_nan_array(rows, cols, npz_nr)
    v = create_nan_array(rows, cols, npz_nr)
    speed = create_nan_array(rows, cols, npz_nr)
    count = create_nan_array(rows, cols, npz_nr)
    
    time = create_nan_array(npz_nr)
    time_matlab = create_nan_array(npz_nr)
    
    # loop over .npz files and stack the resulting regular grids vertically
    for counter, npz in enumerate(npzs):
        
        [x_ras, y_ras, u_ras, v_ras, speed_ras, count_ras, xx, yy, ii, jj, cols, rows] = velocities_to_regular_grid(npz)
        
        u[:, :, counter] = u_ras
        v[:, :, counter] = v_ras
        speed[:, :, counter] = speed_ras 
        count[:, :, counter] = count_ras
        
        basename = osp.basename(npz).split('-')[0]
        
        time[counter] = trm.datetime_to_epoch(dt.datetime.strptime(basename, '%Y%m%d_%H%M'))
        time_matlab[counter] = util.datetime2matlab(dt.datetime.strptime(basename, '%Y%m%d_%H%M'))
        
    npz_combined = osp.join(npz_workspace, npz_combined_name)
        
    # save the files in one final .npz file
    np.savez(npz_combined, x = xx, y = yy, i = ii, j = jj, u = u, v = v, speed = speed, 
             count = count, time = time, time_matlab = time_matlab)

def npz_to_mat(np_file, targetfolder):
    
    '''converts .npz files to .mat files for use in matlab'''
    
    file_loaded = np.load(np_file)
        
    param_dict = {}
        
    param_dict['x'] = file_loaded['x']
    param_dict['y'] = file_loaded['y']
    param_dict['u'] = file_loaded['u']
    param_dict['v'] = file_loaded['v']
    param_dict['speed'] = file_loaded['speed']
    param_dict['count'] = file_loaded['count']
    param_dict['time'] = file_loaded['time_matlab']
    
    scipy.io.savemat(osp.join(targetfolder, osp.basename(np_file).replace('.npz', '.mat')), param_dict)
    
#%% functions to derive spatially and temporally averaged velocities from the 3D .npz file
    
'''
functions to derive spatially and temporally averaged velocities from the 3D .npz file 
'''
    
def spatial_mean(variable, coarseness = 2, nanmean = 1):
    
    '''averages variable spatially over a certain number of grid cells
    defined by the window coarseness x coarseness''' 
    
    # find the next highest multiple of coarseness x coarseness
    shape = np.array(variable.shape, dtype = int)
    new_shape = coarseness * np.ceil(shape / coarseness).astype(int)
    
    # create zero-padded array with new shape and assign the old values
    variable_pad = np.zeros(new_shape)
    variable_pad[:shape[0], :shape[1]] = variable    
    
    # reshape
    temp = variable_pad.reshape((new_shape[0] // coarseness, coarseness,
                             new_shape[1] // coarseness, coarseness))
    
    # average over the reshaped array, either nanmean or just mean
    if nanmean == 1:
        mean = np.nanmean(temp, axis = (1, 3))
    else:
        mean = np.mean(temp, axis = (1, 3))
        
    return mean
  
def average_spatially_temporally(start_time, end_time, coarseness, npz, title, fjord_outline_path, plot_num, viz):    
    
    '''averages velocities temporally over user-defined time periods, 
    and spatially over user-defined window size'''
        
    # convert times to epoch
    start_time_epoch = trm.datetime_to_epoch(start_time)
    end_time_epoch = trm.datetime_to_epoch(end_time)
    
    # load data from .npz file    
    time = npz['time'] 
    u = npz['u']
    v = npz['v']
    count = npz['count']  
    xx = npz['x']
    yy = npz['y']
    
    # create time mask and select data within mask
    time_mask = (time >= start_time_epoch) & (time < end_time_epoch)
    
    u_sel = u[:, :, time_mask]
    v_sel = v[:, :, time_mask]
    count_sel  = count[:, :, time_mask]  
    
    # average over time using nanmean
    u_mean = np.nanmean(u_sel, 2) 
    v_mean = np.nanmean(v_sel, 2) 
    speed_mean = np.hypot(u_mean, v_mean)
    count_sum = np.nansum(count_sel, 2)
    
    # filter out vectors with low counts
    # u_mean[count_sum < 100] = np.nan  
    # v_mean[count_sum < 100] = np.nan   
    
    # no data available at all
    if np.isnan(np.nanmean(speed_mean)):

        print('{} - {}: no data available...'.format(start_time, end_time))      
        
        return [np.nan, np.nan, np.nan, np.nan]
        
    else:
        
        # conduct spatial averaging
        if coarseness > 1:
        
            u_coarse = spatial_mean(u_mean, coarseness = coarseness, nanmean = 0) 
            v_coarse = spatial_mean(v_mean, coarseness = coarseness, nanmean = 0) 
            xx_coarse = spatial_mean(xx, coarseness = coarseness, nanmean = 0)
            yy_coarse = spatial_mean(yy, coarseness = coarseness, nanmean = 0) 
            speed_mean_coarse = np.hypot(u_coarse, v_coarse)
            
            
        #--------------------------------------------------------------------------
        # plotting
            
        # load fjord outline               
        fjord = np.load(fjord_outline_path) 
        
        # figsize, x and y extent

        # Load the fjord npz file and grab bounding coordinates
        x_coords = fjord['x']
        y_coords = fjord['y']

        # Calculate the bounding box 300 m outside of the fjord outline (500m on the left side to have room for camera picture)
        min_x, max_x = int(np.min(x_coords)-500), int(np.max(x_coords)+300)
        min_y, max_y = int(np.min(y_coords)-300), int(np.max(y_coords)+300)

        # Set the x and y limits
        xlim = [min_x, max_x] 
        ylim = [min_y, max_y]
        
        
        # calculate ratio between extents
        ratio = (float(xlim[1]) - xlim[0]) / (float(ylim[1]) - ylim[0])
        
        x_extent_ax = 10 # extent of main axes
        x_extent_ax_cb = 0.4 # extent of colorbar    
        
        x_space_left = 0.8 # space on left side of ax in cm
        x_space_right = 0.75 # space on right side of ax in cm
        x_space_between = 0.2 # space between main ax and colorbar ax
        
        y_space_bottom = 0.5 # space on bottom of ax 
        y_space_top = 0.6 # space on top of ax
    
        y_extent_ax = x_extent_ax / ratio
    
        x_extent_total = float(sum([x_space_left, x_extent_ax, x_space_between, 
                                    x_extent_ax_cb, x_space_right]))
        y_extent_total = float(sum([y_space_bottom, y_extent_ax, y_space_top]))
    
        figsize = (x_extent_total, y_extent_total)
         
        # create figure and add axes
        fig = plt.figure(figsize = figsize, facecolor = 'w', edgecolor = 'k')
        
        ax = fig.add_axes([x_space_left / x_extent_total, y_space_bottom / y_extent_total, 
                           x_extent_ax / x_extent_total, y_extent_ax / y_extent_total], 
                           anchor = 'SW', zorder = 10)
                           
        ax.set_title(title, size = 17, loc = 'center', y = 1.01)


        yy = np.flipud(yy)
        u_mean = np.flipud(u_mean)
        v_mean = np.flipud(v_mean)
        speed_mean = np.flipud(speed_mean)   

        if coarseness == 1:
        
            # call the scaling function to scale the velocities
            [coarsegrid_u_adapted, coarsegrid_v_adapted] = trm.scale_arrows(u_mean, v_mean, exponent = 0.2, factor = 100)
            
            if plot_num == 1:
                # plot the velocities
                quivers = ax.quiver(xx, yy, coarsegrid_u_adapted, 
                                    coarsegrid_v_adapted, speed_mean,
                                    clim = [0.0, 0.25], pivot = 'mid', 
                                    cmap = 'gist_rainbow', units = 'x', 
                                    scale = 0.2, width = 4, alpha = 1, zorder = 1000)
                plot_var = quivers

            if plot_num == 2:    
                # streamplot
                             
                strm = ax.streamplot(xx[0, :], yy[:, 0], u_mean, v_mean, color=speed_mean, linewidth=0.5, cmap='viridis',
                                    density=3, norm=mpl.colors.Normalize(0, 0.25), minlength=0.05, arrowsize=0.6, zorder=1)
                plot_var = strm.lines                    
        else:
            [coarsegrid_u_adapted_coarse, coarsegrid_v_adapted_coarse] = trm.scale_arrows(u_coarse, v_coarse, exponent = 0.2, factor = 100 * coarseness)

            if plot_num == 1:                     
            
                # plot the velocities, width changes with coarseness 
                quivers = ax.quiver(xx_coarse, yy_coarse, coarsegrid_u_adapted_coarse, 
                                    coarsegrid_v_adapted_coarse, speed_mean_coarse,
                                    clim = [0.0, 0.25], pivot = 'mid', 
                                    cmap = 'gist_rainbow', units = 'x', 
                                    scale = 0.2, width = 4 + coarseness, alpha = 1, zorder = 1000)
                plot_var = quivers
            if plot_num == 2:  
            # streamplot    
            # arrange the data in the correct order
                              
                strm = ax.streamplot(xx_coarse[0, :], yy_coarse[:, 0], u_coarse, v_coarse, color = speed_mean_coarse, 
                                    linewidth = 1.0, cmap = 'gist_rainbow', density = 1, 
                                    norm = mpl.colors.Normalize(0, 1.0), minlength = 0.1, arrowsize = 2.5)
                plot_var = strm.lines
                             
        # plot fjord lines
        if 'id' in fjord:
            #AKB version of fjord had no 'id' field, so this threw an error.
            for idnr in [0, 1, 2]:
                ax.plot(fjord['x'][fjord['id'] ==  idnr], fjord['y'][fjord['id'] == idnr], 
                        '-', lw = 0.6, color = 'k')
        else:
            ax.plot(fjord['x'], fjord['y'],'-', lw = 0.6, color = 'k')

        # create axes for colorbar            
        cax = fig.add_axes([(x_space_left + x_extent_ax + x_space_between) / x_extent_total, y_space_bottom/y_extent_total, 
                            x_extent_ax_cb/x_extent_total, y_extent_ax/y_extent_total], anchor = 'SW', zorder = 10) 
           
        cb = fig.colorbar(plot_var, cax = cax)
        cb.set_label(label = 'Speed (m/s)', labelpad = 10, size = 15)
        
        # set figure extents and ticks               
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
                    
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        if viz == 0:
            plt.show()
        else:
            if plot_num == 1:
                plt.savefig(osp.join(figure_workspace, title + '_quiver.png'), dpi = 300)
            if plot_num == 2:

                plt.savefig(osp.join(figure_workspace, title + '_streamline.png'), dpi = 300)
            plt.close()
        
        if coarseness == 1:    
            return [xx, yy, u_mean, v_mean]
            
        else:
            return [xx_coarse, yy_coarse, u_coarse, v_coarse]       
    
if __name__ == '__main__':
    main()
    
    #AKB NOTE: can I move all this up into main and streamline filenames?
    
    
    #time range for spatial averaging 
    days = pd.date_range('2019-07-24', '2019-07-26', freq = 'd') # create daily averages
    
    # path to .npz file
    # npz = np.load('/hdd3/opensource/iceberg_tracking/output/run1/npz/cam1_2_3_4_utm_200m_30min.npz')
    npz = np.load(Path('G:/Glacier/GD_ICEH_iceHabitat/output/run1/npz/cam1_2_3_4_utm_200m_30min.npz'))

    # fjord outline
    fjord_outline_name = 'fjord_outline.npz'
    # determine directory of current script (does not work in interactive mode)
    # file_path = osp.dirname(osp.realpath(__file__))
    file_path=Path('G:/Glacier/GD_ICEH_iceHabitat/data')
    fjord_outline_path = osp.join(file_path, fjord_outline_name)     

    # output folder
    # figure_workspace = '/hdd3/opensource/iceberg_tracking/output/post_process'
    figure_workspace = Path('G:/Glacier/GD_ICEH_iceHabitat/output/post_process')
    create_directory(figure_workspace)

    coarseness = 1
    plot_num = 1 # 1 for quiver plot, 2 for streamplot
    viz = 1 # 0 for visualization, 1 for save to file
    

    # 20190724_1300-1330_30min_200m.npz
    for day in days: 
        
         start_time = day + dt.timedelta(hours = 12) # measurements never start before noon UTC
         end_time = start_time + dt.timedelta(hours = 22) # measurements always last less than 24 hours      
        
         start_time_str = start_time.strftime('%Y-%m-%d %H:%M')
         end_time_str = end_time.strftime('%Y-%m-%d %H:%M')
        
         title = 'Daily average {}'.format(start_time.strftime('%Y-%m-%d'))
    
         [x, y, u, v] = average_spatially_temporally(start_time, end_time, coarseness, npz, title, fjord_outline_path, plot_num, viz)
