#!/usr/bin/env python

import os
import os.path as osp
import subprocess
import glob
import platform

import pandas as pd
import datetime as dt
import numpy as np

import cv2
from PIL import Image

import matplotlib.pyplot as plt 
from matplotlib import collections as mc

from imports.stop_watch import stop_watch 
import imports.utilities as util
import imports.camtools as ct

from pathlib import Path

#%%

''' 
script to track icebergs on oblique photos

VARIABLES:

camnames: 
    list of camera names (e.g., ['UAS7', 'UAS8'] or single camera name (e.g. 'UAS7') 
    name must match the folder name in 'source_workspace' 
    it also has to match the camera name in the parameter excel file 'paramfile_name' 
    
source_workspace:
    source workspace, e.g., '/hdd2/leconte_2018/cameras'
    source workspace must contain subfolders for each camera (e.g. 'UAS7')
    each of these subfolders must contain daily folders with timestamped
    images, as created with the script s0_rename_jpgs.py
    
target_workspace:
    workspace for results, e.g., '/raid/image_processing_2018'
    target workspace will contain subfolders for each camera processed (e.g. 'UAS7'),
    each of these subfolders will have a folder 'oblique', which will contain  
    the tracking results (numpy .npz files) in daily folders
    target_workspace/camname/oblique/%Y%m%d/.npz files
    
paramfile_name:
    excel file featuring processing parameters 
    e.g., camera calibration parameters, parameters to crop the photos when copied
    from source_workspace to target_workspace, path to fjord mask (restricts tracking to fjord)
    file needs to be placed under ./ (in the same directory as this script) 
    e.g., 'parameter_file_2018.xlsx' 
      
min_date: 
    start date of period of interest, e.g. 20180904 
max_date: 
    end date of period of interest, e.g. 20180905
n_proc: 
    number of processors used during processing, e.g. 8

track_len: 
    track length in number of images, e.g. 2. '2' means corners are 
    detected on first image of an image triplet and tracked over two subsequent images: 
    [0,1,2][2,3,4][4,5,6]...
    track_len should be at least 2
    greater track_lens yield increasingly reliable yet thin fields
    greater track_lens work best for photos taken at high rates

startlist:
    list to control at what rate new corners are detected (e.g., every image, every other image)
    in case startlist = [0] and track_len = 2, processing is as follows: [0,1,2][2,3,4]... end
    in case startlist = [0, 1] and track_len = 2, processing is a follows: [0,1,2][2,3,4]... end
    ... start over... [1,2,3][3,4,5] ... done
    another example: in case startlist = [0, 2] and track_len = 4, processing is a follows:
    [0,1,2,3,4][4,5,6,7,8]... end ... start over... [2,3,4,5,6][6,7,8,9,10] ... done
    note: numbers in startlist must be integers that are smaller than track_len
    longer startlists provide additional (potentially new) vectors, but also reflect velocities 
    multiple times; also processing will take longer
                        
mask_switch
    switch [1 or 0] to control whether the water portion of the fjord is masked with a 
    predefined mask (mask name is defined in the parameter excel file)
    
plot_switch
    switch [1 or 0] to control whether plots with tracking results are created in the
    subfolders target_workspace/camname/plots
    
movie_switch
    switch [1 or 0] to control whether above plots are animated, 
    one movie per day under target_workspace/camname/plots 
    uses mencoder and works on linux only
    
delete_jpgs_switch:
    switch [1 or 0] to control whether cropped photos are deleted after processing 

NOTE: 
    parameters of the Shi-Tomasi corner detector and the Lucas-Kanade tracker 
    are defined in the lucaskanade_tracking function      
'''   
 
def main():
    
    # parameters
    # camnames = ['cam4']
    camnames = ['cam1','cam2','cam3','cam4']
    # source_workspace = Path('/hdd3/opensource/iceberg_tracking/data/')
    source_workspace = Path('G:/Glacier/GD_ICEH_iceHabitat/data') #Path would take care of trailing slash
    
    # #temporary
    # source_workspace = Path(r'D:\U\Glacier\GD_ICEH_iceHabitat\data') #Path would take care of trailing slash
    # source_workspace = str(source_workspace)
    
    # target_workspace = Path('/hdd3/opensource/iceberg_tracking/output/')
    target_workspace = Path('G:/Glacier/GD_ICEH_iceHabitat/output') #Path would take care of trailing slash
    # target_workspace = Path(r'D:\U\Glacier\GD_ICEH_iceHabitat\output') #Path would take care of trailing slash
        
    paramfile_name = 'parameter_file_2019.xlsx'
    
    min_date = 20190724
    max_date = 20190726
    n_proc = 8 #14

    track_len = 2
    startlist =  [0] 
    mask_switch = 1
    
    plot_switch = 1
    movie_switch = 1
    delete_jpgs_switch = 0
    
    #--------------------------------------------------------------------------
    
    # determine directory of current script (does not work in interactive mode)
    # file_path = osp.dirname(osp.realpath(__file__))
    file_path=Path(r'D:\U\iceberg_tracking_code')

    # read parameter file      
    # paramfile_path = osp.join(file_path, paramfile_name)             
    paramfile_path = Path(source_workspace, paramfile_name)             
    paramfile = pd.read_excel(paramfile_path)
    
    # create a new folder in the results workspace if folder does not yet exist
    if osp.isdir(target_workspace) == 0:
        os.makedirs(target_workspace)
        
    # Create a README file in the target_workspace with tracklength and startlist information 
    #AKB note: does this get used anywhere? I can find another reference to it in code
    #     assume it is just for user and so format doesn't matter. Adjusting commas...
    # with open(osp.join(target_workspace, 'README.md'), 'w') as f:
    #     f.write(f"Track Length: {track_len}\n,")
    #     f.write(f"Start List: {startlist}\n,")   
    #     f.write(f"Cameras: {camnames}\n,")
    #     f.write(f"Time Period: {min_date} - {max_date}\n,")
    #     f.write(f"Parameters: {paramfile_name}\n")
    # 
    #     #AKB note: why is this repeated?
    #     for item in startlist:
    #         f.write(f"- {item}\n")
    with open(Path(target_workspace, 'README.md'), 'w') as f:
        f.write(f"Track Length: {track_len}\n")
        f.write(f"Start List: {startlist}\n")   
        f.write(f"Cameras: {camnames}\n")
        f.write(f"Time Period: {min_date} - {max_date}\n")
        f.write(f"Parameters: {paramfile_name}\n")

    # start timer
    sw = stop_watch()
    sw.start()
    
    # check whether camnames is a list, and create list if not
    if type(camnames) != list:
        camnames = [camnames] 
    
    for camname in camnames:
        
        print('\n\nprocessing: ' + camname) #\n\n adds blank lines
    
        # list daily folders in source workspace 
        # 20 followed by ? matches the folder name structure, e.g., 20180810
        workspaces = sorted(glob.glob(osp.join(source_workspace, camname, '20??????')))
        # keep only folders within the defined time range     
        for wsp in workspaces[:]:
            if int(osp.basename(wsp)) < min_date or int(osp.basename(wsp)) > max_date:
                workspaces.remove(wsp)

        foldernr = len(workspaces)
        if foldernr == 0:
            print('No photos available for camera {} in timeperiod {}-{}'.format(camname, min_date, max_date))
         
        # loop over the daily folders and launch tracking script for each folder 
        #TODO: parallelize here?
        for counter, wsp in enumerate(workspaces, start = 1): #start is starting value for the counter
            date = osp.basename(wsp)

            # obtain the parameters for day of interest
            # parameters are used to crop the images and mask the fjord
            parameters = paramfile.loc[(paramfile['camera'] == camname) & 
            (paramfile['start_day'] <= int(date)) & 
            (paramfile['end_day'] >= int(date))] 
            
            # if there are no parameters, skip that day
            if len(parameters) == 0:
                print(camname + ' ' + date + ': no parameter file for this day available')
                continue
            
            tgw = osp.join(target_workspace, camname, 'oblique', osp.basename(wsp))
            # tgw = Path(target_workspace, camname, 'oblique', osp.basename(wsp))
            
            # create a new folder in the results workspace if folder does not yet exist
            if osp.isdir(tgw) == 0:
                os.makedirs(tgw)
                        
            # obtain time difference between photos from parameter file 
            track_len_sec = parameters['tracking_interval'].iloc[0] 
                                          
            lucaskanade_tracking(file_path, wsp, tgw, camname, track_len, track_len_sec,
                              startlist, mask_switch, plot_switch,
                              movie_switch, delete_jpgs_switch, paramfile_path, 
                              n_proc)
                        
            print('Folder {} / {} done for {} {}'.format(counter, foldernr,camname, date))
    
            sw.print_intermediate_time()
                
    # stop timer        
    sw.print_elapsed_time()
    sw.stop() 
    
#%%
    
def lucaskanade_tracking(file_path, ws_source, ws_target, camname, track_len, track_len_sec, 
                         startlist, mask_switch, plot_switch, movie_switch, 
                         delete_jpgs_switch, paramfile_path, n_proc):
    
    #-----------------------------------------------
    # parameters for Shi-Tomasi corner detector
    feature_params = dict(maxCorners = 50000000, 
                          qualityLevel = 0.007, 
                          minDistance = 10, 
                          blockSize = 10)    
    
    # parameters for the Lucas-Kanade tracker                       
    lk_params = dict(winSize = (35, 35), 
                     maxLevel = 4, 
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.03))                  
    #-----------------------------------------------
       
    datestring = osp.basename(ws_source)
                
    # create the camera object
    cam = ct.Camera(camname = camname, date = datestring, paramfile_path = paramfile_path, mask = mask_switch)
    
    # list the imagery in the original folder  
    imagelist = sorted(glob.glob(ws_source + '/*.jpg'))
    #AKB NOTE: ws_source is string now, if it were Path, then would use this:
    # imagelist=sorted(ws_source.glob('*.jpg'))

    # avoid folders with image_numbers smaller than track length (would throw errors)
    if len(imagelist) > track_len:
        
        # create a new folder in the target workspace if folder does not exist
        #AKB NOTE: does mkdirs above take care of all these, or are there more cases?
        if osp.isdir(ws_target) == 0:
            os.mkdir(ws_target)
            print(f'mkdir in loop ran for {ws_target}')
                
        # crop the photos with camera specific values (specified in the excel file) 
        # and save the cropped images in the target folder
        cam.crop_image_parallel(imagelist, ws_target, n_proc)    
        #TODO/AKB NOTE: if crop values are all 0, don't run crop code or copy/paste all files
        
        #----------------------------------------------- 
               
        # list cropped photos
        imagelist = sorted(glob.glob(ws_target + '/*.jpg'))
        
        # open first frame and convert it to grayscale to use it as a template 
        # for the mask (mask makes sure corners are detected only within fjord)
        frame = np.array(Image.open(imagelist[0])) 
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        mask = np.zeros_like(frame_gray)
        
        if mask_switch == 1: 
                    
            y, x = np.mgrid[0:frame_gray.shape[0], 0:frame_gray.shape[1]]
            mask1 = cam.mask_meshgrid(x, y, origin = 'upper left')
            mask[mask1==1] = 255
            
        else: # use no mask, set everything to 255
            mask[:] = 255
        
        # empty lists for tracks and track quality 
        # tracks is a list of lists, with each sublist containing the x and y 
        # coordinates of the vertices
        # trackquality contains the pixel distance between tracked and backtracked
        # corners
        tracks = []
        trackquality = []
        
        for start in startlist:        
        
            # loop over photos
            for counter, image in enumerate(imagelist[start:]):
                
                # open frame and convert to grayscale
                frame = np.array(Image.open(image)) 
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if len(tracks) > 0:
                    
                    # allocate images
                    img0 = prev_gray
                    img1 = frame_gray
                    
                    # obtain last point from each track
                    p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                    
                    # run tracker
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                    
                    # backtracking for match verification
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                    
                    # calculate distance between original and backtracked coordinates of p0
                    diff = abs(p0 - p0r).reshape(-1, 2)
                    dist = np.hypot(diff[:, 0], diff[:, 1])
                     
                    # create boolean array for the points with differences < 1 pixel
                    valid = dist < 1
                    
                    new_tracks = []
                    new_trackquality = []
                    
                    # loop over old tracks (tr) and corresponding new extensions (x ,y)
                    # only tracks that can be extented are retained
                    for tr, (x, y), valid_flag, trq, d in zip(tracks, p1.reshape(-1, 2), valid, trackquality, dist):
                        
                        # only add the track extension if flag says 1 (= valid)
                        if valid_flag == 1:
                        
                            # append x, y coordinates to the tracklist
                            tr.append((x, y))
                            trq.append(d)
                            
                            # delete track elements longer than track_len
                            # -1 to account for the fact that tr corresponds to the number of vertices
                            if (len(tr) - 1) > track_len:
                                del tr[0]
                                
                            new_tracks.append(tr)
                            new_trackquality.append(trq)
                    
                    # update the tracklist with new tracks
                    tracks = new_tracks
                    trackquality = new_trackquality 
        
                # save and plot tracking results, determine new corners
                if counter % track_len == 0:
                                                   
                    if counter > 0:
                        
                        # determine the time difference between images
                        # create two lists, one [x, x + 1], the other one [x + 1, x + 2]
                        jpgs1 = imagelist[start:][(counter - track_len) : counter]
                        jpgs2 = imagelist[start:][(counter - track_len + 1) : (counter + 1)]
                        
                        jpgs_time1 = [dt.datetime.strptime(osp.basename(jpg1), '%Y%m%d-%H%M%S.jpg') for jpg1 in jpgs1]
                        jpgs_time2 = [dt.datetime.strptime(osp.basename(jpg2), '%Y%m%d-%H%M%S.jpg') for jpg2 in jpgs2]
                        
                        # subtract two arrays to get time difference for each image pair
                        timediff = list(np.array(jpgs_time2) - np.array(jpgs_time1))
                        
                        # convert to seconds
                        timediff = [tdiff.seconds for tdiff in timediff]
            
                        kswitch = 1
            
                        for tdiff in timediff:
                            
                            # clock may drift, so allow for +- 2 seconds
                            if tdiff not in [track_len_sec - 2, track_len_sec - 1, track_len_sec, 
                                             track_len_sec + 1, track_len_sec + 2]:
                                
                                kswitch = 0

                        if kswitch == 1: 
                            
                            # only save to .npz if the time difference between images matches the expected time difference
                            # check is required because some cameras missed photos at random
                            npzname = '{}_{}sec_at_{}sec_tracks.npz'.format(path.split('.')[0], track_len * track_len_sec, track_len_sec)
                            np.savez(npzname, tracks = tracks, trackquality = trackquality)
                    
                            if plot_switch == 1:
                                
                                plot_ws = osp.join(ws_target, 'plots') 
                                
                                if osp.isdir(plot_ws) == 0:
                                    os.makedirs(plot_ws)                                
                                
                                # turn interactive plotting off
                                plt.ioff() 
                                
                                figsize = (15.0, 15.0 * frame_gray.shape[0] / frame_gray.shape[1])
                                fig, ax = plt.subplots(1, 1, figsize = figsize, facecolor = 'w')
                                
                                ax.imshow(frame_gray, cmap = 'gray')
                                                                       
                                # line collection is ideal to print many lines            
                                ax.add_collection(mc.LineCollection(tracks, color = 'red', alpha = 0.4))
                                
                                endpoints = np.float32([tr[-1] for tr in tracks]).reshape(-1, 2)
                                
                                ax.plot(endpoints[:, 0], endpoints[:, 1], '.', color = 'red', ms = 2.5, alpha = 0.6)
                                
                                ax.set_xlim([0, frame_gray.shape[1]])
                                ax.set_ylim([frame_gray.shape[0], 0])                            
                                ax.set_xticklabels([])
                                ax.set_yticklabels([])
                                
                                fig.tight_layout()
                                
                                util.annotatefun(ax, ['Displacement over {} seconds, tracking every {} seconds'.format(track_len * track_len_sec, 
                                                      track_len_sec), osp.splitext(osp.basename(image))[0]], 0.03, 0.93, fonts = 22, col = '#2b8cbe') #'white')
                                              
                                plotname = '{}/{}_{}sec.png'.format(plot_ws, osp.splitext(osp.basename(image))[0], 
                                            track_len * track_len_sec)
                                
                                plt.savefig(plotname, format = 'png', dpi = 80)
                                
                                plt.close(fig)
                    
                    # call Shi-Tomasi corner detector
                    p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                                    
                    # reset tracks
                    tracks = []
                    trackquality = []
                    # path string required to save the .npz file with the correct name
                    path = image
                    
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            tracks.append([(x, y)])
                            trackquality.append([])
                                    
                prev_gray = frame_gray
        
        if plot_switch == 1 and movie_switch == 1: 
        
            # determine whether Linux or Windows
            if platform.system() == 'Windows':
                #AKB added create_animation code to utilities.py
                #this version may work on Linux too, only tested for Windows
                util.create_animation(plot_ws, 'tracks_oblique_{}sec.avi'.format(track_len * track_len_sec))
                
            else:
        
                dim1 = 2000
                dim2 = -10
                moviename = 'tracks_oblique_{}sec.avi'.format(track_len * track_len_sec) 
                
                # determine directory of current script (does not work in interactive mode)
                #NOTE: commmented here, now gets passed in.
                # file_path = osp.dirname(osp.realpath(__file__))
                   
                shell_script = osp.join(file_path, 'imports', 'timelapse.sh') 
        
                subprocess.call([shell_script, plot_ws + '/', str(int(dim1)), 
                                 str(int(dim2)), moviename, '8', '*.png'])
                
        if delete_jpgs_switch == 1:
            # delete the cropped .jpgs to open up space                       
            for img in imagelist:
                
                os.remove(img)    
    
#%%
      
if __name__ == '__main__':
    
    main()
