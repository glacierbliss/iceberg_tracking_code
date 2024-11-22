#%%

#!/usr/bin/env python

'''
script to test the LucasKanade tracker
requires time-stamped photos as input and outputs plots of photos including tracks
based on openCV example code
'''

import glob  
import os
import numpy as np

import cv2
from PIL import Image

import matplotlib.pyplot as plt 
from matplotlib import collections as mc
from pathlib import Path

#------------------------------------------------------------------------------
                      
def annotatefun(ax, textlist, xstart, ystart, ydiff=0.05, fonts=12, col='k', ha='left'):
    for counter, textstr in enumerate(textlist):
        ax.text(xstart, (ystart - counter * ydiff), textstr, fontsize=fonts, ha=ha, 
                va='center', transform=ax.transAxes, color=col)
    
class LucasKanade:
    
    def __init__(self, workspace, detect_interval, time_spacing):        

        self.detect_interval = detect_interval # over how many photos to track
        self.time_spacing = time_spacing # time difference of individual photos in seconds
        
        # parameters for Shi-Tomasi corner detector 
        self.feature_params = dict(maxCorners=50000000,
                  qualityLevel=0.007, # test and increase/decrease if needed
                  minDistance=10,
                  blockSize=10)
        
        # paramters for Lucas-Kanade tracker
        self.lk_params = dict(winSize=(35, 35), # test and increase/decrease if needed 
                              maxLevel=4, # test and increase/decrease if needed 
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.03))

        self.track_len = self.detect_interval
        self.tracks = []
        self.imagelist = sorted(glob.glob(workspace + '/' +'*.jpg'))
        #asdf=sorted(glob.glob(workspace + '/' +'*.jpg'))
        self.imagelist2=sorted(workspace2.glob('*.jpg'))
        #qwer=sorted(workspace2.glob('*.jpg'))
        self.distthreshold = 1.0
        self.mask = 0
        self.date = workspace.split('/')[-1]
        self.date2=workspace2.parts[-1]
        self.workspace = workspace
                 
    def run(self):
        
        time_covered_plot = self.time_spacing * self.detect_interval
        
        plotfolder = '/plots_{}/'.format(time_covered_plot)
        
        try:
            os.mkdir(self.workspace + plotfolder)  
        except:
            pass

        frame = np.array(Image.open(self.imagelist[0])) 
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        
        # loop over sorted images
        for counter, image in enumerate(self.imagelist): 
            
            frame = np.array(Image.open(image)) #
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if len(self.tracks) > 0:
                
                # allocate the images
                img0 = self.prev_gray
                img1 = frame_gray
                
                # obtain the last point from each track
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                
                # run tracking
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
                
                # backtracking for match verification
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
                
                # calculates the difference between the original and the backtracked coordinates of p0
                diff = abs(p0 - p0r).reshape(-1, 2)
                dist = (diff[:, 0] ** 2 + diff[:, 1] ** 2) ** 0.5
                 
                # create boolean array for the points with differences < 1 pixel
                good = dist < self.distthreshold
                
                new_tracks = []
                
                # loop over old tracks (tr) and corresponding new extensions (x,y)
                # only the tracks that can be extented are retained
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    
                    # only add the track extension if flag says 1 (= good)
                    if good_flag == 1:
                    
                        # append x,y coordinates to the tracklist
                        tr.append((x, y))
                        
                        # delete track element that is longer than track_len threshold
                        # -1 to account for the fact that tr corresponds to the number of vertices
                        if (len(tr) - 1) > self.track_len:
                            del tr[0]
                            
                        new_tracks.append(tr)
                
                # update the tracklist with new tracks
                self.tracks = new_tracks

            # detect new features if modulus equals zero
            if counter % self.detect_interval == 0:
                
                print('{} tracks'.format(len(self.tracks)))
                
                endpoints = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 2)
                
                if counter > 0:                
                
                    fig = plt.figure(figsize=(12.0, 12.0 * frame_gray.shape[0] / frame_gray.shape[1]), facecolor='w')
                    ax = fig.add_subplot(111)
                    ax.imshow(frame_gray, cmap='gray')
                                                           
                    lc = mc.LineCollection(self.tracks, color='red', alpha=0.4)
                    ax.add_collection(lc)
                    
                    ax.plot(endpoints[:,0], endpoints[:,1], '.' , color='red', ms=2.5, alpha=0.6)
                    
                    ax.set_xlim([0, frame_gray.shape[1]])
                    ax.set_ylim([frame_gray.shape[0], 0])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    fig.tight_layout()
                    
                    photo_name = image.split('/')[-1].split('.')[0] 
     
                    annotatefun(ax, ['Displacement over {} seconds, sampling every {} seconds'.format(time_covered_plot, self.time_spacing), photo_name], 
					0.03, 0.95, fonts=22, col='white')
                    
                    workspace = self.workspace + plotfolder                 
                    
                    plotname = '{}{}_{}sec.png'.format(workspace, photo_name, time_covered_plot)
                    plotname2=workspace2/ f'{photo_name}_{time_covered_plot}sec.png'
                    print(image)
                    print(photo_name)
                    print(self.workspace)
                    print(plotfolder)
                    print(workspace)
                    print(plotname)
                    # G:/Glacier/GD_ICEH_iceHabitat/data/test\20190724-130256.jpg
                    # test\20190724-130256
                    # G:/Glacier/GD_ICEH_iceHabitat/data/test
                    # /plots_360/
                    # G:/Glacier/GD_ICEH_iceHabitat/data/test/plots_360/
                    # G:/Glacier/GD_ICEH_iceHabitat/data/test/plots_360/test\20190724-130256_360sec.png

                    plt.savefig(plotname, format='png', dpi=100)
                    
                    plt.close()
                    
                p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
                
                # add the features to the tracks already available (tracks is a list of lists,
                # with each sublist containing the x and y coordinates of the vertices)
                
                # reset tracks
                self.tracks = []
                
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])
                
            self.prev_gray = frame_gray

            print('{} / {} done...'.format(counter+1, len(self.imagelist))) #AKB added +1 to python 0-based count

       
if __name__ == '__main__':
    # workspace = 'G:\Glacier\GD_ICEH_iceHabitat\data\test\'
    # workspace = 'G:/Glacier/GD_ICEH_iceHabitat/data/test/'
    workspace = 'G:/Glacier/GD_ICEH_iceHabitat/data/test' #does better without trailing space (I think)
    workspace2 = Path('G:/Glacier/GD_ICEH_iceHabitat/data/test') #does better without trailing space (I think)
    #workspace = 'G:\\Glacier\\GD_ICEH_iceHabitat\\data\\test\\'
    #workspace = '/hdd3/opensource/iceberg_tracking/data/test/'
    detect_interval = 3 #set between 2 and 4
    time_spacing = 120 #set between 60 and 240
    plt.ioff()
    
    LucasKanade(workspace,
                 detect_interval, time_spacing).run() 




# %%
