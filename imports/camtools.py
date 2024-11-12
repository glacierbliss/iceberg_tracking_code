#!/usr/bin/env python

import os.path as osp
import multiprocessing
import logging

from PIL import Image
from PIL import ImageFile

import numpy as np
import matplotlib.path as mplPath

import datetime as dt

import pandas as pd
import shapefile


def x_y_to_poly(x, y):
    
    '''turns lists of x and y coordinates into lists of tuples for polygons'''
    
    poly = [(x[0], y[0])]
    [poly.append((x[i], y[i])) for i in range(len(x)-1, -1, -1)]
        
    return poly
    
def x_y_from_shapefile(shp, tuples = 0):
    
    '''extracts x and y coordinates from points digitized on the oblique photograph'''
    
    shapes = shapefile.Reader(shp)
    
    shapetype = shapes.shapeType 
    
    shapes = shapefile.Reader(shp).shapes()
    
    if tuples == 0:
        
        # if point of line shapefile
        if shapetype != 5: # 5 means polygon shapefile
            xcoords = np.array([shape.points[0][0] for shape in shapes])
            ycoords = np.array([shape.points[0][1] * -1 for shape in shapes])
        
        else:
            xcoords = np.array([[shape.points[c][0] for c in range(0, len(shape.points))] 
            for shape in shapes])[0]
            ycoords = np.array([[shape.points[c][1] * -1 for c in range(0, len(shape.points))] 
            for shape in shapes])[0]
        
        return [xcoords, ycoords]
        
    else:
        if shapetype != 5:
            list_of_tuples = [(int(shape.points[0][0]), int(shape.points[0][1] * -1)) for shape in shapes]
            
        else:
            list_of_tuples = [[(int(shape.points[c][0]), int(shape.points[c][1] * -1)) 
            for c in range(0, len(shape.points))] for shape in shapes][0]
        
        return list_of_tuples
    
    
def crop_image_standalone(args):
    
    '''crops photos using predefined extents
    try/except structure accounts for truncated (partly saved) images, which
    caused cropping and tracking to crash'''
    
    inpath, outpath, width, height, cropleft, cropright, croptop, cropbottom = args
    
    try:
        
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        
        img = Image.open(inpath)
           
        # The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate
        img_crop = img.crop((cropleft, croptop, width - cropright, height - cropbottom))
        
        img_crop.save(outpath)
        
    except:
        
        # go two levels up to get the logfilepath
        logfilepath = osp.dirname(osp.dirname(outpath))
    
        logging.basicConfig(level = logging.DEBUG, filename = logfilepath + '/logfile_image_cropping.log')
        
        logging.info('------------------')
        logging.info(inpath)
        logging.info(dt.datetime.now().isoformat())
        logging.exception('')
        
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        img = Image.open(inpath)
           
        # The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate
        img_crop = img.crop((cropleft, croptop, width - cropright, height - cropbottom))
        
        img_crop.save(outpath)
        
        print(str(inpath) + ' TRUNCATED...')
         
    
class Camera(object):
    
    '''initiates the class as a function of the camera name'''
    
    def __init__(self, camname, date, paramfile_path, mask = 0, tide_corr = 0, tide_file = '', datetime = ''):
                       
        paramfile = pd.read_excel(paramfile_path)
                
        # obtain the parameters for specified day 
        parameters = paramfile.loc[(paramfile['camera'] == camname) & 
        (paramfile['start_day'] <= int(date)) & 
        (paramfile['end_day'] >= int(date))] 
        
        if len(parameters) == 0:
            
            raise ValueError('No calibration parameters found for this day')
            
        self.date = dt.datetime.strptime(date, '%Y%m%d').date() 
        
        self.cam = {}
        self.pic = {}
        
        self.pic['width'] = parameters['image_width'].iloc[0]
        self.pic['height'] = parameters['image_height'].iloc[0]
        self.cam['chipsize'] = parameters['sensor_width'].iloc[0]
        
        self.cam['E'] = parameters['easting'].iloc[0] # Easting in UTM
        self.cam['N'] = parameters['northing'].iloc[0] # Northing in UTM
        # elevation above waterlevel
        self.cam['H'] = parameters['elevation'].iloc[0] - parameters['antenna_height'].iloc[0]
                
        self.cam['theta'] = np.radians(parameters['theta'].iloc[0]) # azimuth from east
        self.cam['phi'] = np.radians(parameters['phi'].iloc[0]) # tilt angle
        self.cam['psi'] = np.radians(parameters['psi'].iloc[0]) # roll angle
        # enlargement factor * focal length
        self.cam['sigma'] = (self.pic['width'] / self.cam['chipsize']) * parameters['sigma'].iloc[0]
        
        self.pic['cropleft'] = parameters['crop_left'].iloc[0]
        self.pic['cropright'] = parameters['crop_right'].iloc[0]
        self.pic['croptop'] = parameters['crop_top'].iloc[0]
        self.pic['cropbottom'] = parameters['crop_bottom'].iloc[0]
        
        if mask == 1:
            
            # concatenate path to mask shapefile
            shp = osp.join(osp.dirname(paramfile_path), 'data', camname, parameters['mask'].iloc[0]) 
            
            self.maskpoly = x_y_from_shapefile(shp, tuples = 1)
            
        if tide_corr == 1:
            
            # create time object
            dt_orig = dt.datetime.strptime(datetime, '%Y%m%d-%H%M%S')
            
            # round to full minutes (we have one tide value per minute)
            datetime_floor = dt.datetime(year = dt_orig.year, month = dt_orig.month, 
                                         day = dt_orig.day, hour = dt_orig.hour, minute = dt_orig.minute)
            
            # load tide file into a dataframe
            tides = pd.read_pickle(osp.join(osp.dirname(paramfile_path), 'data', tide_file))

            # query the tide for the time of interest 
            tideelevation = tides.loc[tides['date'] == datetime_floor]['depth_tide_ellipsoid'].iloc[0]
            # check if the DataFrame is not empty
            if not tides.loc[tides['date'] == datetime_floor].empty:
                tideelevation = tides.loc[tides['date'] == datetime_floor]['depth_tide_ellipsoid'].iloc[0]
            else:
            # handle the case where the DataFrame is empty
            # for example, you can set tideelevation to None or to a default value
                tideelevation = None
                        
            # correct the camera elevation (positive tides -> 
            # distance between camera and water smaller)
            self.cam['H'] = self.cam['H'] - float(tideelevation)
        
    def mask_meshgrid(self, x, y, origin = 'lower left'):
        
        '''1) Moves the water polygon to account for cropping of the image
          2) Checks whether coordinates of x and y meshgrids lie within the polygon'''
        
        # move the waterpolygon
        cropleft = self.pic['cropleft']
        croptop = self.pic['croptop']
        poly = self.maskpoly
        
        poly_shift = [(a[0] - cropleft, a[1] - croptop) for a in poly]
        
        # check whether coordinates of x and y meshgrids lie within a polygon
        ny, nx = np.shape(x)
        
        if origin == 'lower left':         
        
            # y needs to be flipped because that meshgrid has origin in lower left corner
            # while the polygon has origin in upper left corner
            y = np.flipud(y) 
    
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T
    
        polypath = mplPath.Path(poly_shift)
        grid = polypath.contains_points(points).reshape((ny,nx))
        
        return grid
         
            
    def crop_image(self, inpath, outpath, rtrn_image = 0):
        
        '''crops the image based on the values defined in init.'''
        
        img = Image.open(inpath)
        
        width = self.pic['width']
        height = self.pic['height']
        cropleft = self.pic['cropleft']
        cropright = self.pic['cropright']
        croptop = self.pic['croptop']
        cropbottom = self.pic['cropbottom']
        
        # The box is a 4-tuple defining the left, upper, right, and lower 
        # pixel coordinate
        img_crop = img.crop((cropleft, croptop, width - cropright, height - cropbottom))
        
        img_crop.save(outpath)
        
        if rtrn_image == 1:
            
            return img_crop
            
    def crop_image_parallel(self, imagelist, targetworkspace, n_cpus = 10):
        
        '''method to crop the images in parallel.'''
        
        width = self.pic['width']
        height = self.pic['height']
        cropleft = self.pic['cropleft']
        cropright = self.pic['cropright']
        croptop = self.pic['croptop']
        cropbottom = self.pic['cropbottom']
        
        arguments = [(img, osp.join(targetworkspace, osp.basename(img)), width, 
        height, cropleft, cropright, croptop, cropbottom) for img in imagelist]
        
        if n_cpus > 1:
            pool = multiprocessing.Pool(processes = n_cpus)
            res = pool.map(crop_image_standalone, arguments)
            pool.close()

        else:
            for argument in arguments:
                crop_image_standalone(argument)

    def calc_orig_xcords(self, x):
        
        '''calculates the original photo x coordinates.'''
        
        cropright = self.pic['cropright']

        try: # single number or np array
            x = x + cropright
        except: # in case of list
            x = [c + cropright for c in x]
        
        return x
        
    def calc_orig_ycords(self, y):
        
        '''calculates the original photo y coordinates.'''
        
        cropbottom = self.pic['cropbottom']
        
        try: # single number or np array
            y = y + cropbottom
        except: # in case of list
            y = [c + cropbottom for c in y]
        
        return y
                       
    def photo_to_utm(self, x, y):        
        
        '''converts the photo coordinates into UTM coordinates under 
        the assumption the points are at sea level.'''
    
        theta = self.cam['theta']
        phi = self.cam['phi']
        psi = self.cam['psi']
        sigma = self.cam['sigma']
        H = self.cam['H']
        E = self.cam['E'] 
        N = self.cam['N'] 
        
        # selected points on photograph relative to the center of the image
        xi = x - self.pic['width'] / 2.0
        yi = y - self.pic['height'] / 2.0
          
        X = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
    
        # U and V are vectors that are used to transform a point in a photograph to
        # (Xi,Yi,Zi) - the central projection (direction vector) of the point onto 
        # the plane that is parallel to the photograph but unit distance from the 
        # camera
        #
        # Equation (7) in Krimmel and Rasmussen
        U = np.array([np.sin(theta) * np.cos(psi) - np.cos(theta) * np.sin(phi) * np.sin(psi), 
                      -np.cos(theta) * np.cos(psi) - np.sin(theta) * np.sin(phi) * np.sin(psi), 
                        np.cos(phi) * np.sin(psi)])
        
        V = np.array([-np.sin(theta) * np.sin(psi) - np.cos(theta) * np.sin(phi) * np.cos(psi), 
                      np.cos(theta) * np.sin(psi) - np.sin(theta) * np.sin(phi) * np.cos(psi), 
                        np.cos(phi) * np.cos(psi)])
           
        # Now, we want to intersect the horizontal plane tz=H. 
        # [tx,ty,tz] = H/Zi * (Xi,Yi,Zi)     
        # See Equation (11) in Krimmel and Rasmussen.
        # 
        # Substitute Xi = tx*Zi/H and ty = H*Yi/Zi into Equation (7) and rearrange
        # to solve for (tx, ty).
        
        tx = H * (sigma * X[0] + xi * U[0] + yi * V[0]) / (sigma * X[2] + xi * U[2] + yi * V[2])
        ty = H * (sigma * X[1] + xi * U[1] + yi * V[1]) / (sigma * X[2] + xi * U[2] + yi * V[2])  
        
        tx = tx + E
        ty = ty + N
        
        return [tx, ty]
        
    def utm_to_photo(self, tx, ty):        
        
        '''converts the UTM coordinates into photo coordinates under 
        the assumption the points are at sea level.'''
    
        theta = self.cam['theta']
        phi = self.cam['phi']
        psi = self.cam['psi']
        sigma = self.cam['sigma']
        H = self.cam['H']
        E = self.cam['E'] 
        N = self.cam['N'] 
        
        tx = tx - E
        ty = ty - N
           
        X = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
    
        # U and V are vectors that are used to transform a point in a photograph to
        # (Xi,Yi,Zi) - the central projection (direction vector) of the point onto 
        # the plane that is parallel to the photograph but unit distance from the 
        # camera
        #
        # Equation (7) in Krimmel and Rasmussen
        U = np.array([np.sin(theta) * np.cos(psi) - np.cos(theta) * np.sin(phi) * np.sin(psi), 
                      -np.cos(theta) * np.cos(psi) - np.sin(theta) * np.sin(phi) * np.sin(psi), 
                        np.cos(phi) * np.sin(psi)])
        
        V = np.array([-np.sin(theta) * np.sin(psi) - np.cos(theta) * np.sin(phi) * np.cos(psi), 
                      np.cos(theta) * np.sin(psi) - np.sin(theta) * np.sin(phi) * np.cos(psi), 
                        np.cos(phi) * np.cos(psi)])
                        
                        
        # Substitute Xi = tx*Zi/H and ty = H*Yi/Zi into Equation (7) and rearrange
        # so that
        # [p; q] = A * [xi; yi], where A = [a b; c d].
           
        # a, b, c, d, p, and q are found to equal:
        
        a = U[2] / H * tx - U[0]
        b = V[2] / H * tx - V[0]
        c = U[2] / H * ty - U[1]
        d = V[2] / H * ty - V[1]
        
        p = sigma * (X[0] - X[2] / H * tx)
        q = sigma * (X[1] - X[2] / H * ty)
        
        # Now calculate xi and yi by inverting the matrix. This could be done for
        # each set of points using \ , but when dealing with a lot of points its 
        # easier to write down the formula explicitly. The following two lines are
        # simply A^{-1}*[p;q].
        
        xi = (d * p - b * q) / (a * d - b * c)
        yi = (-c * p + a * q) / (a * d - b * c)
                  
        x = xi + self.pic['width'] / 2.0
        y = yi + self.pic['height'] / 2.0 
        
        return [x, y]
        
    def project_vectorfield_to_utm(self, x, y, u, v): 
    
        '''projects vectorfield from photo coordinates to utm coordinates'''
        
        [x_utm, y_utm] = self.photo_to_utm(x, y)
        
        photocordx_start = x - 0.5 * u 
        photocordx_end = x + 0.5 * u
        
        photocordy_start = y - 0.5 * v
        photocordy_end = y + 0.5 * v  
        
        [x_start_utm, y_start_utm] = self.photo_to_utm(photocordx_start, photocordy_start)
        [x_end_utm, y_end_utm] = self.photo_to_utm(photocordx_end, photocordy_end)
        
        u_utm = x_end_utm - x_start_utm
        v_utm = y_end_utm - y_start_utm
        
        return [x_utm, y_utm, u_utm, v_utm]
        
    def photocords_cropped_to_uncropped(self, x, y):
        
        '''move cropped coordinates to where they would be on the uncropped pic'''
        
        x = x + self.pic['cropleft']  
        y = y + self.pic['croptop'] 
        
        return [x, y]
        
    def photocords_uncropped_to_cropped(self, x, y):
        
        '''move uncropped coordinates to where they would be on the cropped pic'''
        
        x = x - self.pic['cropleft']  
        y = y - self.pic['croptop'] 
        
        return [x, y]
        
    def get_cam(self):
        
        '''provides the camera properties as a dictionary.'''
        return self.cam    