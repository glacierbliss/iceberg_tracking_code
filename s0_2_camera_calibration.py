#%%

#!/usr/bin/env python

import os.path as osp

import numpy as np

import pandas as pd
import geopandas as gpd
import datetime as dt

import shapefile
from pathlib import Path

#%% functions for shapefile reading and writing

def x_y_from_shapefile(shp, tuples = 0):
    
    '''extracts the x and y coordinates from points digitized on the oblique photograph'''
    
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
            list_of_tuples = [(int(shape.points[0][0]), int(shape.points[0][1] * -1)) 
            for shape in shapes]
            
        else:
            list_of_tuples = [[(int(shape.points[c][0]), int(shape.points[c][1] * -1)) 
            for c in range(0, len(shape.points))] for shape in shapes][0]
        
        return list_of_tuples
    
def x_y_z_from_shapefile(shp):
    
    '''extracts the x, y and z coordinates from points digitized on UTM data. 
    Assuming z is at sea level = 0 m.
    '''
    
    shapes = shapefile.Reader(shp).shapes()
    
    xcoords = np.array([shape.points[0][0] for shape in shapes])
    ycoords = np.array([shape.points[0][1] for shape in shapes])
    zcoords = np.array([0 for shape in shapes])
    
    return [xcoords, ycoords, zcoords]
              
def get_wkt_prj(epsg_code):
    import urllib
    wkt = urllib.request.urlopen('http://spatialreference.org/ref/epsg/{0}/prettywkt/'.format(epsg_code))
    remove_spaces = wkt.read().decode().replace(" ","")
    output = remove_spaces.replace("\n", "")
    return output
       
def create_shapefile(basepath, df, E, N, x, y, imwidth, imheight, sensor_width, shapename):
    
    '''reprojects photo coordinates to utm for quality checks'''
    
    # points on photo relative to the center of the image
    xi = x - imwidth / 2.0 
    yi = y - imheight / 2.0
    
    # create new shapefile
    w = shapefile.Writer(shapename,shapefile.POINT)
    w.field('iteration', 'N')

    # obtain the absolute best combinations
    best_theta = df['theta']
    best_phi = df['phi']
    best_psi = df['psi']
    best_sigma = df['sigma']
    best_H = df['H']
    
    # create and update camera dictionary            
    cam = {}
    cam['theta'] = np.radians(best_theta) # azimuth
    cam['phi'] = np.radians(best_phi) # tilt
    cam['psi'] = np.radians(best_psi) # rotation
    cam['sigma'] = (imwidth / sensor_width) * best_sigma # scaling
    
    # project to UTM
    [X_modeled, Y_modeled] = photo_to_utm(xi, yi, E, N, best_H, cam)
    
    # write and save shapefile
    for x_mod, y_mod in zip(X_modeled, Y_modeled):
        w.point(x_mod, y_mod)
        w.record(0)
           
    # w.save(shapename)  
    
    # prj = open(shapename[:-3] + 'prj', "w")
    prj = open(shapename.with_suffix('.prj'), "w") #pathlib way of doing things.
    epsg = get_wkt_prj(32608) #TODO: make this an input parameter or get it from satllite image fjord_outline.shp?
    prj.write(epsg)
    prj.close() 
    
#%% functions for camera calibration

def photo_to_utm(xi, yi, E, N, H, cam):
    
    theta = cam['theta']
    phi = cam['phi']
    psi = cam['psi']
    sigma = cam['sigma']
      
    X = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

    # U and V are vectors that are used to transform a point in a photograph to
    # (Xi,Yi,Zi) - the central projection (direction vector) of the point onto 
    # the plane that is parallel to the photograph but unit distance from the 
    # camera
    #
    # Equation (7) in Krimmel and Rasmussen
    U = np.array([np.sin(theta) * np.cos(psi) - np.cos(theta) * np.sin(phi) * np.sin(psi), 
    -np.cos(theta) * np.cos(psi) - np.sin(theta) * np.sin(phi) * np.sin(psi), np.cos(phi) * np.sin(psi)])
    
    V = np.array([-np.sin(theta) * np.sin(psi) - np.cos(theta) * np.sin(phi) * np.cos(psi), 
    np.cos(theta) * np.sin(psi) - np.sin(theta) * np.sin(phi) * np.cos(psi), np.cos(phi) * np.cos(psi)])
    
       
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
    
   
def xyz_to_mulambda(theta, phi, xvec, yvec, zvec): 
    
    # Equations 2 to 4 in Krimmel and Rasmussen
    # xvec, yvec and zvec are np.arrays
    veclen = xvec.size
    
    lambdahatvec = np.zeros((veclen, 1))
    muhatvec = np.zeros((veclen, 1))
    
    for n in range(0, veclen):
        
        x = xvec[n]
        y = yvec[n]
        z = zvec[n]
    
        # Combine equations 2 and 3
        P = (1 / (np.cos(theta) * np.cos(phi) * x + np.sin(theta) * np.cos(phi) * y + np.sin(phi) * z)) * np.array([x, y, z])

        # Equation 4
        lambdahat = P[0] * np.sin(theta) - P[1] * np.cos(theta)
        muhat = P[2] * np.cos(phi) - (P[0] * np.cos(theta) + P[1] * np.sin(theta)) * np.sin(phi)
        
        lambdahatvec[n] = lambdahat
        muhatvec[n] = muhat    
    
    return [lambdahatvec, muhatvec] 
    
def mulambda_to_psisigma(lambdaphotovec, muphotovec, lambdahatvec, muhatvec):

    # Equations 6, 8 and 9 in Krimmel and Rasmussen
    
    veclen = lambdaphotovec.size
    
    sum1 = np.zeros((1))
    sum2 = np.zeros((1))
    sum3 = np.zeros((1))
    
    for n in range(0, veclen):
        
        lambdahat = lambdahatvec[n]
        muhat = muhatvec[n]
        lambdaphoto = lambdaphotovec[n]
        muphoto = muphotovec[n]
       
        sum1 = sum1 + (lambdahat ** 2 + muhat ** 2)
        sum2 = sum2 + (lambdahat * lambdaphoto + muhat * muphoto)
        sum3 = sum3 + (lambdaphoto * muhat - lambdahat * muphoto)
    
    # Equation 8
    T = (1 / sum1) * np.array([[sum2], [sum3]])
    
    sigma = (T[0] ** 2 + T[1] ** 2) ** 0.5
    psi1 = float(np.arcsin(T[1] / sigma))
    
    # Equation 6
    lambdavec = np.zeros((veclen, 1))
    muvec = np.zeros((veclen, 1))
    
    sum4 = np.zeros((1))
    
    for n in range(0, veclen):
        
        Q = sigma * np.dot(np.array([[np.cos(psi1), np.sin(psi1)], 
        [-np.sin(psi1), np.cos(psi1)]]), 
        np.array([[float(lambdahatvec[n])], [float(muhatvec[n])]]))
        
        lambdavec[n] = Q[0]
        muvec[n] = Q[1]
        
        sum4 = sum4 + ((lambdavec[n] - lambdaphotovec[n]) ** 2 + (muvec[n] - muphotovec[n]) ** 2) 
    
    # Equation 9 (gives the mismatch on the 2D photograph in pixels)
    mismatch = (sum4 / (n + 1)) ** 0.5

    return [float(sigma), psi1, float(mismatch)]
    
def closest_node(node, nodes):
    '''https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
    compares one point (node) to a list of points (nodes) and in that list
    identifies the closest one and returns that distance'''

    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.min(dist_2 ** 0.5)
    
def optimizefun_calibration(params, x, y, imwidth, imheight, sensor_width, E, N, waterline_points):
    
    '''projects waterline points from oblique photographs to UTM and calculates 
    distance between these points and reference points derived from satellite image
    '''
    
    parvals = params.valuesdict()
    theta = parvals['theta']
    phi = parvals['phi']
    psi = parvals['psi']
    sigma = parvals['sigma']
    H = parvals['H']
    
    # convert angle ranges to radians
    theta = np.radians(theta) # azimuth
    phi = np.radians(phi) # tilt
    psi = np.radians(psi) # rotation
    sigma = (imwidth / sensor_width) * sigma # scaling
    
    # points on photo relative to the image center   
    xi = x - imwidth / 2.0 
    yi = y - imheight / 2.0

    # create and update the camera dictionary        
    cam = {} 
    cam['theta'] = theta # azimuth
    cam['phi'] = phi # tilt
    cam['psi'] = psi # rotation
    cam['sigma'] = sigma # scaling
           
    [X_modeled, Y_modeled] = photo_to_utm(xi, yi, E, N, H, cam)
    
    # loop through projected points, find closest point on the utm waterline
    # distances = [closest_node(point, waterline_points) for point in zip(X_modeled, Y_modeled)] 
    distances = [closest_node(np.array(point), waterline_points) for point in zip(X_modeled, Y_modeled)]      
    return distances
                  
#------------------------------------------------------------------------------
    
def run_calibration(workspace,calibration_filename,fjord_outline,tide_file=None):
    
    '''conducts the camera calibration based on waterline points from 
    oblique photographs and satellite imagery'''
    
    import lmfit
    
    # read input file into dataframe (contains input parameters for calibration)   
    # df = pd.read_excel(osp.join(workspace, calibration_filename))
    df = pd.read_excel(Path(workspace, calibration_filename))
    
    # add new, empty columns to dataframe
    df = df.reindex(columns = df.columns.tolist() + ['H', 'theta', 'phi', 'psi', 
                    'sigma', 'rmse', 'tide']) 
        
    # iterate over dataframe, each row corresponds to one calibration
    for index, row in df.iterrows():
            
        print('---------------------------------------------------------------')
        print ('running {}: {}'.format(row['camera'], row['image'][0:8]))
               
        cam = row['camera']
        
        # width of image chip
        sensor_width = row['sensor_width']
    
        # read the camera's UTM coordinates
        E = row['easting']
        N = row['northing']
        H = row['elevation']
        antenna_height = row['antenna_height'] # GPS antenna height above camera
        
        # read calibration ranges for theta (azimuth), phi (tilt)  
        # psi (rotation), and sigma (scaling) 
        # ranges are based on measurements in the field
        theta_min_max = [row['theta_min'], row['theta_max']]
        phi_min_max = [row['phi_min'], row['phi_max']]
        psi_min_max = [row['psi_min'], row['psi_max']]
        sigma_min_max = [row['sigma_min'], row['sigma_max']]
        
        imwidth = row['image_width']
        imheight = row['image_height']
        
        # read time and convert to time string
        time_string = str(row['image']).split('.')[0]
        imagetime = dt.datetime.strptime(time_string, '%Y%m%d-%H%M%S')
        
        # set seconds to zero for tide prediction        
        imagetime = imagetime.replace(second = 0)
        
        # load the tides (contains tide value for each minute) 
        if tide_file is not None:
            # tides = pd.read_pickle(workspace + '/data/' + tide_file)
            tides = pd.read_pickle(Path(workspace,tide_file)) #workspace already includes /data/

            # find the current tide value
            current_tide = tides.loc[tides['date'] == imagetime]['depth_tide_ellipsoid'].iloc[0]
            current_tide=float(current_tide)
        
            # account for GPS antenna height and tide (negative tide increases H)
            H = H - antenna_height - current_tide 

        else:
            tides = None  # or some default value
                
        
        # load waterline coordinates from oblique photograph
        # points were previously digitized along the waterline and saved to a shapefile
        # shoreline_points = osp.join(workspace, cam, time_string + '.shp')
        shoreline_points = Path(workspace, cam, time_string + '_' + cam + '.shp') #AKB added "'_' + cam + " to match instruction document
        [x, y] = x_y_from_shapefile(shoreline_points)
        # plt.scatter(x,y)
        # plt.show()
        
        # load utm waterline coordinates, previously digitized from satellite image 
        # select .npz file that matches the region covered by the cam         
        # fjord_npz = convert_shp_to_npz(workspace + '/' + fjord_outline)
        fjord_npz = convert_shp_to_npz(Path(workspace,fjord_outline))
        #AKB NOTE: gives warning, but can proceed... Warning 3: Cannot find header.dxf (GDAL_DATA is not defined)

        fjord = np.load(fjord_npz)      

        # convert fjord coordinates into tuples
        waterline_points = np.array(list(zip(fjord['x'], fjord['y'])))
        
        # set up parameters for the least-squares minimization
        params = lmfit.Parameters()
        
        params.add('theta', value = np.mean(theta_min_max), 
                   min = theta_min_max[0], max = theta_min_max[1])
        params.add('phi', value = np.mean(phi_min_max), 
                   min = phi_min_max[0], max = phi_min_max[1])
        params.add('psi', value = np.mean(psi_min_max), 
                   min = psi_min_max[0], max = psi_min_max[1])
        params.add('sigma', value = np.mean(sigma_min_max), 
                   min = sigma_min_max[0], max = sigma_min_max[1])
        params.add('H', value = H, min = H - 1.5, max = H + 1.5, vary = False) # keep fixed

        minimizer_object = lmfit.Minimizer(optimizefun_calibration, params, 
                                           fcn_args=(x, y, imwidth, imheight, 
                                           sensor_width, E, N, waterline_points))
        
        # run minimization                                 
        output = minimizer_object.minimize()
        
        # make sure script runs for lmfit versions older and newer than version 0.9
        # older than 0.9 
        if output == 1:
            fitted_parameters = minimizer_object.params
            rmse = round((np.mean(minimizer_object.residual ** 2)) ** 0.5, 2)
        
        # newer than 0.9
        else:
            fitted_parameters = output.params
            rmse = round((np.mean(output.residual ** 2)) ** 0.5, 2)
            
        print('RMSE: ' + str(rmse))
                
        # select the best parameter value for each parameters
        best_theta = round(fitted_parameters['theta'].value, 5)
        best_phi = round(fitted_parameters['phi'].value, 5)
        best_psi = round(fitted_parameters['psi'].value, 5)
        best_sigma = round(fitted_parameters['sigma'].value, 5)
        best_H = round(fitted_parameters['H'].value, 2)
        
        print('best theta: {}'.format(best_theta))
        print('best phi: {}'.format(best_phi))
        print('best psi: {}'.format(best_psi))
        print('best sigma: {}'.format(best_sigma))
        print('best H: {}'.format(best_H))
                
        # update dataframe with fitted parameter values
        # df.set_value(index, 'H', best_H)
        # df.set_value(index, 'theta', best_theta)
        # df.set_value(index, 'phi', best_phi)
        # df.set_value(index, 'psi', best_psi)
        # df.set_value(index, 'sigma', best_sigma)
        # df.set_value(index, 'rmse', rmse)
        # df.set_value(index, 'tide', round(current_tide, 2))
        df.at[index, 'H'] = best_H
        df.at[index, 'theta'] = best_theta
        df.at[index, 'phi'] = best_phi
        df.at[index, 'psi'] = best_psi
        df.at[index, 'sigma'] = best_sigma
        df.at[index, 'rmse'] = rmse
        df.at[index, 'output_step'] = index +1

        if tide_file is not None:
            df.at[index, 'tide'] = round(current_tide, 2)
        #------------------------------------------ 
        # create shapefile to visually check the best projection in a GIS, but only if RMSE is less than 1000
        # if rmse < 1000:
        if rmse < 1e100:
            # shapename = osp.join(workspace, cam, f'shoreline_{index+1}_{time_string}_utm.shp')
            shapename = Path(workspace, cam, f'shoreline_{cam}_{time_string}_utm.shp') #substitute cam for index+1 in case non-consecutive
            print(shapename)
           
            create_shapefile(workspace, df.iloc[index, :], E, N, x, y, imwidth, 
                            imheight, sensor_width, shapename)
        else:
            print('RMSE too big to create shapefile')
    
    # delete columns in the dataframe    
    del_fields = ['image', 'imagefolder', 'sigma_min', 'sigma_max', 'theta_min', 
                  'theta_max', 'phi_min', 'phi_max', 'psi_min', 'psi_max']
    
    for field in del_fields:
        del df[field]
        
    # save to final excel parameter file
    # df.to_excel(workspace + '/' + output_file, index = 0)
    df.to_excel(Path(workspace,output_file), index = 0)

def convert_shp_to_npz(shp_file):
    '''create numpy zip file for shape'''
    # Read the shapefile
    gdf = gpd.read_file(shp_file)

    # Get the exterior coordinates of the Polygon
    exterior_coords = gdf['geometry'][0].exterior.coords.xy

    # Create 'x' and 'y' arrays from the exterior coordinates
    x = np.array(exterior_coords[0])
    y = np.array(exterior_coords[1])

    # Save the 'x' and 'y' arrays to a .npz file
    base_name = os.path.splitext(shp_file)[0]
    npz_file = base_name + '.npz'

    #AKB NOTE: Better to just recreate npz each time? This way requires user to know to delete .npz if shapefile is updated.
    # Check if the file already exists
    if not os.path.exists(npz_file):
        # If the file doesn't exist, save the 'x' and 'y' arrays to a .npz file
        np.savez(npz_file, x=x, y=y)
    else:
        print(f"File {npz_file} already exists. Not creating a new one.")

    # Return the filename of the .npz file
    return npz_file


if __name__ == '__main__':
    
    # workspace = '/hdd3/opensource/iceberg_tracking/data'
    workspace = Path('G:/Glacier/GD_ICEH_iceHabitat/data') #Path would take care of trailing slash
    fjord_outline = 'fjord_outline.shp'
    calibration_filename = 'calibration_input_2019.xlsx'
    output_file = 'parameter_file_2019.xlsx'
    calibration_filename = 'calibration_combinations_all.xlsx'
    output_file = 'parameter_file_all.xlsx'
   
    run_calibration(workspace,calibration_filename,fjord_outline,tide_file=None)
# %%
