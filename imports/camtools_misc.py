#!/usr/bin/env python

import numpy as np
import platform
import os.path as osp
import pandas as pd
import datetime as dt

from PIL import Image
import shutil

import glob
#------------------------------------------------------------------------------
# Functions
       
def get_focal_length(path):
    '''focal length of camera during image aquisition'''
    return Image.open(path)._getexif()[37386]

def get_image_width(path):
    '''width of image in pixels'''
    return Image.open(path)._getexif()[40962]
    
def get_image_height(path):
    '''height of image in pixels'''
    return Image.open(path)._getexif()[40963]
    
def get_image_x_res_ppi(path):
    '''x image resolution in ppi'''
    res = Image.open(path)._getexif()[41486]
    return res[0]/res[1]
    
def get_image_y_res_ppi(path):
    '''y image resolution in ppi'''
    res = Image.open(path)._getexif()[41487]
    return res[0]/res[1] 
    
def get_sensor_size(pixels, ppi):
    return (pixels/float(ppi)) * 25.4
     
def scale_params(inp, subtractor, factor):
    outp = (inp - subtractor) * factor
    return outp 
      
def times_at_tide_elevations(date, tideelev = -2.13, range_m = 0.9):
    
    '''returns the times when tides are at a certain elevation (tideelev)
    - 2.13 m is the tide on 20160410-2009, when we derived the shoreline from 
    a WorldView 3 image'''
    
    # determine whether Linux or Windows
    oper_sys = platform.system()
        
    if oper_sys == 'Windows':
        basepath = 'C:\Users\Ch.Kienholz\Dropbox\LeConte_ck\Scripts\Python_Tools_Modelbuilder\Photogrammetry\Camera\data'
    else:
        basepath = '/home/ckienholz/Dropbox/LeConte_ck/Scripts/Python_Tools_Modelbuilder/Photogrammetry/Camera/data'
        
    tides = pd.read_pickle(basepath +  '/tide_prediction_2018.pickle')
    
    year = int(date[0:4]) 
    month = int(date[4:6]) 
    day = int(date[6:8])
    
    # if there is no picture matching, search the ten next days
    for counter in range(0, 1):
        
        # pictures are typically taken between 17:00 and 1:00 the next day
        imagetime1 = dt.datetime(year, month, day, 14, 30) + dt.timedelta(days=counter)
        imagetime2 = dt.datetime(year, month, day + 1, 2, 30) + dt.timedelta(days=counter)
        
        times_matching = tides.loc[(tides['date'] >= imagetime1) & (tides['date'] <= imagetime2) 
        & (tides['depth_tide_ellipsoid'] >= (tideelev - range_m)) & (tides['depth_tide_ellipsoid'] <= (tideelev + range_m))]
        
        times_matching['difference'] = abs(tideelev - times_matching['depth_tide_ellipsoid'])
        
        counter += 1
        
        if len(times_matching) > 0:
            break
        
    return times_matching
     
times_at_tide_elevations('20180908', range_m = .05)
     
def collect_calibration_images():
    
    '''function to copy selected calibration images into a specified folder'''
    
    df = pd.read_excel('/home/ckienholz/Dropbox/LeConte_ck/Scripts/Python_Tools_Modelbuilder/Photogrammetry/Calibration_dates.xlsx')
    targetworkspace = '/home/ckienholz/Dropbox/LeConte_ck/Scripts/Python_Tools_Modelbuilder/Photogrammetry/data/cam'
    
    images = df['image']
    
    imagefolders = df['imagefolder']
    
    cameras = df['camera']
    
    for cam, imagefolder, image in zip(cameras, imagefolders, images):
        
        shutil.copy(imagefolder + '/' + image, targetworkspace + str(cam) + '/' + image)
        
def add_raster_to_mxd(mxd, workspace, raster, visibility):
    
    '''function to add raster datasets to ArcGIS .mxd'''
    
    import arcpy
    
    df = arcpy.mapping.ListDataFrames(mxd, 'Layers')[0]

    layername = raster[:raster.rfind('.')] + '.lyr'  
    layername = layername.split('\\')[-1]
    
    arcpy.MakeRasterLayer_management(raster, layername)
    arcpy.SaveToLayerFile_management(layername, workspace + '\\' + layername, 'ABSOLUTE')
    arcpy.Delete_management(layername)

    addLayer = arcpy.mapping.Layer(workspace + '\\' + layername)
    arcpy.mapping.AddLayer(df, addLayer, 'TOP') #
    
    layers = arcpy.mapping.ListLayers(mxd, layername, df)
    layers[0].visible = visibility


def add_shape_to_mxd(mxd, workspace, shapefile, templatelayer, visibility):
    
    '''function to add shapefiles to ArcGIS .mxd'''
    
    import arcpy
    
    df = arcpy.mapping.ListDataFrames(mxd, 'Layers')[0]
    
    layername = shapefile[:shapefile.rfind('.')] + '.lyr'  
    layername = layername.split('\\')[-1]

    # Create layer file and save it with an absolute path
    arcpy.MakeFeatureLayer_management(shapefile, layername)
    arcpy.SaveToLayerFile_management(layername, workspace + '\\' + layername, 'ABSOLUTE')
    arcpy.Delete_management(layername)

    arcpy.ApplySymbologyFromLayer_management(workspace + '\\' + layername, templatelayer)

    addLayer = arcpy.mapping.Layer(workspace + '\\' + layername)
    arcpy.mapping.AddLayer(df, addLayer, 'TOP')
    
    layers = arcpy.mapping.ListLayers(mxd, layername, df)
    layers[0].visible = visibility
    
def add_pics_to_mxd():
    '''adds the oblique pics to the .mxd so that the shoreline can be digitized'''
    
    import arcpy
    
    arcpy.env.overwriteOutput = 1
    
    mxd = r'C:\Users\Ch.Kienholz\Dropbox\LeConte_ck\Scripts\Python_Tools_Modelbuilder\Photogrammetry\data\Calibration.mxd'
    
    folders = glob.glob('C:\Users\Ch.Kienholz\Dropbox\LeConte_ck\Scripts\Python_Tools_Modelbuilder\Photogrammetry\data\cam*')
    
    mxd_obj = arcpy.mapping.MapDocument(mxd)
    
    for folder in folders:
        
        pictures = glob.glob(folder+'/*.jpg')
        
        for picture in pictures:
            
            path = picture[0:picture.rfind('\\')]
            pic = picture[picture.rfind('\\')+1:]
            visibility = 0
            
            add_raster_to_mxd(mxd_obj, path, picture, visibility)
            
            outshapefile = path + '\\' + pic.split('.')[0][0:-2]+ '.shp'
            
            shapefile = arcpy.Copy_management(r'C:\Users\Ch.Kienholz\Dropbox\LeConte_ck\Scripts\Python_Tools_Modelbuilder\Photogrammetry\data\shoreline_template.shp', outshapefile)
            
            shapetemplate = r'C:\Users\Ch.Kienholz\Dropbox\LeConte_ck\Scripts\Python_Tools_Modelbuilder\Photogrammetry\data\shoreline_template.lyr'            
            
            add_shape_to_mxd(mxd_obj, path, str(shapefile), shapetemplate, visibility)
            
            mxd_obj.save()
        
    del mxd_obj