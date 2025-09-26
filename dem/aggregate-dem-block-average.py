#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 11:08:23 2025

@author: ykaitsas
"""

import os
import rasterio
import subprocess
import numpy as np
import xarray as xr
#from osgeo import gdal
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

from scipy.ndimage import zoom
from sklearn.impute import KNNImputer
from skimage.measure import block_reduce


target_resolution = 0.03  # in degrees
field_native_resolution = 0.25  # the era5/era5-land fields resolution
input_dem = "cropped-era5.tif"
extent = [19, 42, 30, 34.5]  #W-N-E-S
export_nc_to_device = False


#%% functions
def crop_dem_gdal(input_dem, cropped_dem, extent, field_native_resolution,
                       reference_center=True,
                       expand_to_all_directions=True,
                       compression="LZW",  # lossless
                       block_size=512  # 256 also a good option
                       ):
    if reference_center == True:
        # subwindow to extract - in projected coordinates
        if expand_to_all_directions == True:
            subwindow = [
                extent[0]-field_native_resolution/2, 
                extent[1]+field_native_resolution/2, 
                extent[2]+field_native_resolution/2, 
                extent[3]-field_native_resolution/2
                ]
        else:
            subwindow = [
                extent[0]-field_native_resolution/2, 
                extent[1]+field_native_resolution/2, 
                extent[2]-field_native_resolution/2, 
                extent[3]+field_native_resolution/2
                ]
    else:
        subwindow = extent
         

    cmd_crop = [
        "gdal_translate", 
        "-projwin",
        f"{subwindow[0]}", 
        f"{subwindow[1]}", 
        f"{subwindow[2]}", 
        f"{subwindow[3]}",
        "-projwin_srs", 
        "EPSG:4326",
        "-co", f"COMPRESS={compression}",   
        "-co", "TILED=YES",         
        "-co", f"BLOCKXSIZE={block_size}",    
        "-co", f"BLOCKYSIZE={block_size}", 
        input_dem,
        cropped_dem
        ]
    
    subprocess.run(cmd_crop, check=True)
    
    return {"path": cropped_dem, "extent": subwindow}


def aggregate_dem_gdal(input_dem, aggregated_dem, target_resolution,
                       overwrite=True,
                       method="average",
                       t_srs="EPSG:4326",
                       compression="LZW",  # lossless
                       block_size=512  # 256 also a good option
                       ):
    # see GDAL documentation for available methods of aggregation
    # popular ones include average, bilinear, lanczos
    '''
    # DOULEVEI KANONIKA ALLA EGW MALAKIZOMAI...
    # TO PREDICTOR=2 ISWS THELEI AFAIRESH
    gdalwarp -tr 0.01 0.01 -r average -t_srs EPSG:4326 \
      -co COMPRESS=LZW -co PREDICTOR=2 -co TILED=YES \
      -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 \
      cropped.tif cropped_001.tif
    '''
    if overwrite == True:
        if os.path.exists(aggregated_dem):
            os.remove(aggregated_dem)
    
    cmd_aggregate = [
        "gdalwarp",
        "-tr",
        f"{target_resolution}", 
        f"{target_resolution}",
        "-r",
        f"{method}",
        "-t_srs", 
        f"{t_srs}",
        "-co", f"COMPRESS={compression}",   
        "-co", "TILED=YES",         
        "-co", f"BLOCKXSIZE={block_size}",    
        "-co", f"BLOCKYSIZE={block_size}", 
        input_dem,
        aggregated_dem
        ]
    
    subprocess.run(cmd_aggregate, check=True)
    
    return {"path": aggregated_dem, "target resolution": target_resolution}


def pad_to_block_size(array, block_shape, pad_value=0):
    pad_y = (-array.shape[0]) % block_shape
    pad_x = (-array.shape[1]) % block_shape
    
    pad_y = (-array.shape[0]) % block_shape
    pad_x = (-array.shape[1]) % block_shape
    
    padded_array = np.pad(
        array,
        ((0, pad_y), (0, pad_x)),  # only bottom and right
        mode='constant',
        constant_values=pad_value
    )
    return padded_array, pad_y, pad_x

'''
def dem_block_aggregate(input_dem, target_resolution, 
                        pad_value=0, raster_dtype="int16"):
    
    with rasterio.open(input_dem) as ds:
        array = ds.read(1).astype(raster_dtype)
        gt = ds.transform
        rows, cols = ds.height, ds.width
        x_res = gt[0]
        y_res = gt[4]
        x_min = gt[2]
        y_max = gt[5]
        
    scale_factor = int(target_resolution/x_res)  # this is the block size
    new_rows, new_cols = int(rows//scale_factor), int(cols//scale_factor)
    
    array, pad_y, pad_x = pad_to_block_size(array, scale_factor)
    new_rows_padded = array.shape[0] // scale_factor
    new_cols_padded = array.shape[1] // scale_factor
    
    reshaped = array.reshape(new_rows_padded, scale_factor, new_cols_padded, scale_factor)
    array_aggregated = np.nanmean(reshaped, axis=(1, 3)).round()[:new_rows,:new_cols]
    
    return array_aggregated
'''

#%% perform aggregation
with rasterio.open(input_dem) as ds:
    array = ds.read(1).astype("int16")
    gt = ds.transform

    rows, cols = ds.height, ds.width
    x_res = gt[0]
    y_res = gt[4]
    x_min = gt[2]
    y_max = gt[5]
    
    
    latitudes = np.arange(y_max, y_max+rows*y_res, y_res)
    longitudes = np.arange(x_min, x_min+cols*x_res, x_res)


print(f'Native resolution: {x_res*3600:.1f} arcseconds.')
print(f'Target resolution: {target_resolution} degrees.')
#add error raising if target_res%x_res!=0
scale_factor = int(target_resolution/x_res)  # this is the block size
rows_aggregated, cols_aggregated = int(rows//scale_factor), int(cols//scale_factor)
print(f'Rows x columns: {rows_aggregated}x{cols_aggregated}.')

if array.shape[0]%scale_factor==0 and array.shape[1]%scale_factor==0:
    print("Perfectly divisable scaling factor...")
    print("Proceeding without padding...")
    padding = False
    array_agg = np.nanmean(
        array.reshape(rows_aggregated, scale_factor, 
                      cols_aggregated, scale_factor), 
        axis=(1, 3)
        ).round()
else:
    print("Non-integer upscaling factor...")
    print("Padding the edges symetrically...")
    padding = True
    array, pad_y, pad_x = pad_to_block_size(array, scale_factor)
    rows_aggregated_padded = array.shape[0] // scale_factor
    cols_aggregated_padded = array.shape[1] // scale_factor
    array_agg = np.nanmean(
        array.reshape(rows_aggregated_padded, scale_factor, 
                      cols_aggregated_padded, scale_factor), 
        axis=(1, 3)
        ).round()[:rows_aggregated,:cols_aggregated]


y_res_agg = -target_resolution
x_res_agg = target_resolution


if padding == True:
    #  AYTA DE NTA EFTAKSA AKOMA
    latitudes_agg = np.arange(
        y_max, y_max+rows_aggregated_padded*y_res_agg, y_res_agg
        ) + y_res/2 + y_res_agg/2
    latitudes_agg = latitudes_agg[:array_agg.shape[0]]
    longitudes_agg = np.arange(
        x_min, x_min+cols_aggregated_padded*x_res_agg, x_res_agg
        ) + x_res/2 + x_res_agg/2
    longitudes_agg = longitudes_agg[:array_agg.shape[1]]
else:
    latitudes_agg = np.arange(
        y_max, y_max+rows_aggregated*y_res_agg, y_res_agg
        ) + y_res/2 + y_res_agg/2 
    latitudes_agg = latitudes_agg[:array_agg.shape[0]]
    longitudes_agg = np.arange(
        x_min, x_min+cols_aggregated*x_res_agg, x_res_agg
        ) + x_res_agg/2+ x_res/2 
    longitudes_agg = longitudes_agg[:array_agg.shape[1]]


#%% export to netCDF4
ds = xr.Dataset(
    {
        "dem": (("lat", "lon"), array_agg)
    },
    coords={
        "latitude": latitudes_agg,
        "longitude": longitudes_agg
    }
)

# Add attributes (optional but good practice)
ds["dem"].attrs["units"] = "meters"
ds["dem"].attrs["long_name"] = "Elevation"
ds.attrs["data_source"] = "GLO30-GEE"
ds["latitude"].attrs["units"] = "degrees_north"
ds["longitude"].attrs["units"] = "degrees_east"

# Save to NetCDF
if export_nc_to_device == True:
    ds.to_netcdf(f"dem-aggregated-{target_resolution}deg.nc")


#%%  this command for gdal aggregation in command line:
gdal_dem = f"gdal-{target_resolution}deg.tif"

aggregate_dem_gdal(input_dem, gdal_dem, target_resolution=target_resolution)

with rasterio.open(gdal_dem) as gds:
    array_gdal = gds.read(1).astype("int16")
    

