#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 11:08:23 2025

@author: ykaitsas
"""

import os
import rasterio
import rioxarray
import subprocess


target_resolution = 0.025  # in degrees
field_native_resolution = 0.25  # the era5/era5-land fields resolution
input_dem = "cropped-era5-land.tif"
extent = [19, 42, 30, 34.5]  #W-N-E-S
export_nc_to_device = False

gdal_dem = f"dem-era5-land-{target_resolution}deg.tif"


dem_storage = "dem-files-for-downscaling"
os.makedirs(dem_storage, exist_ok=True)
gdal_dem_path = os.path.join(dem_storage, gdal_dem)


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
                       overwrite_existing=True,
                       method="average",
                       t_srs="EPSG:4326",
                       compression="LZW",  # lossless
                       block_size=512  # 256 also a good option
                       ):
    # see GDAL documentation for available methods of aggregation
    # popular ones include average, bilinear, lanczos
    
    if overwrite_existing == True:
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


#%% lessgoo
aggregate_dem_gdal(input_dem, gdal_dem_path, target_resolution=target_resolution)

with rasterio.open(gdal_dem_path) as gds:
    array_gdal = gds.read(1)#.astype("int16")

