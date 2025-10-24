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

band = "WBM"
dataset = "ERA5"
target_resolution = 0.01 # in degrees
extent = [18, 43, 31, 34]  #W-N-E-S
crop_first = True


input_tif = f"{band}-merged-1arc-corrected.tif"


if dataset == "ERA5":
    field_native_resolution = 0.25
elif dataset == "ERA5-Land":
    field_native_resolution = 0.1
else:
    field_native_resolution = 0

if (dataset=="ERA5") or (dataset=="ERA5-Land"):
    cropped_tif = f"{band}-cropped-{dataset}.tif"
    aggregated_tif = f"{band}-{dataset}-{target_resolution}deg.tif"
else:
    cropped_tif = f"{band}-cropped.tif"
    aggregated_tif = f"{band}-{target_resolution}deg.tif"

tif_storage = "processed-tifs"
aggregated_tif_path = os.path.join(tif_storage, aggregated_tif)


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
        "-a_nodata", "255",
        "-co", f"COMPRESS={compression}",   
        "-co", "TILED=YES",         
        "-co", f"BLOCKXSIZE={block_size}",    
        "-co", f"BLOCKYSIZE={block_size}", 
        "-co", "BIGTIFF=IF_NEEDED",
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
                       block_size=512,  # 256 also a good option
                       nodata_value=-9999,  # optional nodata handling
                       output_type="Float32" 
                       ):
    # see GDAL documentation for available methods of aggregation
    # popular ones include average, bilinear, lanczos
    
    if overwrite_existing == True:
        if os.path.exists(aggregated_dem):
            os.remove(aggregated_dem)
    
    cmd_aggregate = [
        "gdalwarp",
        "-tr", f"{target_resolution}", f"{target_resolution}",
        "-r",f"{method}",
        "-t_srs", f"{t_srs}",
        "-ot", f"{output_type}",
        "-co", f"COMPRESS={compression}",   
        "-co", "TILED=YES",         
        "-co", f"BLOCKXSIZE={block_size}",    
        "-co", f"BLOCKYSIZE={block_size}", 
        "-co", "BIGTIFF=IF_NEEDED",
        "-dstnodata", str(nodata_value),
        input_dem,
        aggregated_dem
        ]
    
    subprocess.run(cmd_aggregate, check=True)
    
    return {"path": aggregated_dem, "target resolution": target_resolution}


#%% lessgoo
input_tif = os.path.join(tif_storage, input_tif)
cropped_tif = os.path.join(tif_storage, cropped_tif)

if crop_first == True:
    print("Cropping...")
    crop_dem_gdal(
        input_tif, cropped_tif, 
        extent=extent, field_native_resolution=field_native_resolution
        )
print("Aggregating...")
aggregate_dem_gdal(
    cropped_tif, aggregated_tif_path, 
    target_resolution=target_resolution
    )

'''
import matplotlib.pyplot as plt
import numpy as np
array = rioxarray.open_rasterio(aggregated_tif_path).squeeze("band", drop=True)
plt.imshow(array)
plt.show()
print(len(np.unique(array)))
'''
