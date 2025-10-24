#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 14:34:40 2025

@author: ykaitsas
"""

import os
import glob
import numpy as np
from osgeo import gdal
gdal.UseExceptions()

from tqdm import tqdm
import matplotlib.pyplot as plt



wbm_directory = "tiles-GLO30-WBM"
dem_directory = "tiles-GLO30-DEM"
wbm_files = sorted(glob.glob(os.path.join(wbm_directory, "*.tif")))
dem_files = sorted(glob.glob(os.path.join(dem_directory, "*.tif")))

wbm_directory_corrected = "tiles-GLO30-WBM-corrected"
os.makedirs(wbm_directory_corrected, exist_ok=True)


for dem_path, wbm_path in tqdm(zip(dem_files,wbm_files), 
                               total=len(dem_files),
                               desc="Correcting WBM issues..."):
    
    wbm_path_corrected = os.path.join(wbm_directory_corrected, os.path.basename(wbm_path))
    
    try:
        src_dem = gdal.Open(dem_path, gdal.GA_ReadOnly)
        src_dem_band = src_dem.GetRasterBand(1)
        dem_array = src_dem_band.ReadAsArray()

        src_wbm = gdal.Open(wbm_path, gdal.GA_ReadOnly)
        src_wbm_band = src_wbm.GetRasterBand(1)
        wbm_array = src_wbm_band.ReadAsArray()

        
        zero_count = np.sum(wbm_array==0)
        one_count = np.sum(wbm_array==1)
        one_count_inner = np.sum(wbm_array[1:-1,1:-1])
        dem_mean = np.mean(dem_array)
        
        wbm_array_corrected = wbm_array.copy()
        
        # handle sea tiles
        if (one_count_inner <= 0) and (np.abs(dem_mean<1)):
            wbm_array_corrected[wbm_array==1]=0
        
        # handle normal tiles
        else:
            wbm_array_corrected[wbm_array==1]=0
            wbm_array_corrected[wbm_array==2]=0
            wbm_array_corrected[wbm_array==3]=0
            wbm_array_corrected[wbm_array==0]=1
        

        # Create output raster with same metadata as input
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.CreateCopy(wbm_path_corrected, src_wbm, 1)
        out_band = out_ds.GetRasterBand(1)
        
        # Write the modified data
        out_band.WriteArray(wbm_array_corrected)
        out_band.FlushCache()
        
        # Preserve nodata if present
        nodata = src_wbm_band.GetNoDataValue()
        if nodata is not None:
            out_band.SetNoDataValue(nodata)
        
        out_ds.FlushCache()
        out_ds = None
        src_ds = None

        
    except Exception as e:
        print(f"Error processing {wbm_path}: {str(e)}")





