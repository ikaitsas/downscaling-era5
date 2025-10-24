#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 12:20:02 2025

@author: ykaitsas

KATI PAEI LATHOS
"""

import os
import glob
from osgeo import gdal
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


directory = "tiles-GLO30-WBMmm"
tif_files = glob.glob(os.path.join(directory, "*.tif"))

for tif_path in tqdm(tif_files):
    
    try:
        ds = gdal.Open(tif_path, gdal.GA_Update)
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()

        height, width = data.shape
        total_pixels = width * height

        # Count zeros, ones, threes
        zero_count = np.sum(data==0)
        one_count = np.sum(data==1)
        two_count = np.sum(data==2)
        three_count = np.sum(data==3)
        
        one_count_inner = np.sum(data[1:-1,1:-1]==1)
        
        percentage_nonzero = ( 1 - (zero_count/total_pixels) )*100
        
        if (two_count==0) or (three_count==0): 
                if one_count_inner == 0:
                    data[data==0] = 1
                    band.WriteArray(data)
                    band.FlushCache()
                    print(f"    -> converted zeros to 1s")
        
        
    except Exception as e:
        print(f"Error processing {tif_path}: {str(e)}")
