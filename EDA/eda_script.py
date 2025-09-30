# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:14:30 2025

@author: yiann
"""


import os
import dask.dataframe as dd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar

from preprocessing_tabular_functions import load_and_preprocess

from scipy.stats import linregress


ddf_path = "dataframe-dump-client"
ddf_name_convention = "part.*.parquet"
sample_name = "sample_index_for_dataset_reconstruction.parquet"

pressure_levels_directory = "/home/ykaitsas/Documents/CDS-data-and-scripts/ERA5-pressure-levels-daily-Greece/*.nc"
single_levels_directory = "/home/ykaitsas/Documents/CDS-data-and-scripts/ERA5-single-levels-daily-Greece/*.nc"
dem_file = "/home/ykaitsas/Documents/downscaling-era5/dem/dem-files-for-downscaling/dem-era5-0.25deg.tif"


ddf = dd.read_parquet(os.path.join(ddf_path, ddf_name_convention))

sample_index = pd.read_parquet(os.path.join(ddf_path, sample_name))
sample_index = pd.MultiIndex.from_frame(sample_index)

df = ddf.compute()

pl, sl = load_and_preprocess(
    pressure_levels_directory, 
    single_levels_directory, 
    dem_file,
    keep_part_of_timeline=None,
    valid_time_slice=None
)

#%% Elevation dependence visualization
# need to exclude the sea tiles and focus on land
# will implement a new function to spot sea tiles
# this is "easy" for ERA5, as there is sst, that counts as sea pixels??
for l,lat in enumerate(sl.latitude.values):
    print(l,lat)
    for d in range(0,30):
        plt.scatter(sl.t2m.values[d,l,:], sl.dem.values[l,:]/1000, s=2)
        plt.title(
            f"Date: {np.datetime_as_string(sl.valid_time.values[d], unit='D')},\
            Latitude: {lat}"
                )
        lapse_rate = linregress(sl.dem.values[l,:]/1000, 
                                sl.t2m.values[d,l,:]
                                ).slope
        plt.legend([f"Empirical Lapse Rate: {lapse_rate:.2f} C/km"])
        plt.ylabel("ERA5 Grid Elevation [km]")
        plt.xlabel("2m Air Temperature [C]")
        plt.grid()
        plt.show()




