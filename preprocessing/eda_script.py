# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:14:30 2025

@author: yiann
"""


import os
from tqdm import tqdm

import dask.dataframe as dd
import pandas as pd
import numpy as np
import xarray as xr

import preprocessing_tabular_functions as preproc

from scipy.stats import linregress

import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature


load_ddf_to_memory = True

ddf_path = "dataframe-dump-client"
ddf_name_convention = "part.*.parquet"
sample_index_name = "sample_index_for_dataset_reconstruction.parquet"

pressure_levels_directory = "/home/ykaitsas/Documents/CDS-data-and-scripts/ERA5-pressure-levels-daily-Greece/*.nc"
single_levels_directory = "/home/ykaitsas/Documents/CDS-data-and-scripts/ERA5-single-levels-daily-Greece/*.nc"
dem_file = "/home/ykaitsas/Documents/downscaling-era5/dem/dem-files-for-downscaling/dem-era5-0.25deg.tif"
dem_era5_file="/home/ykaitsas/Documents/downscaling-era5/dem/era5-static-variables.nc"

images_path = "images"
os.makedirs(images_path, exist_ok=True)

ddf = dd.read_parquet(os.path.join(ddf_path, ddf_name_convention))

sample_index = pd.read_parquet(os.path.join(ddf_path, sample_index_name))
sample_index = pd.MultiIndex.from_frame(sample_index)

if load_ddf_to_memory == True:
    df = ddf.compute()


sl = preproc.preprocess_single_levels(
    single_levels_directory, 
    dem_file_external=dem_file,
    dem_file_era5=dem_era5_file,
    drop_unrelated_vars=True,
    keep_part_of_timeline=None,
    valid_time_slice=None
    )

pl = preproc.preprocess_pressure_levels(
    pressure_levels_directory, 
    dem_file_external=dem_file,
    dem_file_era5=dem_era5_file,
    drop_unrelated_vars=True,
    keep_part_of_timeline=None,
    valid_time_slice=None
    )


#%% Elevation dependence visualization
# need to exclude the sea tiles and focus on land
# will implement a new function to spot sea tiles
# this is "easy" for ERA5, as there is sst, that counts as sea pixels??
'''
for l,lat in enumerate(sl.latitude.values):
    print(l,lat)
    for d in range(0,30):
        plt.scatter(sl.t2m.values[d,l,:], sl.dem_era5.values[l,:], s=2)
        plt.title(
            f"Date: {np.datetime_as_string(sl.valid_time.values[d], unit='D')},\
            Latitude: {lat}"
                )
        lapse_rate = linregress(sl.dem_era5.values[l,:], 
                                sl.t2m.values[d,l,:]
                                ).slope
        plt.legend([f"Empirical Lapse Rate: {lapse_rate:.2f} C/km"])
        plt.ylabel("ERA5 Grid Elevation [km]")
        plt.xlabel("2m Air Temperature [C]")
        plt.grid()
        plt.show()
'''

#corr = df.corr()

#%%
'''
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, 
            fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Matrix Heatmap", fontsize=16)
plt.savefig(os.path.join(images_path, "correlation_heatmap.png"), dpi=500)
plt.show()
'''
for month in tqdm(df.month.unique()):
    corr_m = df[df.month==month].drop(columns=["month"]).corr()
    
    plt.figure(figsize=(18, 14))
    sns.heatmap(corr_m, annot=True, 
                fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title(f"Correlation Matrix Heatmap - {month}", fontsize=16)
    plt.savefig(os.path.join(images_path, f"correlation_heatmap_{month}.png"), 
                dpi=500)
    plt.show()
    




