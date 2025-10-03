# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:14:30 2025

@author: yiann

ERA5 has its own elevation inside, coded through the geopotential
convert geopotential to geopotential height: h = z/9.80665  (g0)
then do the elevation = Re*h/(Re-h)  Re=6367470 meters (radius of earth sphere)

ERA5 also has some other useful land cover and vegetation type, land-sea mask,
lake cover static fields. Same for ERA5-Land. Maybe I could use these..
"""


import os
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



load_ddf_to_memory = False

ddf_path = "dataframe-dump-client"
ddf_name_convention = "part.*.parquet"
sample_name = "sample_index_for_dataset_reconstruction.parquet"

pressure_levels_directory = "C:\\Users\\yiann\\Documents\\ERA5-ERA5-Land\\ERA5-pressure-levels-daily-Greece\\*.nc"
single_levels_directory = "C:\\Users\\yiann\\Documents\\ERA5-ERA5-Land\\ERA5-single-levels-daily-Greece\\*.nc"
dem_file = "C:\\Users\\yiann\\Documents\\downscaling-era5\\dem\\dem-files-for-downscaling\\dem-era5-0.25deg.tif"


ddf = dd.read_parquet(os.path.join(ddf_path, ddf_name_convention))

sample_index = pd.read_parquet(os.path.join(ddf_path, sample_name))
sample_index = pd.MultiIndex.from_frame(sample_index)

if load_ddf_to_memory == True:
    df = ddf.compute()

pl, sl = preproc.load_and_preprocess(
    pressure_levels_directory, 
    single_levels_directory, 
    dem_file,
    keep_part_of_timeline=None,
    valid_time_slice=None
)


lsm_era5 = preproc.extract_era5_kand_sea_mask(single_levels_directory)

'''
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
'''
'''
#%% Plot "ERA5 Land-Sea Mask" derived from valid SST tiles
lon2d, lat2d = np.meshgrid(sl.longitude.values, sl.latitude.values)

fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.gridlines(draw_labels=True)

# Plot mask (use pcolormesh for grid data)
c = ax.pcolormesh(lon2d, lat2d, lsm_era5, cmap="coolwarm", alpha=0.6)

# Colorbar
plt.colorbar(c, ax=ax, orientation="vertical", label="1 = Land, 0 = Sea")

plt.title("ERA5 Land-Sea Mask")
plt.savefig("land_sea_mask_era5.png", dpi=500)
plt.show()
'''

#%% Correlation visualization through df
corr = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, 
            fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Matrix Heatmap", fontsize=16)
plt.savefig("correlation_heatmap.png", dpi=500)
plt.show()

