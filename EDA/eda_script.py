# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 15:02:51 2025

@author: yiann
"""

import numpy as np
import xarray as xr
import rioxarray as rio
from pandas import merge
import matplotlib.pyplot as plt

from scipy.stats import skew

storage_directory = "C:\\Users\\yiann\\Documents\\ERA5-ERA5-Land\\*.nc"
era5_dem = "C:\\Users\\yiann\\Documents\\downscaling-era5\\dem\\dem-files-for-downscaling\\dem-era5-0.25deg.tif"


era5 = xr.open_mfdataset(storage_directory, combine="by_coords")

dem = rio.open_rasterio(era5_dem)

#%% preprocessing
# drop precipitation related vars
if "tp" in era5.data_vars.keys():
    era5 = era5.drop_vars("tp")
if "sst" in era5.data_vars.keys():
    era5 = era5.drop_vars("sst")
    
if "w" in era5.data_vars.keys():
    era5 = era5.drop_vars("w")
if "q" in era5.data_vars.keys():
    era5 = era5.drop_vars("q")
if "r" in era5.data_vars.keys():
    era5 = era5.drop_vars("r")


# match era5 and dem grids - usually minor tweaks get done
template = era5.isel(valid_time=0).rio.write_crs("EPSG:4326")
print("DEM shape:", dem.shape)
print("ERA5 shape:", template.latitude.shape[0], template.longitude.shape[0])
dem = dem.rio.reproject_match(template, resampling=1)
dem = dem.squeeze(drop=True) 
dem = dem.rename({"y":era5.latitude.name, "x":era5.longitude.name})

era5 = era5.assign_coords(dem=(('latitude','longitude'), dem.values))


#%% stack the data into a dataframe
era = era5.isel(valid_time=slice(0, 1200))

plev_vars = [v for v in era.data_vars if 'pressure_level' in era[v].dims]
surf_vars = [v for v in era.data_vars if 'pressure_level' not in era[v].dims]
dsp = era[plev_vars]
dss = era[surf_vars]

# Convert pressure levels to dataframe
dfp = dsp.to_dataframe().drop(columns=["number", "dem"]).unstack('pressure_level')
dfp.columns = [f"{var}{int(lev)}" for var, lev in dfp.columns]
dfp = dfp.reset_index()

# Convert single levels to dataframe
df = dss.to_dataframe().drop(columns=["number"])#.unstack('pressure_level')
df = df.reset_index()

df = merge(df, dfp, how="inner")
del dfp, dsp, dss


#%%
corr = df.corr()

# Plot heatmap
fig, ax = plt.subplots()
cax = ax.matshow(np.abs(corr), cmap="coolwarm")
plt.colorbar(cax)

# Set ticks
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.columns)

plt.title("Correlation Matrix", pad=20)
plt.show()