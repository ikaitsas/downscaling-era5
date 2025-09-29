# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 15:02:51 2025

@author: yiann
"""

import os
import xarray as xr
import pandas as pd
import rioxarray as rio
from pandas.api.types import is_datetime64_any_dtype
from numpy import ceil as numpy_ceil
import dask.dataframe as dd
import matplotlib.pyplot as plt

from dask.diagnostics import ResourceProfiler, Profiler, CacheProfiler, ProgressBar
from dask.distributed import Client


#client = Client() 
#client.restart()
#print(client.dashboard_link)


pressure_levels_directory = "/home/ykaitsas/Documents/CDS-data-and-scripts/ERA5-pressure-levels-daily-Greece/*.nc"
single_levels_directory = "/home/ykaitsas/Documents/CDS-data-and-scripts/ERA5-single-levels-daily-Greece/*.nc"

dem_file = "dem-era5-0.25deg.tif"
era5_dem_storage = os.path.join("/home/ykaitsas/Documents/downscaling-era5/dem/dem-files-for-downscaling", dem_file)

keep_subset_of_days = False
days_to_keep = 1200

convert_geopotential = True
convert_temperature = True

drop_year_variable = True

chunk_size_static = 200_000
dynamic_chunk_size = False
n_target_partitions = 20

path_to_saving_dataframes = "dataframe-dump"
os.makedirs(path_to_saving_dataframes, exist_ok=True)


#%% functions
def add_time_features(df, time_col='valid_time', drop_original=True):
    #  works for both regular and dask dataframes
    
    if time_col not in df.columns:
        return df
    
    is_dask = isinstance(df, dd.DataFrame)
    
    if not is_datetime64_any_dtype(df[time_col]):
        if is_dask:
            df[time_col] = dd.to_datetime(df[time_col], errors='coerce')
        else:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    
    # Extract features
    df['year'] = df[time_col].dt.year
    df['month'] = df[time_col].dt.month
    df['day'] = df[time_col].dt.dayofyear
    
    if drop_original:
        df = df.drop(columns=[time_col], errors='ignore')
    
    return df

#%% preprocessing
pl = xr.open_mfdataset(pressure_levels_directory, 
                       combine="by_coords",
                       compat="no_conflicts"
                       )
sl = xr.open_mfdataset(single_levels_directory, 
                       combine="by_coords",
                       compat="no_conflicts"
                       )

dem = rio.open_rasterio(era5_dem_storage)

# drop unecessary  vars
for v in ["tp", "sst"]:
    if v in sl.data_vars:
        sl = sl.drop_vars(v)
for v in ["w", "q", "r"]:
    if v in pl.data_vars:
        pl = pl.drop_vars(v)

# convert units
if convert_geopotential:
    pl["z"] = pl.z / 9.80665
if convert_temperature:
    pl["t"] = pl.t - 273.16


# match era5 and dem grids - usually minor tweaks get done
template = pl.isel(valid_time=0).rio.write_crs("EPSG:4326")
print("DEM shape:", dem.shape)
print("ERA5 shape:", template.latitude.shape[0], template.longitude.shape[0])
dem = dem.rio.reproject_match(template, resampling=1).squeeze(drop=True).\
    rename({"x":pl.longitude.name,"y":pl.latitude.name})
del template

pl = pl.assign_coords(dem=(('latitude','longitude'), dem.values))
sl = sl.assign_coords(dem=(('latitude','longitude'), dem.values))


if keep_subset_of_days == True:
    pl = pl.isel(valid_time=slice(0, days_to_keep))
    sl = sl.isel(valid_time=slice(0, days_to_keep))



#%% dask implementation
# widen pressure levels - to match sl's 3 dimensions
pl_vars = [v for v in pl.data_vars if 'pressure_level' in pl[v].dims]

pl_wide = xr.Dataset()
for var in pl_vars:
    for level in pl.pressure_level.values:
        # Extract the slice at this pressure level
        da = pl[var].sel(pressure_level=level, drop=True).\
            reset_coords("number", drop=True)
        
        # Create a new variable in the Dataset
        # Name: variable + level, e.g., "z850", "t500"
        pl_wide[f"{var}{int(level)}"] = da
        del da

pl_wide_vars = list(pl_wide.data_vars)


# keep sl as is, we care about ground conditions
sl_vars = [v for v in sl.data_vars if 'pressure_level' not in sl[v].dims]
sl = sl.reset_coords("number", drop=True)


# merge datasets - extract datafram for tabula rmachine learning training
ds = xr.merge([pl_wide, sl], compat="no_conflicts")
ds_stacked = ds.stack(sample=("valid_time", "latitude", "longitude"))

if dynamic_chunk_size == True:
    chunk_size_dynamic = int(
        numpy_ceil(
            ds_stacked.sizes["sample"] / n_target_partitions
            ))
    ds_stacked = ds_stacked.chunk({"sample": chunk_size_dynamic})
else:
    ds_stacked = ds_stacked.chunk({"sample": chunk_size_static})

ddf = ds_stacked.to_dask_dataframe()
ddf = add_time_features(ddf) 

# extract regular dataframe for prototyping
if keep_subset_of_days == True:
    df = ds.to_dataframe().reset_index()
    df = add_time_features(df)


print("saving the dataframes...")


ddf.to_parquet(
    os.path.join(path_to_saving_dataframes, "ddf"), 
    engine="pyarrow",
    write_index=False,  
    overwrite=True
    )
    
    
    
    


#%% EDA
'''
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
'''


