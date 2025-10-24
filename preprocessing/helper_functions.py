#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 14:48:02 2025

@author: ykaitsas
"""

import os

import numpy as np
import xarray as xr
import pandas as pd
import dask.dataframe as dd
import rioxarray as rio

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.metrics import  root_mean_squared_error as rmse 

import matplotlib.pyplot as plt


#%% functions
def preprocess_pressure_levels(
    pressure_levels_directory,
    variables_to_keep=None,
    pressure_levels_to_keep=None,
    drop_unrelated_vars=False,
    keep_part_of_timeline=None,
    convert_geopotential=True, 
    convert_temperature=True,
    drop_vars_pl=None, 
    valid_time_slice=None,
    geopotential_units="km",
    valid_time_chunks=250
):
    """Load ERA5 pressure level datasets, align DEM, convert units."""
    
    pl = xr.open_mfdataset(pressure_levels_directory, 
                           combine="by_coords", compat="no_conflicts",
                           chunks={
                               "valid_time": valid_time_chunks, 
                               "latitude": 256, 
                               "longitude": 256
                               }
                           )
    if drop_unrelated_vars == True:
        for v in drop_vars_pl:
            if v in pl.data_vars:
                pl = pl.drop_vars(v)
    
    
    if "pressure_level" in pl.coords:
        if pressure_levels_to_keep is not None:
            available_levels = set(pl.pressure_level.values)
            levels_to_select = [
                lev for lev in pressure_levels_to_keep if lev in available_levels
                ]
            if levels_to_select:
                pl = pl.sel(pressure_level=levels_to_select)
        
                
    if "z" in pl.data_vars:
        if convert_geopotential:
            original_attrs = pl["z"].attrs.copy()
            pl["z"] = pl.z / 9.80665
            pl["z"].attrs = original_attrs
            pl["z"].attrs["units"] = "m"
            if geopotential_units == "km":
                pl["z"] = pl.z / 1000
                pl["z"].attrs = original_attrs
                pl["z"].attrs["units"] = "km"
    
    if "t" in pl.data_vars:
        if convert_temperature:
            original_attrs = pl["t"].attrs.copy()
            pl["t"] = pl.t - 273.16
            pl["t"].attrs = original_attrs
            pl["t"].attrs["units"] = "C"
    
    if "u" in pl.data_vars and "v" in pl.data_vars:
        pl["V"] = np.sqrt(pl["u"]**2 + pl["v"]**2)
        pl["V"].attrs["short_name"] = "V"   
        pl["V"].attrs["units"] = "m s**-1"
        pl["V"].attrs["long_name"] = "Wind speed"   
        
        pl["phi"] = (180 + (180 / np.pi) * np.arctan2(pl["u"], pl["v"])) % 360
        pl["phi"].attrs["short_name"] = "phi"   
        pl["phi"].attrs["units"] = "degrees"
        pl["phi"].attrs["long_name"] = "Wind direction" 
    
    
    if variables_to_keep is not None:
        available_variables = set(pl.data_vars)
        variables_to_select = set(set(variables_to_keep) & available_variables)
        pl = pl[variables_to_select]
    
    
    if "number" in pl.coords:
        pl = pl.drop_vars("number")

    if "expver" in pl.coords:
        pl = pl.drop_vars("expver")
        
    
    if valid_time_slice is not None:
        start, end = valid_time_slice
        pl = pl.isel(valid_time=slice(start, end))
    elif keep_part_of_timeline is not None:
        pl = pl.isel(valid_time=slice(0, keep_part_of_timeline))

    return pl


def preprocess_single_levels(
    single_levels_directory, 
    dem_file_external=None,
    lsm_file_external=None,
    dem_file_era5=None,
    variables_to_keep=None,
    drop_unrelated_vars=False,
    keep_part_of_timeline=None,
    convert_temperature=True,
    convert_pressure=True,
    drop_vars_sl=None,
    valid_time_slice=None,
    dataset="ERA5",  # other option: "ERA5-Land"
    dem_units="km",  # for extracting dem in km
    valid_time_chunks=250
):
    """Load ERA5 single level datasets, align DEM, convert units."""
    
    sl = xr.open_mfdataset(single_levels_directory, 
                           combine="by_coords", compat="no_conflicts",
                           chunks={
                               "valid_time": valid_time_chunks, 
                               "latitude": 256, 
                               "longitude": 256
                               }
                           )
    
    if drop_unrelated_vars == True:
        for v in drop_vars_sl:
            if v in sl.data_vars:
                sl = sl.drop_vars(v)
    
                
    if "t2m" in sl.data_vars:
        original_attrs = sl["t2m"].attrs.copy()
        if convert_temperature:
            sl["t2m"] = sl.t2m - 273.16
            sl["t2m"].attrs = original_attrs
            sl["t2m"].attrs["units"] = "C" 
            
    if "sst" in sl.data_vars:
        original_attrs = sl["sst"].attrs.copy()
        if convert_temperature:
            sl["sst"] = sl.sst - 273.16        
            sl["sst"].attrs = original_attrs
            sl["sst"].attrs["units"] = "C" 
    
    if "msl" in sl.data_vars:
        original_attrs = sl["msl"].attrs.copy()
        if convert_pressure:
            sl["msl"] = sl.msl / 100
            sl["msl"].attrs = original_attrs
            sl["msl"].attrs["units"] = "hPa" 
            
    if "sp" in sl.data_vars:
        original_attrs = sl["sp"].attrs.copy()
        if convert_pressure:
            sl["sp"] = sl.sp / 100
            sl["sp"].attrs = original_attrs
            sl["sp"].attrs["units"] = "hPa"
    
    if "tp" in sl.data_vars:
        original_attrs = sl["tp"].attrs.copy()
        if convert_pressure:
            sl["tp"] = sl.tp * 1000
            sl["tp"].attrs = original_attrs
            sl["tp"].attrs["units"] = "mm"
            

    # Align external DEM
    if dem_file_external is not None:
        dem = rio.open_rasterio(dem_file_external)
        if dem_units == "km":
            dem = dem/1000
        template = sl.isel(valid_time=0).rio.write_crs("EPSG:4326")
        dem = dem.rio.reproject_match(template, resampling=1).squeeze(drop=True).rename(
            {"x": sl.longitude.name, "y": sl.latitude.name})

        sl = sl.assign_coords(dem=(('latitude', 'longitude'), dem.values))
        
        sl["dem"].attrs["long_name"] = "Elevation derived from external source"
        sl["dem"].attrs["short_name"] = "dem"
        sl["dem"].attrs["units"] = "m"
        if dem_units == "km":
            sl["dem"].attrs["units"] = "km"
        del template
    
    # Align external LSM
    if lsm_file_external is not None:
        lsm = rio.open_rasterio(lsm_file_external)
        template = sl.isel(valid_time=0).rio.write_crs("EPSG:4326")
        lsm = lsm.rio.reproject_match(template, resampling=1).squeeze(drop=True).rename(
            {"x": sl.longitude.name, "y": sl.latitude.name})

        sl = sl.assign_coords(lsm=(('latitude', 'longitude'), lsm.values))
        
        sl["lsm"].attrs["long_name"] = "Land-sea mask derived from external source"
        sl["lsm"].attrs["short_name"] = "lsm"
        sl["lsm"].attrs["units"] = "0 to 1 fraction"
        del template
    
    # Import internal DEM
    if dem_file_era5 is not None:
        if dataset=="ERA5-Land":
            dem_era5 = xr.open_mfdataset(dem_file_era5)
        else:
            dem_era5 = xr.open_dataset(dem_file_era5)
        
        dem_era5 = dem_era5.isel(valid_time=0).squeeze()
        
        if "number" in dem_era5.coords:
            dem_era5 = dem_era5.drop_vars("number")
        if "expver" in dem_era5.coords:
            dem_era5 = dem_era5.drop_vars("expver")
        
        lat_match = len(dem_era5.latitude) == len(sl.latitude)
        lon_match = len(dem_era5.longitude) == len(sl.longitude)        

        if not lat_match or not lon_match:
            dem_era5 = dem_era5.sel(
                latitude=slice(sl.latitude.max().item(), sl.latitude.min().item()),
                longitude=slice(sl.longitude.min().item(), sl.longitude.max().item())
            ) # also add a dem_era5.interp optionally, to match the dems exactly
        
        # RIGHT NOW IT ONLY KEEPS THE DEM PART OF THE FILE, NOT THE SLOPE
        # ORIENTATION OR EHATEVER ELSE IS IN THERE. FIX IT LATER...
        if "z" in dem_era5:
            Re = 6371222.9  # GRIB2 Earth radius in meters (6367470 for GRIB1)
            g0 = 9.80665  # earth gravity constant in m/s^2
            dem_era5["z"] = Re*(dem_era5["z"]/g0)/(Re - (dem_era5["z"]/g0))
            if dem_units=="km":
                dem_era5["z"] = dem_era5["z"]/1000
            
            sl = sl.assign_coords(
                dem_era5=(("latitude", "longitude"), dem_era5["z"].values)
                )
            
            sl["dem_era5"].attrs["long_name"] = "Elevation derived from ERA5 surface geopotential"
            sl["dem_era5"].attrs["short_name"] = "dem_era5"
            sl["dem_era5"].attrs["units"] = "m"
            if dem_units == "km":
                sl["dem_era5"].attrs["units"] = "km"

        if "lsm" in dem_era5:
            sl = sl.assign_coords(
                lsm_era5=(("latitude", "longitude"), dem_era5["lsm"].values)
                )
            
            sl["lsm_era5"].attrs["long_name"] = "Land-sea mask taken from ERA5"
            sl["lsm_era5"].attrs["short_name"] = "lsm_era5"
            sl["lsm_era5"].attrs["units"] = "0 to 1 fraction"
        
    
    if variables_to_keep is not None:
        available_variables = set(sl.data_vars)
        variables_to_select = set(set(variables_to_keep) & available_variables)
        sl = sl[variables_to_select]
           
        
    if "number" in sl.coords:
        sl = sl.drop_vars("number")
        
    if "expver" in sl.coords:
        sl = sl.drop_vars("expver")
        

    if valid_time_slice is not None:
        start, end = valid_time_slice
        sl = sl.isel(valid_time=slice(start, end))
    elif keep_part_of_timeline is not None:
        sl = sl.isel(valid_time=slice(0, keep_part_of_timeline))

    return sl


def widen_pressure_levels(pl):
    """Convert pressure level variables into separate 2D variables."""
    pl_vars = [v for v in pl.data_vars if 'pressure_level' in pl[v].dims]
    pl_wide = xr.Dataset()
    for var in pl_vars:
        for level in pl.pressure_level.values:
            da = pl[var].sel(pressure_level=level, drop=True)
            if "number" in da.coords:
                da = da.drop_vars("number")
            pl_wide[f"{var}{int(level)}"] = da
            del da
    return pl_wide


def merge_pressure_and_single_levels(sl, pl, widened_pressure_levels=True):
    
    if widened_pressure_levels==True:
        ds = xr.merge([sl, widen_pressure_levels(pl)])
    else:
        ds = xr.merge([sl, pl])

    return ds


def add_time_features(df, time_col='valid_time', drop_original=True):
    """Add year, month, day features from a time column."""
    if time_col not in df.columns:
        return df
    
    is_dask = isinstance(df, dd.DataFrame)
    
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        if is_dask:
            df[time_col] = dd.to_datetime(df[time_col], errors='coerce')
        else:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    
    df['year'] = df[time_col].dt.year
    df['month'] = df[time_col].dt.month
    df['dayofyear'] = df[time_col].dt.dayofyear
    
    if drop_original:
        df = df.drop(columns=[time_col], errors='ignore')
    
    return df


def stack_and_chunk(pl_wide, sl, create_index=True,
                    dynamic_chunk_size=False, n_target_partitions=20, 
                    chunk_size_static=200_000):
    """Merge datasets, stack for ML, and chunk."""
    ds = xr.merge([pl_wide, sl], compat="no_conflicts").\
        stack(sample=("valid_time", "latitude", "longitude"), 
              create_index=create_index)

    if dynamic_chunk_size:
        chunk_size = int(np.ceil(ds.sizes["sample"] / n_target_partitions))
    else:
        chunk_size = chunk_size_static

    ds = ds.chunk({"sample": chunk_size})
    
    sample_index = ds.indexes["sample"]
    
    return ds, sample_index


def convert_to_ddf(ds_stacked):
    """Convert stacked Dataset to Dask DataFrame."""
    ddf = ds_stacked.to_dask_dataframe()
    ddf = add_time_features(ddf)
    return ddf

def convert_columns_to_float32(ddf, columns):
    """Convert specific columns in a Dask DataFrame to int16 if they exist."""
    mapping = {col: "float32" for col in columns if col in ddf.columns}
    return ddf.astype(mapping)


#%% lets play
chunk_number = 500_000
pressure_levels = [850, 700, 500]

time_frequency = "monthly"
dataset = "ERA5"

# Paths
pressure_levels_directory = f"/home/ykaitsas/Documents/CDS-data-and-scripts/ERA5-pressure-levels-{time_frequency}-Greece/*.nc"
single_levels_directory = f"/home/ykaitsas/Documents/CDS-data-and-scripts/ERA5-single-levels-{time_frequency}-Greece/*.nc"
dem_file_external = f"/home/ykaitsas/Documents/downscaling-era5/dem/processed-tifs/DEM-{dataset}-0.25deg.tif"
lsm_file_external = f"/home/ykaitsas/Documents/downscaling-era5/dem/processed-tifs/WBM-{dataset}-0.25deg.tif"
dem_era5_file="/home/ykaitsas/Documents/downscaling-era5/dem/era5-static-variables.nc"


sl = preprocess_single_levels(
    single_levels_directory, 
    dem_file_external=dem_file_external,
    lsm_file_external=lsm_file_external,
    dem_file_era5=dem_era5_file,
    variables_to_keep=["t2m","msl","sp","tp"],
    drop_unrelated_vars=False,
    keep_part_of_timeline=None,
    valid_time_slice=None,
    valid_time_chunks=100
    )

pl = preprocess_pressure_levels(
    pressure_levels_directory, 
    variables_to_keep=["z","t","q","u","v","V","phi"],
    pressure_levels_to_keep=[850,700,500,400],
    keep_part_of_timeline=None,
    valid_time_slice=None,
    valid_time_chunks=100
    )

ds=merge_pressure_and_single_levels(sl, pl, widened_pressure_levels=True)


if time_frequency != "monthly":
    dsm = xr.merge([
        ds['tp'].resample(valid_time='MS').sum(),
        ds.drop_vars('tp').resample(valid_time='MS').mean()
        ]) # or "ME" for end of month valid_time values
    dsm.attrs = {}


#%% construct HR baseline dataset
target_resolution = 0.05
demHD = rio.open_rasterio(
    f"/home/ykaitsas/Documents/downscaling-era5/dem/processed-tifs/DEM-{dataset}-{target_resolution}deg.tif"
    ).squeeze("band", drop=True) / 1000  # conver to km
lsmHD = rio.open_rasterio(
    f"/home/ykaitsas/Documents/downscaling-era5/dem/processed-tifs/WBM-{dataset}-{target_resolution}deg.tif"
    ).squeeze("band", drop=True)


demHD = demHD.assign_coords(
    y=np.round(demHD.y.values, 3),
    x=np.round(demHD.x.values, 3)
)
demHD = demHD.rename({"y": "latitude", "x": "longitude"})

lsmHD = lsmHD.assign_coords(
    y=np.round(lsmHD.y.values, 3),
    x=np.round(lsmHD.x.values, 3)
)
lsmHD = lsmHD.rename({"y": "latitude", "x": "longitude"})


coord_vars = ["dem", "lsm", "dem_era5", "lsm_era5"]
nearest_interpolation_vars = ["t2m"] + coord_vars

dsHD = ds.reset_coords(coord_vars)
dsHD = xr.Dataset({
    var: dsHD[var].interp(
        latitude=demHD.latitude,
        longitude=demHD.longitude,
        method="nearest" if var in nearest_interpolation_vars else "linear",
        kwargs={"fill_value": "extrapolate"}
    )
    for var in dsHD.data_vars
}).set_coords(coord_vars)

dsHD = dsHD.assign_coords(demHD=(('latitude', 'longitude'), demHD.values))
dsHD = dsHD.assign_coords(lsmHD=(('latitude', 'longitude'), lsmHD.values))


data_folder = '/home/ykaitsas/Documents/downscaling-era5/data'
os.makedirs(data_folder, exist_ok=True)
#ds.to_netcdf(os.path.join(data_folder, "dsLD.nc"))
#dsHD.to_netcdf(os.path.join(data_folder, "dsHD.nc"))

'''
#%%
df = ds.to_dataframe()

df = df[df.index.get_level_values("valid_time").year<=2024]
df["latitude"] = df.index.get_level_values("latitude")
df["longitude"] = df.index.get_level_values("longitude")
df["month"] = df.index.get_level_values("valid_time").month
'''
'''
import seaborn as sns
for month in df.month.unique():
    corr = df[df.month==month].corr()
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        corr,
        annot=True,            # shows correlation values
        cmap="coolwarm",       # color palette
        vmin=-1, vmax=1,       # correlation range
        square=True,           # make cells square
        fmt=".2f",             # format numbers
        cbar_kws={"shrink": 0.8}
    )
    plt.title(f"Correlation Heatmap - Monthly - {month}")
    plt.tight_layout()
    plt.savefig(f"correlation-monthly-{month}.png", dpi=500)
    plt.show()
    '''


