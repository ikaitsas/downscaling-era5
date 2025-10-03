#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 15:55:29 2025

@author: ykaitsas

dask.delayed needs better implementation:
    https://docs.dask.org/en/stable/delayed-best-practices.html
"""

import os
import xarray as xr
import pandas as pd
import rioxarray as rio
from numpy import ceil as numpy_ceil
import dask.dataframe as dd



#%% preprocessing functions

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


def load_and_preprocess(
    pressure_levels_directory, 
    single_levels_directory, 
    dem_file,
    drop_unrelated_vars=False,
    keep_part_of_timeline=None,
    convert_geopotential=True, 
    convert_temperature=True,
    drop_vars_pl=None, 
    drop_vars_sl=None,
    valid_time_slice=None
):
    """Load ERA5 pressure and single level datasets, align DEM, convert units."""
    
    drop_vars_pl = drop_vars_pl or ["w", "q", "r"]
    drop_vars_sl = drop_vars_sl or ["tp", "sst"]

    pl = xr.open_mfdataset(pressure_levels_directory, 
                           combine="by_coords", compat="no_conflicts",
                           chunks={"valid_time": 250, "latitude": 256, "longitude": 256}
                           )
    sl = xr.open_mfdataset(single_levels_directory, 
                           combine="by_coords", compat="no_conflicts",
                           chunks={"valid_time": 250, "latitude": 256, "longitude": 256}
                           )
    dem = rio.open_rasterio(dem_file)

    for v in drop_vars_sl:
        if v in sl.data_vars:
            sl = sl.drop_vars(v)
    for v in drop_vars_pl:
        if v in pl.data_vars:
            pl = pl.drop_vars(v)

    if convert_geopotential:
        pl["z"] = pl.z / 9.80665
    if convert_temperature:
        pl["t"] = pl.t - 273.16
        sl["t2m"] = sl.t2m - 273.16

    # Align DEM
    template = pl.isel(valid_time=0).rio.write_crs("EPSG:4326")
    dem = dem.rio.reproject_match(template, resampling=1).squeeze(drop=True).rename(
        {"x": pl.longitude.name, "y": pl.latitude.name})
    pl = pl.assign_coords(dem=(('latitude', 'longitude'), dem.values))
    sl = sl.assign_coords(dem=(('latitude', 'longitude'), dem.values))
    del template
    
    if "number" in sl.coords:
        sl = sl.drop_vars("number")
    if "number" in pl.coords:
        pl = pl.drop_vars("number")

    if valid_time_slice is not None:
        start, end = valid_time_slice
        pl = pl.isel(valid_time=slice(start, end))
        sl = sl.isel(valid_time=slice(start, end))
    elif keep_part_of_timeline is not None:
        pl = pl.isel(valid_time=slice(0, keep_part_of_timeline))
        sl = sl.isel(valid_time=slice(0, keep_part_of_timeline))

    return pl, sl


def preprocess_pressure_levels(
    pressure_levels_directory, 
    dem_file_external=None,
    dem_file_era5=None,
    drop_unrelated_vars=False,
    keep_part_of_timeline=None,
    convert_geopotential=True, 
    convert_temperature=True,
    drop_vars_pl=["w"], 
    valid_time_slice=None,
    dem_scale="km"
):
    """Load ERA5 pressure level datasets, align DEM, convert units."""
    
    #drop_vars_pl = drop_vars_pl or ["w", "q", "r"]

    pl = xr.open_mfdataset(pressure_levels_directory, 
                           combine="by_coords", compat="no_conflicts",
                           chunks={"valid_time": 250, "latitude": 256, "longitude": 256}
                           )
    if drop_unrelated_vars == True:
        for v in drop_vars_pl:
            if v in pl.data_vars:
                pl = pl.drop_vars(v)

    if convert_geopotential:
        pl["z"] = pl.z / 9.80665
    if convert_temperature:
        pl["t"] = pl.t - 273.16
        

    # Align external DEM
    if dem_file_external is not None:
        dem = rio.open_rasterio(dem_file_external)
        if dem_scale=="km":
            dem = dem/1000
        template = pl.isel(valid_time=0).rio.write_crs("EPSG:4326")
        dem = dem.rio.reproject_match(template, resampling=1).squeeze(drop=True).rename(
            {"x": pl.longitude.name, "y": pl.latitude.name})
        pl = pl.assign_coords(dem=(('latitude', 'longitude'), dem.values))
        del template
    
    # Import internal DEM
    if dem_file_era5 is not None:
        dem_era5 = xr.open_dataset(dem_file_era5)
        
        dem_era5 = dem_era5.isel(valid_time=0).squeeze()
        
        if "number" in dem_era5.coords:
            dem_era5 = dem_era5.drop_vars("number")
        if "expver" in dem_era5.coords:
            dem_era5 = dem_era5.drop_vars("expver")
            
        if "z" in dem_era5:
            Re = 6371222.9  # GRIB2 Earth radius in meters (6367470 for GRIB1)
            g0 = 9.80665  # earth gravity constant in m/s^2
            dem_era5["z"] = Re*(dem_era5["z"]/g0)/(Re - (dem_era5["z"]/g0))
            if dem_scale=="km":
                dem_era5["z"] = dem_era5["z"]/1000
            
            pl = pl.assign_coords(
                dem_era5=(("latitude", "longitude"), dem_era5["z"].values)
                )
        
        if "lsm" in dem_era5:
            pl = pl.assign_coords(
                lsm_era5=(("latitude", "longitude"), dem_era5["lsm"].values)
                )
    

    if "number" in pl.coords:
        pl = pl.drop_vars("number")

    if valid_time_slice is not None:
        start, end = valid_time_slice
        pl = pl.isel(valid_time=slice(start, end))
    elif keep_part_of_timeline is not None:
        pl = pl.isel(valid_time=slice(0, keep_part_of_timeline))

    return pl


def preprocess_single_levels(
    single_levels_directory, 
    dem_file_external=None,
    dem_file_era5=None,
    drop_unrelated_vars=False,
    keep_part_of_timeline=None,
    convert_temperature=True,
    drop_vars_sl=["tp", "sst"],
    valid_time_slice=None,
    dataset="ERA5",  # other option: "ERA5-Land"
    dem_scale="km"  # for extracting dem in km
):
    """Load ERA5 single level datasets, align DEM, convert units."""
    

    #drop_vars_sl = drop_vars_sl or ["tp", "sst"]


    sl = xr.open_mfdataset(single_levels_directory, 
                           combine="by_coords", compat="no_conflicts",
                           chunks={"valid_time": 250, "latitude": 256, "longitude": 256}
                           )
    if drop_unrelated_vars == True:
        for v in drop_vars_sl:
            if v in sl.data_vars:
                sl = sl.drop_vars(v)

    if convert_temperature:
        sl["t2m"] = sl.t2m - 273.16
        

    # Align external DEM
    if dem_file_external is not None:
        dem = rio.open_rasterio(dem_file_external)
        if dem_scale == "km":
            dem = dem/1000
        template = sl.isel(valid_time=0).rio.write_crs("EPSG:4326")
        dem = dem.rio.reproject_match(template, resampling=1).squeeze(drop=True).rename(
            {"x": sl.longitude.name, "y": sl.latitude.name})

        sl = sl.assign_coords(dem=(('latitude', 'longitude'), dem.values))
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
        
        # RIGHT NOW IT ONLY KEEPS THE DEM PART OF THE FILE, NOT THE SLOPE
        # ORIENTATION OR EHATEVER ELSE IS IN THERE. FIX IT LATER...
        if "z" in dem_era5:
            Re = 6371222.9  # GRIB2 Earth radius in meters (6367470 for GRIB1)
            g0 = 9.80665  # earth gravity constant in m/s^2
            dem_era5["z"] = Re*(dem_era5["z"]/g0)/(Re - (dem_era5["z"]/g0))
            if dem_scale=="km":
                dem_era5["z"] = dem_era5["z"]/1000
            
            sl = sl.assign_coords(
                dem_era5=(("latitude", "longitude"), dem_era5["z"].values)
                )
        
        if "lsm" in dem_era5:
            sl = sl.assign_coords(
                lsm_era5=(("latitude", "longitude"), dem_era5["lsm"].values)
                )
        
        
    if "number" in sl.coords:
        sl = sl.drop_vars("number")

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


def extract_era5_land_sea_mask(single_levels_directory, key="sst"):
    
    lsm = xr.open_mfdataset(single_levels_directory, 
                           combine="by_coords", compat="no_conflicts",
                           chunks={"valid_time": 250, "latitude": 256, "longitude": 256}
                           )[key].values[0,:,:]
    
    return lsm


#%% tabularization functions

def stack_and_chunk(pl_wide, sl, create_index=True,
                    dynamic_chunk_size=False, n_target_partitions=20, 
                    chunk_size_static=200_000):
    """Merge datasets, stack for ML, and chunk."""
    ds = xr.merge([pl_wide, sl], compat="no_conflicts").\
        stack(sample=("valid_time", "latitude", "longitude"), 
              create_index=create_index)

    if dynamic_chunk_size:
        chunk_size = int(numpy_ceil(ds.sizes["sample"] / n_target_partitions))
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

def convert_columns_to_int32(ddf, columns):
    """Convert specific columns in a Dask DataFrame to int16 if they exist."""
    mapping = {col: "int32" for col in columns if col in ddf.columns}
    return ddf.astype(mapping)

def convert_columns_to_float32(ddf, columns):
    """Convert specific columns in a Dask DataFrame to int16 if they exist."""
    mapping = {col: "float32" for col in columns if col in ddf.columns}
    return ddf.astype(mapping)



    



