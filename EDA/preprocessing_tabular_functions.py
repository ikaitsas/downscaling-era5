#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 15:55:29 2025

@author: ykaitsas

dask.delayed needs better implementation:
    https://docs.dask.org/en/stable/delayed-best-practices.html
"""

import os
import dask
import xarray as xr
import pandas as pd
import rioxarray as rio
from numpy import ceil as numpy_ceil
import dask.dataframe as dd
from dask.diagnostics import ResourceProfiler, ProgressBar


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
    df['day'] = df[time_col].dt.dayofyear
    
    if drop_original:
        df = df.drop(columns=[time_col], errors='ignore')
    
    return df


def load_and_preprocess(
    pressure_levels_directory, 
    single_levels_directory, 
    dem_file, 
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
                           chunks={"valid_time": 50, "latitude": 256, "longitude": 256}
                           )
    sl = xr.open_mfdataset(single_levels_directory, 
                           combine="by_coords", compat="no_conflicts",
                           chunks={"valid_time": 50, "latitude": 256, "longitude": 256}
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

    if valid_time_slice is not None:
        start, end = valid_time_slice
        pl = pl.isel(valid_time=slice(start, end))
        sl = sl.isel(valid_time=slice(start, end))
    elif keep_part_of_timeline is not None:
        pl = pl.isel(valid_time=slice(0, keep_part_of_timeline))
        sl = sl.isel(valid_time=slice(0, keep_part_of_timeline))

    return pl, sl


def widen_pressure_levels(pl):
    """Convert pressure level variables into separate 2D variables."""
    pl_vars = [v for v in pl.data_vars if 'pressure_level' in pl[v].dims]
    pl_wide = xr.Dataset()
    for var in pl_vars:
        for level in pl.pressure_level.values:
            da = pl[var].sel(pressure_level=level, drop=True).reset_coords("number", drop=True)
            pl_wide[f"{var}{int(level)}"] = da
            del da
    return pl_wide


def stack_and_chunk(pl_wide, sl, 
                    dynamic_chunk_size=False, n_target_partitions=20, 
                    chunk_size_static=200_000):
    """Merge datasets, stack for ML, and chunk."""
    ds = xr.merge([pl_wide, sl], compat="no_conflicts")
    ds_stacked = ds.stack(sample=("valid_time", "latitude", "longitude"))

    if dynamic_chunk_size:
        chunk_size = int(numpy_ceil(ds_stacked.sizes["sample"] / n_target_partitions))
    else:
        chunk_size = chunk_size_static

    ds_stacked = ds_stacked.chunk({"sample": chunk_size})
    return ds_stacked


def convert_to_ddf(ds_stacked):
    """Convert stacked Dataset to Dask DataFrame."""
    ddf = ds_stacked.to_dask_dataframe()
    ddf = add_time_features(ddf)
    return ddf


def save_ddf(ddf, path, engine="pyarrow"):
    
    os.makedirs(path, exist_ok=True)
    df = ddf.compute()
    df.to_parquet(path)

    



