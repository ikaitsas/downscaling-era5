#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 16:00:52 2025

@author: ykaitsas
"""

from dask.distributed import Client
import preprocessing_tabular_functions as preproc
import os
import xarray as xr
import dask.dataframe as dd
import pandas as pd


if __name__ == '__main__':
    

    # Start Dask client
    client = Client(
        n_workers=4,              # fewer workers - 4 
        threads_per_worker=3,     # don’t oversubscribe threads - 3
        memory_limit="4GB"        # or set per-worker memory - 4
    )    
    print("Dask dashboard:")
    print(client.dashboard_link)
    
    chunk_number = 500_000
    
    # Paths
    pressure_levels_directory = "C:\\Users\\yiann\\Documents\\ERA5-ERA5-Land\\ERA5-pressure-levels-daily-Greece\\*.nc"
    single_levels_directory = "C:\\Users\\yiann\\Documents\\ERA5-ERA5-Land\\ERA5-single-levels-daily-Greece\\*.nc"
    dem_file = "C:\\Users\\yiann\\Documents\\downscaling-era5\\dem\\dem-files-for-downscaling\\dem-era5-0.25deg.tif"

    output_path = "dataframe-dump-client"
    os.makedirs(output_path, exist_ok=True)
    

    
    pl, sl = preproc.load_and_preprocess(
        pressure_levels_directory, 
        single_levels_directory, 
        dem_file,
        keep_part_of_timeline=None,
        valid_time_slice=None
    )

    # Widen pressure levels
    pl_wide = preproc.widen_pressure_levels(pl)

    # Stack & chunk
    ds_stacked, sample_index = preproc.stack_and_chunk(
        pl_wide, sl, dynamic_chunk_size=True, create_index=True,
        n_target_partitions=10, chunk_size_static=chunk_number
    )
    
    
    # Convert to Dask DataFrame
    ddf = preproc.convert_to_ddf(ds_stacked)
    ddf = ddf.drop(columns=["number", "sample"], errors="ignore")
    ddf = preproc.convert_columns_to_float32(ddf, ["latitude", "longitude"])
    ddf.to_parquet(output_path, write_index=False, overwrite=True)
    
    sample_index.to_frame(index=False).to_parquet(
        os.path.join(output_path, 
                     "sample_index_for_dataset_reconstruction.parquet"),
        index=False
        )

    print("Pipeline finished successfully.")        
    
    

    