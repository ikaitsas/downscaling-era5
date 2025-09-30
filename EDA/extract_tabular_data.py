#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 16:00:52 2025

@author: ykaitsas
"""

from dask.distributed import Client
import preprocessing_tabular_functions as pretab
import os
import xarray as xr


if __name__ == '__main__':
    

    # Start Dask client
    #client = Client()
    #print("Dask dashboard:")
    #print(client.dashboard_link)
    
    chunk_number = 200_000
    
    # Paths
    pressure_levels_directory = "/home/ykaitsas/Documents/CDS-data-and-scripts/ERA5-pressure-levels-daily-Greece/*.nc"
    single_levels_directory = "/home/ykaitsas/Documents/CDS-data-and-scripts/ERA5-single-levels-daily-Greece/*.nc"
    dem_file = "/home/ykaitsas/Documents/downscaling-era5/dem/dem-files-for-downscaling/dem-era5-0.25deg.tif"
    
    output_path = "dataframe-dump"
    os.makedirs(output_path, exist_ok=True)
    
    sample_ds = xr.open_mfdataset(pressure_levels_directory, combine="by_coords", compat="no_conflicts")
    n_valid_time = sample_ds.valid_time.size
    sample_ds.close()
    del sample_ds
    valid_time_chunk = 2400
    valid_time_slices = [(i, min(i + valid_time_chunk, n_valid_time)) 
                         for i in range(0, n_valid_time, valid_time_chunk)]
    
    

    for idx, (start, end) in enumerate(valid_time_slices):
        print(f"Processing slice {idx}: valid_time {start} to {end}")
        
        pl, sl = pretab.load_and_preprocess(
            pressure_levels_directory, 
            single_levels_directory, 
            dem_file,
            keep_part_of_timeline=None,
            valid_time_slice=(start, end)
        )
    
        # Widen pressure levels
        pl_wide = pretab.widen_pressure_levels(pl)
    
        # Stack & chunk
        ds_stacked = pretab.stack_and_chunk(
            pl_wide, sl, dynamic_chunk_size=False, 
            n_target_partitions=20, chunk_size_static=chunk_number
        )
    
        # Convert to Dask DataFrame
        ddf = pretab.convert_to_ddf(ds_stacked)
        ddf = ddf.drop(columns=["number", "sample"], errors="ignore")
    
        # Save to Parquet
        df = ddf.compute()
        storage_path = os.path.join(output_path, f"df-{idx}.parquet")
        df.to_parquet(storage_path)
        del df
    
    print("Pipeline finished successfully.")
    
    

    