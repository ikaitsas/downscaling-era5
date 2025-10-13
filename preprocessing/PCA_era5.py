#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 19:46:50 2025

@author: ykaitsas
"""

import os
import numpy as np
import xarray as xr
import dask.dataframe as dd
from sklearn.decomposition import PCA
import preprocessing_tabular_functions as preproc
import matplotlib.pyplot as plt


pressure_levels_directory = "/home/ykaitsas/Documents/CDS-data-and-scripts/ERA5-pressure-levels-daily-Greece/*.nc"
single_levels_directory = "/home/ykaitsas/Documents/CDS-data-and-scripts/ERA5-single-levels-daily-Greece/*.nc"


sl = preproc.preprocess_single_levels(
    single_levels_directory, 
    dem_file_external=None,
    dem_file_era5=None,
    drop_unrelated_vars=True,
    keep_part_of_timeline=None,
    valid_time_slice=None
    )

pl = preproc.preprocess_pressure_levels(
    pressure_levels_directory, 
    dem_file_external=None,
    dem_file_era5=None,
    drop_unrelated_vars=True,
    keep_part_of_timeline=None,
    valid_time_slice=None
    )

# Widen pressure levels
pl_wide = preproc.widen_pressure_levels(pl)

PCA_fields = [
    'z850', 
    'z700',
    'z500', 
    #'r850', 
    #'r700', 
    #'r500', 
    'q850', 
    'q700', 
    'q500', 
    't850', 
    't700',
    't500', 
    'msl', 
    #'sp',
    ]

ds = xr.merge([sl, pl_wide])

data_arrays = [ds[field] for field in PCA_fields]
available_fields = [v for v in PCA_fields if v in ds]


if not available_fields:
    raise ValueError("None of the requested PCA fields are in the merged dataset!")

print(f"Using {len(available_fields)} variables for PCA: {available_fields}")


da_all = xr.concat([ds[v] for v in available_fields], dim="variable")
da_all["variable"] = available_fields

da_2d = da_all.stack(feature=("latitude", "longitude", "variable"))

# Drop any features with NaNs
da_2d = da_2d.dropna(dim="feature")

X = da_2d.values  # shape: (time, n_features)


n_components = 5  # choose number of modes
pca = PCA(n_components=n_components)
pcs = pca.fit_transform(X)
explained_var = pca.explained_variance_ratio_
