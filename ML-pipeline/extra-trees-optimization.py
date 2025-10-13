#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 17:55:31 2025

@author: ykaitsas
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
import dask.dataframe as dd


from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def import_tabular_data_for_model_calibration(
        data_folder, 
        data_name_convention = "part.*.parquet",
        period_to_keep_start = 1960,
        period_to_keep_end = None,
        period_column_name = "year",
        compute_dataframe = False
        ):
    
    ddf = dd.read_parquet(os.path.join(data_folder, data_name_convention))
    
    if period_to_keep_end is None:
        period_to_keep_end = ddf[period_column_name].max().compute()
     
     
    ddf = ddf[
        (ddf[period_column_name]>=period_to_keep_start) & \
           (ddf[period_column_name]<=period_to_keep_end) 
        ]
    
    return ddf


def organize_preprocessor(
        dataframe,
        selected_datafame_columns=None,
        columns_passthrough=None,
        columns_minmaxed=None,
        columns_standardized=None,
        perform_transform=True
        ):
    
    if selected_datafame_columns == None:
        selected_datafame_columns = dataframe.columns
    
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("passthrough", "passthrough", columns_passthrough),
            ("minmax", MinMaxScaler(), columns_minmaxed),
            ("standard", StandardScaler(), columns_standardized),
            ],
        remainder="drop"
    )
    
    return preprocessor


#%%
ddf_path = "/home/ykaitsas/Documents/downscaling-era5/preprocessing/dataframe-dump-client"
ddf_name_convention = "part.*.parquet"
ddf = dd.read_parquet(os.path.join(ddf_path, ddf_name_convention))
ddf = ddf[(ddf.year>=1980) & (ddf.year<=2010)]
df = ddf.compute()

train_sample_size = 1_500_000
test_sample_size = 500_000


target_column = "t2m"
selected_covariate_columns = [
    "dem_era5", 
    "lsm_era5", 
    "latitude", 
    #"longitude", #<<
    'z850', #<<
    'z700', #<<
    'z500', 
    't850', 
    't700', 
    't500',
    #'r850', 
    #'r700', 
    #'r500', 
    'q850', 
    'q700', 
    'q500', #<<
    "msl",  #<<
    #"sp",   #<<
    #"month",
    "dayofyear",
    ]

# Keep only selected columns in desired order
df = df[selected_covariate_columns + [target_column]]

# Define scaling per column group
cols_passthrough = [
    "dem_era5", "dayofyear", "lsm_era5"
        ]
cols_minmaxed = [
    "latitude", 
    #"longitude"
    ]
cols_standardized = [
    c for c in selected_covariate_columns if c not in cols_minmaxed + \
        cols_passthrough
        ]
    

preprocessor = ColumnTransformer(
    transformers=[
        ("passthrough", "passthrough", cols_passthrough),
        ("minmax", MinMaxScaler(), cols_minmaxed),
        ("standard", StandardScaler(), cols_standardized),
        ],
    remainder="drop",
    verbose_feature_names_out=False
)

# Define the model outside the pipeline
extratrees_model = ExtraTreesRegressor(
n_estimators=50,
max_depth=32,
min_samples_leaf=2,
min_samples_split=50,
max_features=0.9,
n_jobs=4,
random_state=42
)


# Build pipeline by adding the model later
model = Pipeline(steps=[
("preprocessor", preprocessor),
("extratrees", extratrees_model)
])


