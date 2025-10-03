#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 11:51:31 2025

@author: ykaitsas
"""

import os
import time

import pandas as pd
import dask.dataframe as dd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import root_mean_squared_error as rmse
import matplotlib.pyplot as plt



ddf_path = "/home/ykaitsas/Documents/downscaling-era5/preprocessing/dataframe-dump-client"
ddf_name_convention = "part.*.parquet"
ddf = dd.read_parquet(os.path.join(ddf_path, ddf_name_convention))
df = ddf.compute()


target_column = "t2m"
selected_covariate_columns = [
    "dem_era5", 
    "lsm_era5", 
    "latitude", 
    "longitude", 
    #'z850', 
    'z700', 
    'z500', 
    #'t850', 
    't700', 
    't500',
    #'r850', 
    #'r700', 
    'r500', 
    #'q850', 
    'q700', 
    'q500',
    "msl", 
    "sp", 
    "month",
    "dayofyear",
    ]

# Keep only selected columns in desired order
df = df[selected_covariate_columns + [target_column]]

# Define scaling per column group
cols_minmaxed = [
    "latitude", 
    "longitude"
    ]
cols_standardized = [
    #'z850', 
    'z700', 
    'z500', 
    #'t850', 
    't700', 
    't500',
    #'r850', 
    #'r700', 
    'r500', 
    #'q850', 
    'q700', 
    'q500',
    "msl",
    "sp"
    ]
cols_no_scaling = [
    c for c in selected_covariate_columns if c not in cols_minmaxed + \
        cols_standardized
        ]


# Build ColumnTransformer (order of transformers doesn’t matter, we'll restore order later)
preprocessor = ColumnTransformer(
    transformers=[
        ("minmax", MinMaxScaler(), cols_minmaxed),
        ("standard", StandardScaler(), cols_standardized),
        ("passthrough", "passthrough", cols_no_scaling)
        ],
    remainder="drop"
)


# Separate features and target 
X = df.drop(columns=["t2m"]) 
y = df["t2m"] 
del df

target_rows = int(len(y))


#%%
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


#%% Train model on a sample
sample_size = 3_300_000

X_train = X.iloc[:sample_size,:]
y_train = y.iloc[:sample_size]

X_test = X.iloc[-sample_size:]
y_test = y.iloc[-sample_size:]


print("Starting training...")
# Time the fit
t0 = time.perf_counter()
#weights = np.where(y_train < -12, 100.0, 1.0)
model.fit(X_train, y_train)#, extratrees__sample_weight=weights)
t1 = time.perf_counter()
elapsed = t1 - t0
print(f"Sample training time for {sample_size:,} rows: {elapsed:.2f} seconds")

scale = target_rows / sample_size
naive_seconds = elapsed * scale
naive_hours = naive_seconds / 3600.0
print(f"Naive extrapolation (linear) to {target_rows:,} rows -> {naive_hours:.2f} hours")


#%%
# Feature importances (restore exact order from selected_columns)
extratrees = model.named_steps["extratrees"]


# Use selected_columns as authoritative order
feature_names = selected_covariate_columns.copy()


importances = pd.Series(extratrees.feature_importances_, index=feature_names)
print(importances.sort_values(ascending=False))

#%% some testing
y_pred = model.predict(X_test)

print(f"RMSE: {rmse(y_pred, y_test):.2f}")

plt.scatter(y_test, y_pred, s=0.5)
plt.xlabel("T2m - test")
plt.ylabel("T2m - predicted")
plt.axis('square')
plt.xlim([-20, 40])
plt.ylim([-20, 40])
plt.grid()
plt.show()

