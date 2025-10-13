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

#from sklearn import set_config  # wokrs on jupyter/vscode only...
#set_config(display='diagram')

from sklearn.metrics import root_mean_squared_error as rmse 
from sklearn.metrics import mean_absolute_error as mae 
from sklearn.metrics import median_absolute_error as medae 
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns


ddf_path = "/home/ykaitsas/Documents/downscaling-era5/preprocessing/dataframe-dump-client"
ddf_name_convention = "part.*.parquet"
ddf = dd.read_parquet(os.path.join(ddf_path, ddf_name_convention))
ddf = ddf[(ddf.year>=1980) & (ddf.year<=2010)]
df = ddf.compute()

train_sample_size = 1_500_000
test_sample_size = 500_000

scatter_plot_name = "scatter-all-exclude-longitude-q850-700-500-z850-500.png"


target_column = "t2m"
selected_covariate_columns = [
    "dem_era5", 
    "lsm_era5", 
    "latitude", 
    #"longitude", #<<
    #'z850', #<<
    #'z700', #<<
    'z500', 
    't850', 
    't700', 
    't500',
    #'r850', 
    #'r700', 
    #'r500', 
    'q850', 
    #'q700', 
    #'q500', #<<
    "msl",  #<<
    "sp",   #<<
    #"month",
    "dayofyear",
    ]
'''
best qq plot performance is achieved by incorporating all tempearture pressure
levels.
inclusion of both msl and sp makes results better
for the best qq plot performance, all t levels, z500, and q80 and both msl
and sp are selected
'''

# Keep only selected columns in desired order
df = df[selected_covariate_columns + [target_column]]

# Define scaling per column group
cols_passthrough = [
    "dem_era5", "dayofyear", "lsm_era5"
        ]
cols_minmaxed = [
    #"latitude", 
    #"longitude"
    ]
cols_standardized = [
    c for c in selected_covariate_columns if c not in cols_minmaxed + \
        cols_passthrough
        ]


#%%
# Build ColumnTransformer (order of transformers doesn’t matter, we'll restore order later)
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


#%% Train model on a sample
# Separate features and target 
X = df.drop(columns=[str(target_column)]) 
y = df[str(target_column)] 
del df

target_rows = int(len(y))
X_train = X.iloc[:train_sample_size,:]
y_train = y.iloc[:train_sample_size]

X_test = X.iloc[-test_sample_size:,:]
y_test = y.iloc[-test_sample_size:]


print("Starting training...")
# Time the fit
t0 = time.perf_counter()
#weights = np.where(y_train < -12, 100.0, 1.0)
model.fit(X_train, y_train)#, extratrees__sample_weight=weights)
t1 = time.perf_counter()
elapsed = t1 - t0
print(f"Sample training time for {train_sample_size:,} rows: {elapsed:.2f} seconds")

scale = target_rows / train_sample_size
naive_seconds = elapsed * scale
naive_hours = naive_seconds / 3600.0
print(f"Naive extrapolation (linear) to {target_rows:,} rows -> {naive_hours:.2f} hours")


#%% some testing
# Feature importances (restore exact order from selected_columns)
extratrees = model.named_steps["extratrees"]

# Use selected_columns as authoritative order
feature_names = selected_covariate_columns.copy()

importances = pd.Series(extratrees.feature_importances_, index=feature_names)
print(importances.sort_values(ascending=False))

y_pred = model.predict(X_test)

print(f"RMSE: {rmse(y_pred, y_test):.3f}")
print(f"MAE: {mae(y_pred, y_test):.3f}")
print(f"MedAE: {medae(y_pred, y_test):.3f}")
print(f"R squared: {r2_score(y_pred, y_test):.3f}")

print(model.named_steps['preprocessor'].get_feature_names_out())
#print(model.named_steps['preprocessor'])
'''
X_train= pd.DataFrame(
    preprocessor.transform(X_train), columns=preprocessor.get_feature_names_out()
    )
'''
#print(X_train.head())
#print(X_train.info())


#%% scatter plot
#plt.scatter(y_test, y_pred, s=0.5)
hb = plt.hexbin(y_pred, y_test, gridsize=400, cmap='coolwarm', mincnt=1)
plt.colorbar(hb, label='Counts')

lims = [-20, 40]
plt.plot(lims, lims, 'k--', linewidth=0.75, label='1:1 line', alpha=0.5)

plt.ylabel("T2m - test")
plt.xlabel("T2m - predicted")
plt.axis('square')
plt.xlim(lims)
plt.ylim(lims)
'''
plt.legend(
    [str(model.named_steps['preprocessor'].get_feature_names_out())],
    fontsize=5
    )
'''
plt.grid()
image_path = "images"
os.makedirs(image_path, exist_ok=True)
plt.savefig(os.path.join(image_path, scatter_plot_name), 
            dpi=500, bbox_inches="tight")
plt.show()


sns.kdeplot(data=y_pred, c="k")
sns.kdeplot(data=y_test, c="b")
plt.legend(["predicted", "test"])
plt.show()


plt.scatter(np.sort(y_pred), np.sort(y_test), s=0.5)
lims = [-20, 40]
plt.plot(lims, lims, 'k--', linewidth=0.75, label='1:1 line', alpha=0.5)
plt.ylabel("T2m - test")
plt.xlabel("T2m - predicted")
plt.axis('square')
plt.xlim(lims)
plt.ylim(lims)
plt.grid()
plt.show()
