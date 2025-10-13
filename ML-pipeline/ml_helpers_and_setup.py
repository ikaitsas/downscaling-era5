#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 11:44:11 2025

@author: ykaitsas
"""
from __future__ import annotations
from typing import List, Optional, Dict, Tuple, Union

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


#%% importation of data helpers
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


#%% estimators
class WeightedLinearTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, 
                 linear_features=None, 
                 tree_features=None, 
                 linear_model=None, 
                 tree_model=None,
                 weight=0.5):
        """
        linear_features: indices for features passed to LinearRegression
        extratrees_features: indices for features passed to ExtraTreesRegressor
        linear_model: optional custom linear model
        extratrees_model: optional custom ExtraTreesRegressor
        weight: float in [0, 1], linear contribution to final prediction
        """
        self.linear_features = linear_features
        self.tree_features = tree_features
        self.linear_model = linear_model
        self.tree_model = tree_model
        self.weight = weight

    def fit(self, X, y):
        if not hasattr(X, 'columns'):
            raise ValueError("X must be a pandas DataFrame with column names when using feature names")
        
        if self.linear_features is None or self.tree_features is None:
            raise ValueError("Please specify linear_features and extratrees_features.")
        
        # Validate that all specified features exist in X
        missing_linear = [f for f in self.linear_features if f not in X.columns]
        missing_trees = [f for f in self.tree_features if f not in X.columns]
        
        if missing_linear:
            raise ValueError(f"linear_features not found in X: {missing_linear}")
        if missing_trees:
            raise ValueError(f"extratrees_features not found in X: {missing_trees}")

        X_linear = X[self.linear_features]
        X_tree = X[self.tree_features]
        
        # Initialize models
        if self.linear_model is None:
            self.linear_model_ = LinearRegression()
        else:
            self.linear_model_ = clone(self.linear_model)
            
        if self.tree_features is None:
            self.tree_model_ = ExtraTreesRegressor()
        else:
            self.tree_model_ = clone(self.tree_model)

        self.linear_model_ = clone(self.linear_model)
        self.tree_model_ = clone(self.tree_model)

        self.linear_model_.fit(X_linear, y)
        self.tree_model_.fit(X_tree, y)

        if not (0 <= self.weight <= 1):
            raise ValueError("weight_linear must be in [0, 1].")

        # Store feature names for validation in predict
        self.feature_names_in_ = X.columns.tolist()
        self.linear_features_ = self.linear_features
        self.tree_features_ = self.tree_features
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]
        
        return self


    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        
        # Ensure X has the same columns as during training
        if not hasattr(X, 'columns'):
            raise ValueError("X must be a pandas DataFrame with column names")
        
        if list(X.columns) != self.feature_names_in_:
            raise ValueError("Feature names in X do not match those seen during training")
        
        # Extract feature subsets using stored feature names
        X_linear = X[self.linear_features_]
        X_tree = X[self.tree_features_]

        y_pred_linear = self.linear_model_.predict(X_linear)
        y_pred_tree = self.tree_model_.predict(X_tree)

        w = self.weight
        return w * y_pred_linear + (1 - w) * y_pred_tree


    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        check_is_fitted(self, "is_fitted_")
        return np.array(['weighted_linear_trees_prediction'])


#%% composite estimator with build in PCA extractor - chatGPT gibberish
class ClimatePCAEstimator(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible climate predictor with:
      - PCA applied to selected spatiotemporal fields
      - Standardization of other covariates
      - Linear + Tree ensemble model
      - Geographic weighting (cos(latitude))
      - Time encoding (sin/cos of day of year)
    """

    def __init__(
        self,
        pca_variables: List[str],
        linear_variables: Optional[List[str]] = None,
        tree_variables: Optional[List[str]] = None,
        exclude_standardize: Optional[List[str]] = None,
        n_components: int = 5,
        linear_model_params: Optional[dict] = None,
        tree_model_params: Optional[dict] = None,
        static_covariates: Optional[Dict[str, xr.DataArray]] = None,
        time_variable: str = "valid_time",
        lat_name: str = "latitude",
        lon_name: str = "longitude",
        pca_cache: Optional[dict] = None
    ):
        self.pca_variables = pca_variables
        self.linear_variables = linear_variables or []
        self.tree_variables = tree_variables or []
        self.exclude_standardize = exclude_standardize or []
        self.n_components = n_components
        self.linear_model_params = linear_model_params or {}
        self.tree_model_params = tree_model_params or {}
        self.static_covariates = static_covariates or {}
        self.time_variable = time_variable
        self.lat_name = lat_name
        self.lon_name = lon_name
        self.pca_cache = pca_cache or {}

        # Internal objects
        self.pca_: Optional[PCA] = None
        self.scaler_: Optional[StandardScaler] = None
        self.feature_names_: List[str] = []
        self.model_: Optional[StackingRegressor] = None
        self._coords_template: Optional[xr.Dataset] = None

    # -----------------------------------------------------
    # Utility functions
    # -----------------------------------------------------

    def _encode_time(self, da: xr.Dataset) -> pd.DataFrame:
        """Encode day of year as sin and cos."""
        t = pd.to_datetime(da[self.time_variable].values)
        dayofyear = t.dayofyear
        sin_t = np.sin(2 * np.pi * dayofyear / 365.25)
        cos_t = np.cos(2 * np.pi * dayofyear / 365.25)
        df_time = pd.DataFrame(
            {"sin_doy": sin_t, "cos_doy": cos_t},
            index=pd.Index(t, name="time")
        )
        return df_time

    def _flatten_spatial(self, da: xr.DataArray) -> np.ndarray:
        """Flatten dataset to (time, space) array."""
        return da.stack(points=(self.lat_name, self.lon_name)).values

    def _apply_pca(self, ds: xr.Dataset, fit: bool = False) -> np.ndarray:
        """
        Apply PCA to the specified variables.
        PCA is fitted only on training data (fit=True).
        """
        # Build input matrix
        lat = ds[self.lat_name]
        cos_lat = np.cos(np.deg2rad(lat))
        weights = np.sqrt(cos_lat)
        weights_2d = weights.broadcast_like(ds[self.pca_variables[0]])

        arr_list = []
        for var in self.pca_variables:
            arr = ds[var] * weights_2d
            arr_list.append(self._flatten_spatial(arr))

        X_pca_input = np.concatenate(arr_list, axis=1)

        if fit:
            self.pca_ = PCA(n_components=self.n_components)
            pcs = self.pca_.fit_transform(X_pca_input)
        else:
            pcs = self.pca_.transform(X_pca_input)

        return pcs

    def _build_feature_matrix(
        self, ds: xr.Dataset, fit: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert dataset into tabular features.
        """
        X_list = []
        names = []

        # --- 1) PCA features ---
        if self.pca_variables:
            pcs = self._apply_pca(ds, fit=fit)
            pc_names = [f"PC{i+1}" for i in range(self.n_components)]
            X_list.append(pcs)
            names.extend(pc_names)

        # --- 2) Extra non-PCA variables ---
        vars_to_use = (
            set(self.linear_variables)
            .union(set(self.tree_variables))
            .difference(set(self.pca_variables))
        )

        for var in vars_to_use:
            arr = self._flatten_spatial(ds[var])
            X_list.append(arr)
            names.append(var)

        # --- 3) Static covariates ---
        for var_name, arr_da in self.static_covariates.items():
            arr_rep = arr_da.broadcast_like(ds[self.pca_variables[0]]).stack(points=(self.lat_name, self.lon_name)).values
            X_list.append(np.repeat(arr_rep[None, :], len(ds[self.time_variable]), axis=0))
            names.append(var_name)

        # --- 4) Time encoding ---
        time_features = self._encode_time(ds)
        X_list.append(np.tile(time_features.values, (ds.dims[self.lat_name] * ds.dims[self.lon_name], 1)).reshape(len(ds[self.time_variable]), -1, 2)[:, 0])
        X_list.append(np.repeat(time_features.values[:, np.newaxis, :], ds.dims[self.lat_name] * ds.dims[self.lon_name], axis=1).reshape(len(ds[self.time_variable]), -1, 2)[:, 0])
        names.extend(["sin_doy", "cos_doy"])

        # --- Concatenate ---
        X = np.hstack([x if x.ndim == 2 else x.reshape(len(ds[self.time_variable]), -1) for x in X_list])

        # --- 5) Scaling ---
        # Select columns that need scaling
        mask = [n not in self.exclude_standardize for n in names]
        X_to_scale = X[:, mask]
        if fit:
            self.scaler_ = StandardScaler()
            X[:, mask] = self.scaler_.fit_transform(X_to_scale)
        else:
            X[:, mask] = self.scaler_.transform(X_to_scale)

        return X, names

    # -----------------------------------------------------
    # Main sklearn API
    # -----------------------------------------------------

    def fit(self, X: xr.Dataset, y: xr.DataArray) -> ClimatePCAEstimator:
        """
        Fit PCA + linear and tree ensemble model on training dataset.
        """
        X_tab, names = self._build_feature_matrix(X, fit=True)
        self.feature_names_ = names

        # Prepare target
        y_flat = y.stack(points=(self.lat_name, self.lon_name)).values.reshape(-1)

        # Separate features for linear vs tree
        idx_linear = [i for i, n in enumerate(names) if n in self.linear_variables or n.startswith("PC")]
        idx_tree = [i for i, n in enumerate(names) if n in self.tree_variables or n not in self.linear_variables]

        lin_model = Pipeline([
            ("linreg", LinearRegression(**self.linear_model_params))
        ])

        tree_model = ExtraTreesRegressor(**self.tree_model_params)

        self.model_ = StackingRegressor(
            estimators=[
                ("linear", lin_model),
                ("tree", tree_model)
            ],
            final_estimator=ExtraTreesRegressor(n_estimators=200),
            passthrough=True
        )

        self.model_.fit(X_tab, y_flat)
        self._coords_template = X

        return self

    def predict(self, X: xr.Dataset, return_dataset: bool = True) -> Union[np.ndarray, xr.DataArray]:
        """
        Predict from a new dataset (e.g., validation/test).
        """
        X_tab, _ = self._build_feature_matrix(X, fit=False)
        y_pred = self.model_.predict(X_tab)

        if return_dataset:
            # Reshape back to (time, lat, lon)
            n_time = len(X[self.time_variable])
            n_points = X[self.lat_name].size * X[self.lon_name].size
            y_reshaped = y_pred.reshape(n_time, X[self.lat_name].size, X[self.lon_name].size)

            da_pred = xr.DataArray(
                y_reshaped,
                coords={
                    self.time_variable: X[self.time_variable],
                    self.lat_name: X[self.lat_name],
                    self.lon_name: X[self.lon_name]
                },
                dims=[self.time_variable, self.lat_name, self.lon_name],
                name="prediction"
            )
            return da_pred
        return y_pred

    def get_params(self, deep: bool = True) -> dict:
        """For BayesSearchCV compatibility."""
        return {
            "pca_variables": self.pca_variables,
            "linear_variables": self.linear_variables,
            "tree_variables": self.tree_variables,
            "exclude_standardize": self.exclude_standardize,
            "n_components": self.n_components,
            "linear_model_params": self.linear_model_params,
            "tree_model_params": self.tree_model_params,
            "static_covariates": self.static_covariates,
            "time_variable": self.time_variable,
            "lat_name": self.lat_name,
            "lon_name": self.lon_name,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


#%% setup stuff
chunk_number = 500_000

# Paths
pressure_levels_directory = "/home/ykaitsas/Documents/CDS-data-and-scripts/ERA5-pressure-levels-daily-Greece/*.nc"
single_levels_directory = "/home/ykaitsas/Documents/CDS-data-and-scripts/ERA5-single-levels-daily-Greece/*.nc"
dem_file = "/home/ykaitsas/Documents/downscaling-era5/dem/dem-files-for-downscaling/dem-era5-0.25deg.tif"
dem_era5_file="/home/ykaitsas/Documents/downscaling-era5/dem/era5-static-variables.nc"


sl = preprocess_single_levels(
    single_levels_directory, 
    dem_file_external=dem_file,
    dem_file_era5=dem_era5_file,
    drop_unrelated_vars=True,
    keep_part_of_timeline=None,
    valid_time_slice=None
    )

pl = preprocess_pressure_levels(
    pressure_levels_directory, 
    dem_file_external=dem_file,
    dem_file_era5=dem_era5_file,
    drop_unrelated_vars=True,
    keep_part_of_timeline=None,
    valid_time_slice=None
    )

# Widen pressure levels
pl_wide = widen_pressure_levels(pl)

ds = xr.merge([pl_wide, sl], compat="no_conflicts")
#del pl, pl_wide, sl
ds_stacked, sample_index = stack_and_chunk(pl_wide, sl, dynamic_chunk_size=True)
ddf = convert_to_ddf(ds_stacked).reset_index()
ddf = ddf.drop(columns=["index", "number", "sample"], errors="ignore")
ddf = convert_columns_to_float32(ddf, ["latitude", "longitude"])
ddf = ddf[(ddf.year>=2010) & (ddf.year<2013)]

df = ddf.compute()


X_train = df.drop(columns=["t2m"])[df.year<2012]
y_train = df["t2m"][df.year<2012]
X_test = df.drop(columns=["t2m"])[df.year==2012]
y_test = df["t2m"][df.year==2012]


 
#%%   
linear = LinearRegression()
extra_trees = ExtraTreesRegressor(
    n_estimators=50,
    max_depth=32,
    min_samples_leaf=2,
    min_samples_split=50,
    max_features=0.9,
    n_jobs=3,
    random_state=42
    )

model = WeightedLinearTreeRegressor(
    linear_features=["dem_era5"],
    tree_features=["lsm_era5", "latitude", "dayofyear"],
    linear_model=linear,
    tree_model=extra_trees,
    weight=0.02
    )

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



plt.scatter(y_test, y_pred, s=0.5)
#plt.title(f"Linear Model Weight: {w}")
plt.show()
print(f"RMSE: {rmse(y_test, y_pred)}")    


#%%
import seaborn as sns
sns.kdeplot(data=y_pred, c="k")
sns.kdeplot(data=y_test, c="b")
plt.show()


#%%
fields_to_keep = ["z500", "t850", "t700", "t500", "q850", "msl", "sp"]
ds1 = ds.sel(valid_time=ds.valid_time.dt.year.isin(range(2005, 2007)))

#ds1 = ds1.isel(longitude=slice(0, 22))

field_to_eof = "sp"

from eofs.xarray import Eof

lat_weights = np.sqrt(np.cos(np.deg2rad(ds.latitude)))

#remove seasonal cycle
clim = ds1[field_to_eof].groupby('valid_time.month').mean('valid_time')
anom = ds1[field_to_eof].groupby('valid_time.month') - clim
#anom = ds1[field_to_eof].mean('valid_time')

# standardize - account also for lat/lon grid
da_standardized = anom * lat_weights
da_standardized = da_standardized - da_standardized.mean(dim='valid_time')
#da_standardized = da_standardized / da_standardized.std(dim='valid_time')

da_standardized = da_standardized.drop_vars(['month'])

solver = Eof(da_standardized, center=False)

how_many = 10
# Get principal components (time series)
pcs = solver.pcs(pcscaling=0, npcs=how_many)

# Get explained variance
variance = solver.varianceFraction(neigs=how_many).values
print("Variance Fraction:")
print(np.round(variance,4))

# Get EOFs (spatial patterns)
eofs = solver.eofs(neofs=how_many)

print("Eigenvalues:")
print(solver.eigenvalues(neigs=how_many).values)
print("Egenvalues Typical Errors:")
print(solver.northTest(neigs=how_many).values)


#%%
spatial_mean = ds1[field_to_eof].mean(dim=['latitude', 'longitude'])
plt.figure(figsize=(12, 4))
spatial_mean.plot()
plt.title(f'Spatial Mean Time Series - {ds1[field_to_eof].name}')
plt.grid(True)
plt.show()

for i in range(0,1):
    anom[-i-1,:,:].plot(cmap="viridis")
    plt.title(f"Anomaly from monthly - {anom[-i-1,:,:].valid_time.dt.date.values}")
    plt.xlim([18.875,30.125])
    plt.ylim([34.375,42.125])
    plt.show()

for i in range(0,1):
    ds1[field_to_eof][-i-1,:,:].plot(cmap="viridis")
    plt.title(f"Variable field - {anom[-i-1,:,:].valid_time.dt.date.values}")
    plt.xlim([18.875,30.125])
    plt.ylim([34.375,42.125])
    plt.show()

for neof in range(0,how_many):
    eofs[neof,:,:].plot(cmap="coolwarm")
    plt.title(f"{neof+1}st PC Field")
    #plt.axis("square")
    plt.xlim([18.875,30.125])
    plt.ylim([34.375,42.125])
    plt.show()


#%% check spatial autocorrelation of ERA5 fields
from scipy import signal 

field_to_autocorr = "sp"
da = ds1[field_to_autocorr] * lat_weights
da_t = da.mean(dim="valid_time")
da_anom = da_t - da_t.mean()
arr = da_anom.values

autocorr2d = signal.correlate2d(arr, arr, mode='full', boundary='fill', fillvalue=0)
autocorr2d /= autocorr2d.max()

plt.imshow(autocorr2d)
plt.title(f"Spatial Autocorrelation: {field_to_autocorr}")
plt.show()
