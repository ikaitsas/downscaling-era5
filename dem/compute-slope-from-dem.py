# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 12:44:51 2025

This computation script uses convolution to sole the slope calculations, after 
converting the geographic lat/lon resolution of the original DEM dataset to a
distance metric resolution, in meters.

Note that this metric resolution changes with latitude, mainly the horizontal 
distance, so this is the reason for the development of this specific script for
slope computations, as GDAL might not apply a customized conversion factor (i
am not entirely sure, i must look in to it more...).

Slope computation uses Horn's weigted kernel formula, fitted into a convolution
function, that is then combined with dask operations for handling large tiffs,
without explicitly loading them into the system's RAM.

TA GLO APO TO GOOGLE EART HENGINE KATEVHKAN ME ENA GRID PATTERN KATHE PERIPOU
25-30 PIXELS. GIATI RE GMT TOUS????
MHPWS EXEI GINEI LATHOS KATA TO SAMPLING APO TO API TOU GEE??
@author: Giannis
"""

import numpy as np
import rioxarray
import xarray as xr
import dask.array
from scipy import ndimage
import dask_image.ndfilters as dimg
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt

meters_per_deg = 111320
dem_file = "DEM-GLO30-results\dem-GLO30-N42.0-W19.0-S34.5-E30.0.tif"

export_nc_to_device = True
export_Gtiff_to_device = True
plot_morphography = True
#"delete" some datasets - might help with memory?
save_memory = False

#%%
dem = rioxarray.open_rasterio(
    dem_file, 
    chunks={"x": 1800, "y": 1800}
    ).squeeze("band", drop=True)
print("1")
if dem.rio.crs.is_geographic:
    gt = dem.rio.transform()
    rows, cols = dem.rio.height, dem.rio.width
    
    latitudes = np.linspace(gt[5], gt[5] + gt[4] * rows, rows)
    longitudes = np.linspace(gt[2], gt[2] + gt[0] * cols, cols)
    
    #find center latitude for scaling factor
    top = gt[5]
    bot = top + gt[4]*dem.rio.height
    left = gt[2]
    right = left + gt[0]*dem.rio.width
    res_lat = dem.rio.resolution()[0]
    res_lon = abs(gt[0])
    
    idx_rows = np.arange(rows)
    center_latitudes = top - (idx_rows+0.5)*res_lat
    
    # meters per degree for latitude-longitude
    meters_latitude = meters_per_deg
    meters_per_deg_lat = 111132.92 - \
        559.82*np.cos(2*np.deg2rad(center_latitudes)) + \
            1.175*np.cos(4*np.deg2rad(center_latitudes))
    
    meters_longitude = meters_per_deg * np.cos(np.radians(center_latitudes))
    meters_per_deg_lon = 111412.84*np.cos(np.deg2rad(center_latitudes)) - \
        93.5*np.cos(3*np.deg2rad(center_latitudes))
     
    meters_dx = res_lon * meters_per_deg_lon
    meters_dy = res_lat * meters_per_deg_lat
    
    if meters_dx.shape[0]==meters_dx.size: # check if it is scalar
        meters_dx = np.broadcast_to(meters_dx[:,None], dem.shape)
    if meters_dy.shape[0]==meters_dy.size: # check if it is scalar
        meters_dy = np.broadcast_to(meters_dy[:,None], dem.shape)
else:
    gt = dem.rio.transform()
    dx = abs(gt[0])
    dy = abs(gt[4])
    meters_dx = dx
    meters_dy = dy


def compute_slope_horn(tile, tile_dx, tile_dy, zfactor=1.0):
    # Horn kernels for convolution
    kx = np.array([[ -1,  0,  1],
                   [ -2,  0,  2],
                   [ -1,  0,  1]]) / 8.0

    ky = np.array([[ -1, -2, -1],
                   [  0,  0,  0],
                   [  1,  2,  1]]) / 8.0
    
    # Ensure dx,dy are broadcast to 2D shape
    if np.isscalar(tile_dx):
        tile_dx = np.full_like(tile, tile_dx, dtype=float)
    elif tile_dx.ndim == 1 and tile_dx.shape[0] == tile.shape[0]:
        tile_dx = np.broadcast_to(tile_dx[:, None], tile.shape)

    if np.isscalar(tile_dy):
        tile_dy = np.full_like(tile, tile_dy, dtype=float)
    elif tile_dy.ndim == 1 and tile_dy.shape[0] == tile.shape[0]:
        tile_dy = np.broadcast_to(tile_dy[:, None], tile.shape)
    
    # Convolve DEM with Horn kernels
    dz_dx = ndimage.convolve(tile, kx, mode='mirror') / tile_dx * zfactor
    dz_dy = ndimage.convolve(tile, ky, mode='mirror') / tile_dy * zfactor

    # Compute slope
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    
    return slope_deg

def compute_slope_horn_dask(tile, tile_dx, tile_dy, zfactor=1.0):
    #Ensureeverything is the tile's dtype
    dtype=tile.dtype
    
    # Horn kernels for convolution
    kx = np.array([[ -1,  0,  1],
                   [ -2,  0,  2],
                   [ -1,  0,  1]]) / 8.0
    kx = kx.astype(dtype)

    ky = np.array([[ -1, -2, -1],
                   [  0,  0,  0],
                   [  1,  2,  1]]) / 8.0
    ky = ky.astype(dtype)
    
    zfactor=np.array(zfactor, dtype=dtype)
    tile_data = tile.data

    # Ensure dx, dy are Dask arrays with matching dtype
    if np.isscalar(tile_dx):
        tile_dx_data = dask.array.full_like(tile_data, tile_dx, dtype=dtype)
    elif isinstance(tile_dx, xr.DataArray):
        tile_dx_data = dask.array.from_array(tile_dx.data.astype(dtype), chunks=tile_data.chunks)
    else:  # assume numpy array
        tile_dx_data = dask.array.from_array(tile_dx.astype(dtype), chunks=tile_data.chunks)

    if np.isscalar(tile_dy):
        tile_dy_data = dask.array.full_like(tile_data, tile_dy, dtype=dtype)
    elif isinstance(tile_dy, xr.DataArray):
        tile_dy_data = dask.array.from_array(tile_dy.data.astype(dtype), chunks=tile_data.chunks)
    else:
        tile_dy_data = dask.array.from_array(tile_dy.astype(dtype), chunks=tile_data.chunks)
    
    #tile_dx = dask.array.from_array(tile_dx, chunks=tile.chunks)
    #tile_dy = dask.array.from_array(tile_dy, chunks=tile.chunks)
    
    # Convolve DEM with Horn kernels
    dz_dx = dimg.convolve(tile_data, kx, mode='mirror') / tile_dx_data * zfactor
    dz_dy = dimg.convolve(tile_data, ky, mode='mirror') / tile_dy_data * zfactor

    # Compute slope
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    
    return slope_deg


'''
#%% compute slope
slope_file_name = f"{dem_file[:-4]}-slope.nc"


dem_rechunked = dem.chunk({"y": -1}).astype(np.int16)  # all rows in one chunk
meters_dx_rechunked = dask.array.from_array(meters_dx, chunks=dem_rechunked.chunks)
meters_dy_rechunked = dask.array.from_array(meters_dy, chunks=dem_rechunked.chunks)
print("2")
slope_da = xr.apply_ufunc(
    compute_slope_horn,
    dem_rechunked,        # DEM DataArray
    meters_dx_rechunked,         # full 2D dx array
    meters_dy_rechunked,         # full 2D dy array
    input_core_dims=[["y","x"], ["y","x"], ["y","x"]],
    output_core_dims=[["y","x"]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[np.int16],
    dask_gufunc_kwargs={"allow_rechunk": True}
)

slope_da_nc = xr.DataArray(
    slope_da,                  # the dask-backed slope array from apply_ufunc
    coords={"y": dem.y, "x": dem.x},
    dims=("y", "x"),
    name="slope"
)

slope_da_nc = slope_da_nc.rio.write_crs(dem.rio.crs).rio.write_transform(dem.rio.transform())


# try this too...
encoding = {"slope": {"zlib": True, "complevel": 4, "chunksizes": (900, 900)}}

with ProgressBar():
    slope_da_nc.to_netcdf("slope.nc", format="NETCDF4", encoding=encoding)
'''

print("3")
'''
with ProgressBar():
    slope_da_nc.to_zarr( #.chunk({"y": 1800, "x": 1800})
        "slopes\slope.zarr",
        mode="w",
        #compressor="zlib"
    )
    '''

#%% tha ta spaswww - try this somethime
'''
def block_slope(block, block_info=None):
    # block_info contains the slice indices of this chunk
    # it automatically gets passed by dask when it sees it writen?
    slices = block_info[None]['array-location']
    y0, y1 = slices[0]
    x0, x1 = slices[1]
    
    # slice DX and DY to match this block
    block_dx = DX[y0:y1, x0:x1].compute()
    block_dy = DY[y0:y1, x0:x1].compute()
    
    # compute slope for this block
    return compute_slope_horn(block, block_dx, block_dy)

slope_da = da.map_overlap(
    block_slope,
    dem_da,
    depth=1,             # 1 pixel overlap for 3x3 convolution
    boundary='reflect',
    dtype=np.float32
)

slope_da.to_netcdf(slope_file_name)
'''


#%% test

demaki_file = f"tiles-GLO60\DEM-tile-39.0-21.0.tif"

demaki = rioxarray.open_rasterio(
    demaki_file,
    ).squeeze("band", drop=True)

plt.imshow(demaki, cmap='magma')
plt.title(f"DEM - GEE Sampling: 60 m")
#plt.yticks([])
#plt.xticks([])
#plt.savefig(f"tile-demaki.png", dpi=500)
plt.show()

slopaki = compute_slope_horn(demaki, 24.3, 30.86, zfactor=1.0)
plt.imshow(slopaki, cmap='magma')
plt.title(f"Slope - GEE Sampling: 60 m")
#plt.yticks([])
#plt.xticks([])
#plt.savefig(f"tile-demaki.png", dpi=500)
plt.show()
