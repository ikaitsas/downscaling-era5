# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 13:07:24 2025

@author: ykaitsas

A resampled tiff is created, for matching the 1 arcsecond resolution.
CAUTION: only slightly resample the tiffs (e.g. ~1arcsec -> 1arcsec), so that
no distortion is produced by large areas of aggregation.

A dedicated script will be written for resampling (i.e. averaging) over 
larger areas, using both block aggregation and gaussian filters, since the
latter dont distort in the diagonal direction...

The parameter tiling_shape is there for optimizing the resulting rasters for
dask workflows, if needed. It replaces the traditional/standard row tiling,
that might behave weirdly with dask, or not work at all? (to psaxnw akoma)
"""

import os
import glob
import rasterio
import subprocess
import matplotlib.pyplot as plt


band = "WBM"
dem_tiles_folder = f"tiles-GLO30-{band}-corrected"


tif_file_list = f"{band}_tiles_list-corrected.txt"
vrt_file = f"vrt-merged-{band}-corrected.vrt"
merged_tif = f"{band}-merged-30-corrected.tif"
resampled_tif = f"{band}-merged-1arc-corrected.tif"


# inputs for gdal commands
target_resolution_deg = 1/(60*60)  # 1 arcsecond
tiling_shape = 512  # tiling - must be power of 2

if band == "WBM":
    resampling_method = "nearest"
else:
    resampling_method = "bilinear"  # others: "lanczos", "nearest" 


# this option creates overviews/pyramids
# these are downsampled versions of the big raster
# preferebly font enable it, as they are stored internally
# as a result the image size is inflated
build_overview = False

tif_storage = "processed-tifs"
os.makedirs(tif_storage, exist_ok=True)

tif_files = glob.glob(os.path.join(dem_tiles_folder, "*.tif"))
with open(tif_file_list, "w") as f:
    f.write("\n".join(tif_files))
print(f"Found {len(tif_files)} GeoTIFFs, written to {tif_file_list}")


#%% workflow
#  build the vrt file
subprocess.run([
    "gdalbuildvrt",
    "-input_file_list", tif_file_list,
    os.path.join(tif_storage, vrt_file)
], check=True)
print(f"VRT for merging created: {vrt_file}")


# convert VRT to compressed GeoTIFF
subprocess.run([
    "gdal_translate",
    "-of", "GTiff",
    "-co", "COMPRESS=LZW",
    "-co", "TILED=YES",
    "-co", f"BLOCKXSIZE={tiling_shape}",
    "-co", f"BLOCKYSIZE={tiling_shape}",
    "-co", 'BIGTIFF=YES',
    os.path.join(tif_storage, vrt_file),
    os.path.join(tif_storage, merged_tif)
], check=True)
print(f"Merged DEM written to: {merged_tif}")


# resample the merged GeoTIFF to desired resolution
subprocess.run([
    "gdalwarp",
    "-tr", str(target_resolution_deg), str(target_resolution_deg),  
    "-r", f"{resampling_method}",  # also can try "lanczos"
    "-of", "GTiff",
    "-co", "COMPRESS=LZW",
    "-co", "TILED=YES",
    "-co", "TILED=YES",
    "-co", f"BLOCKXSIZE={tiling_shape}",
    "-co", f"BLOCKYSIZE={tiling_shape}",
    "-co", 'BIGTIFF=YES',
    os.path.join(tif_storage, merged_tif),
    os.path.join(tif_storage, resampled_tif)
], check=True)
print(f"Slightly resampled DEM written to: {resampled_tif}")


# build overviews (pyramids) for fast GIS/Dask usage
if build_overview == True:
    subprocess.run([
        "gdaladdo",
        "-r", "average",  # resampling method for overviews
        os.path.join(tif_storage, resampled_tif),
        "2", "4", "8", "16", "32", "64"
    ], check=True)
    print(f"Overviews built for: {resampled_tif}")


#%% playground
'''
import rioxarray as rio
da = rio.open_rasterio(os.path.join(tif_storage, resampled_tif)).squeeze("band",drop=True)

for tif_file in tif_files:
    array = rasterio.open(tif_file).read(1)
    plt.imshow(array, vmin=0, vmax=3000, cmap="magma")
    #plt.savefig(tif_file[12:-4]+".png", dpi=300)
    plt.show()
'''


