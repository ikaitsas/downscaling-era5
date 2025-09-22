# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 16:34:11 2025

@author: Giannis


Resulting filename convention:
    DEM-tile-{NORTH-WEST EDGE LATITUDE}-{NORTH-WEST EDGE LONGITUDE}.tif
"""

import os
import ee
import requests
from numpy import arange

# Initialize the Earth Engine module
ee.Authenticate()
ee.Initialize(project="downscaling-464711")


band = "DEM"  # water body mask: "WBM"
directory_path = "tiles-GLO90"
os.makedirs(directory_path, exist_ok=True)


# Define your area of interest (AOI) - MUST BE FLOAT
# 18.0, 34.0, 31.0, 43.0 
west_bound, south_bound, east_bound, north_bound = 18.0, 34.0, 31.0, 43.0

resolution = 3*0.5  # small so that GEE accepts the request (1 too big)
scale_gee = 3*30.86  # GEE sampling forresulting raster


# Define the overall AOI for filtering the collection just once for efficiency
overall_aoi = ee.Geometry.Rectangle(
    [west_bound, south_bound, east_bound, north_bound]
    )

# Load and filter the COPERNICUS DEM GLO-30 collection once outside the loop
# This ensures you're only dealing with relevant images.
# Then, mosaic them to create a single, continuous DEM image for your entire region.
collection = ee.ImageCollection("COPERNICUS/DEM/GLO30") \
    .filterBounds(overall_aoi) # Filter the collection to only include images that intersect your overall AOI

# Mosaic the filtered collection once. This single mosaicked image covers your entire AOI.
# You will then clip portions of this *mosaicked* image for each download tile.
image = collection.mosaic().select(band)

# List to hold the chunks
rectangles = []

# Loop from west to east in steps of 2
for lon in arange(west_bound, east_bound, resolution):
    # Loop from south to north in steps of 2
    for lat in arange(north_bound, south_bound, -resolution):
        # Define chunk bounds: [W, S, E, N]
        rectangle = [lon, lat - resolution, lon + resolution, lat]
        rectangles.append(rectangle)
        
        # Define AOI for the current chunk
        aoi_chunk = ee.Geometry.Rectangle(rectangle) # W-S-E-N
        
        # Clip the *overall mosaicked DEM image* to the current chunk's AOI
        # This is the key change to get actual data for the tile.
        image_to_download = image.clip(aoi_chunk)

        # Construct the file name - North and West point of each tile
        file_name = f"DEM-tile-{rectangle[3]}-{rectangle[0]}.tif" 
        file_path = os.path.join(directory_path, file_name)

        # Prepare download URL parameters
        download_params = {
            'scale': scale_gee, # GLO-30 is 30m resolution
            'crs': 'EPSG:4326',
            'region': aoi_chunk.getInfo()['coordinates'], # Pass coordinates from the GEE object
            'format': 'GeoTIFF',
            'bands': [band] # Explicitly specify the band to download
        }

        url = image_to_download.getDownloadURL(download_params)
        
        print(f"Attempting to download {file_name} from URL: {url}")
        
        try:
            # Use stream=True for potentially large files and iterate in chunks
            response = requests.get(url)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            if not response.content:
                print(f"Warning: Downloaded content for {file_name} is empty. Check AOI coverage or GEE data availability.")
                continue # Skip to the next tile if content is empty

            with open(file_path, 'wb') as file:
                    file.write(response.content)
            print(f"Successfully downloaded DEM for {file_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {file_name}: {e}")
            print(f"URL that failed: {url}") # Print URL on error for debugging
        except Exception as e:
            print(f"An unexpected error occurred for {file_name}: {e}")
        
        '''
        #%% Google drive shenanigans
        file_name = f"DEM-tile-{rectangle[3]}-{rectangle[0]}"
        # Set export parameters
        task_dem = ee.batch.Export.image.toDrive(
            image=image,
            description=file_name,
            folder='EarthEngineExports',
            fileNamePrefix=file_name,
            region=aoi,
            scale=30,
            crs='EPSG:4326',
            maxPixels=1e10
        )
        
        # Start the export task
        task_dem.start()
        print(f"Exporting {file_name}")
        '''
        

#%% test
import rasterio
import matplotlib.pyplot as plt

demaki = rasterio.open(
    os.path.join(directory_path, "DEM-tile-41.0-18.0.tif"), 
    )

arr = demaki.read(1)
#plt.imshow(arr[1300:1400,1600:1700],cmap='magma')
#plt.imshow(arr[900:1000,1300:1400],cmap='magma')
plt.imshow(arr,cmap='magma')
plt.title(f"GEE Sampling: {scale_gee} m")
plt.yticks([])
plt.xticks([])
#plt.savefig(f"tile-{scale_gee}m.png", dpi=700)
plt.show()


        