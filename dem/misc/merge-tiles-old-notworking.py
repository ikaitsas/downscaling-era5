# -*- coding: utf-8 -*-
"""
Created on Mon Jul 02 18:24:43 2025

@author: Giannis

Make sure the BATCH_SIZE value matches the total_lwidth/tile_width for 
the merging, as non-matching causes tiles to not be passed to the output
e.g. for merging 0.5 thick tiles for the area 18.0 to 30.5 (including the
31.0 edge), it produces a width of: (31-18)/0.5=26 
For this reason, BATCH_SIZE is 26 in this case

BATCH_SIZE auto detection requires the east and west bounds of the total area,
the width of each tile in degrees
This code merges these tiles, and ensures that their resolution is the native
1 arcsecond through some minor resampling, in case the resolution of the 
merged .tif file does not match the resolution
"""

import os
import glob
import subprocess

dem_dump_path = "tiles-GLO60"
tile_width_degrees = 1  # 0.5 for 1arcsec tiles, 1.0 for 2arcsec tiles

output_tif = "output-GLO60.tif"
# Define the target 1 arcsecond resolution in degrees
TARGET_RESOLUTON_DEG = 2*0.0002777777777777777777778

BATCH_SIZE = 26  # MAKE SURE THIS MATCHES HORIZONTAL RESOLUTION
BATCH_SIZE_SELECTION = "from-bounds-and-degree-width"
tif_files = glob.glob(os.path.join(dem_dump_path, "*.tif"))
# mediorce try to only track DEM tiles, not outputs
tif_files = [
    file for file in tif_files 
    if os.path.basename(file).startswith('D')
    ]

# Needed for batch size for the .vrt file
if BATCH_SIZE_SELECTION == "from-bounds-and-degree-width":
    west_bound, south_bound, east_bound, north_bound = 18.0, 34.0, 31.0, 43.0
    BATCH_SIZE = int( (east_bound-west_bound)/tile_width_degrees )




if not tif_files:
    print("No .tif files found in the directory.")
    
else:
    batch_vrt_files = []
    for i in range(0, len(tif_files), BATCH_SIZE):
        batch_files = tif_files[i:i + BATCH_SIZE]
        batch_output_vrt = os.path.join(dem_dump_path, f"output_batch_{i//BATCH_SIZE}.vrt")
        batch_vrt_files.append(batch_output_vrt)

        print(f'Building Virtual Dataset for batch {i//BATCH_SIZE}...')
        cmd_virtual = ["gdalbuildvrt", batch_output_vrt] + batch_files

        result_virtual = subprocess.run(cmd_virtual,
                                        capture_output=True,
                                        text=True)
        print(f"gdalbuildvrt Standard Output (batch {i//BATCH_SIZE}):", result_virtual.stdout)
        print(f"gdalbuildvrt Standard Error (batch {i//BATCH_SIZE}):", result_virtual.stderr)
        
        if result_virtual.returncode != 0:
            print(f"Error building VRT for batch {i//BATCH_SIZE}. Exiting.")
            exit()

    # Now, merge the batch VRTs into a final VRT
    print('\nMerging batch VRTs into a final VRT...')
    final_output_vrt = "output_GEE.vrt"
    cmd_virtual_final = ["gdalbuildvrt", final_output_vrt] + batch_vrt_files
    
    result_virtual_final = subprocess.run(cmd_virtual_final,
                                          capture_output=True,
                                          text=True)
    print("gdalbuildvrt Final Standard Output:", result_virtual_final.stdout)
    print("gdalbuildvrt Final Standard Error:", result_virtual_final.stderr)

    if result_virtual_final.returncode != 0:
        print("Error building final VRT. Exiting.")
        exit()
        
        
    # adjust the resolution a bit, as it is it migh tnot exactly be 1 arcsecond
    print(f'\nAdjusting resolution of {final_output_vrt} to 1 arcsecond ({TARGET_RESOLUTON_DEG:.5f} degrees)...')
    # Create a new VRT for the resampled output, also in the current directory
    resampled_output_vrt = "output_resampled.vrt"

    cmd_resample = [
        "gdal_translate",
        "-tr", str(TARGET_RESOLUTON_DEG), str(TARGET_RESOLUTON_DEG), # Target resolution
        "-r", "lanczos", # Resampling method ("nearest","bilinear","cubicspline",etc)
        final_output_vrt,  # Input VRT (the one just created)
        resampled_output_vrt # Output VRT (with adjusted resolution)
    ]

    result_resample = subprocess.run(cmd_resample,
                                     capture_output=True,
                                     text=True)
    print("gdal_translate Resample Output:", result_resample.stdout)
    print("gdal_translate Resample Error:", result_resample.stderr)

    if result_resample.returncode != 0:
        print("Error resampling VRT. Exiting.")
        exit()
    else:
        print(f"Successfully created resampled VRT: {resampled_output_vrt}")
    
    
    # Translate the final VRT to a .tif file
    print('\nTranslating to a .tif file from the final VRT...')

    cmd_translate = ["gdal_translate", resampled_output_vrt, output_tif]

    result_translate = subprocess.run(cmd_translate,
                                      capture_output=True,
                                      text=True)
    print("gdal_translate Output:", result_translate.stdout)
    print("gdal_translate Error:", result_translate.stderr)
    
    if result_translate.returncode != 0:
        print("Error translating final VRT to TIFF. Exiting.")
        exit()
    else:
        print(f"\nSuccessfully created {output_tif}")


#%%
import rasterio
import matplotlib.pyplot as plt

array = rasterio.open(output_tif).read(1)

for i in range(0,4):
    for j in range(0,7):
        print(i,j)
        plt.imshow(array[4050*i:4050*(i+1), 3343*j:3343*(j+1)])        
        plt.show()