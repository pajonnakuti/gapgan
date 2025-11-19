# prepare_static_mask.py
import os
import xarray as xr
import numpy as np
import glob
from tqdm import tqdm

# Configuration
INPUT_FOLDER = 'daily_nc_files' # Folder containing your gapped daily files
OUTPUT_FILE = 'geo_mask.nc'

def create_static_mask():
    print("Creating static geographical mask...")
    
    # 1. Load an example file to get dimensions and coordinates
    example_file = glob.glob(os.path.join(INPUT_FOLDER, '*.nc'))
    if not example_file:
        print(f"Error: No .nc files found in {INPUT_FOLDER}. Cannot create mask.")
        return
    
    ds_example = xr.open_dataset(example_file[0])
    
    # 2. Initialize a mask with all zeros (outside valid area)
    # We want 1 for valid ocean/land, 0 for outside
    static_mask = np.zeros_like(ds_example['wind_speed'].values, dtype=np.float32)
    
    # 3. Iterate through all daily files to build a composite mask
    # A pixel is considered "valid" if it has ANY data in ANY of the daily files.
    # This effectively creates a mask of the entire valid geographical area.
    files = sorted(glob.glob(os.path.join(INPUT_FOLDER, '*.nc')))
    for f in tqdm(files, desc="Processing files for static mask"):
        ds = xr.open_dataset(f)
        data = ds['wind_speed'].values
        valid_pixels = ~np.isnan(data)
        static_mask[valid_pixels] = 1.0 # Mark as valid if data exists here
        ds.close() # Close dataset to free up memory

    # 4. Create an xarray Dataset for the mask
    ds_mask = xr.Dataset(
        data_vars={'geo_mask': (('lat', 'lon'), static_mask)},
        coords={'lat': ds_example['lat'], 'lon': ds_example['lon']}
    )
    
    # 5. Save the mask
    ds_mask.to_netcdf(OUTPUT_FILE)
    print(f"Static geographical mask saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    create_static_mask()