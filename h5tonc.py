import h5py
import numpy as np
import xarray as xr
import os
import glob
import re
from collections import defaultdict

# --- CONFIGURATION ---
INPUT_FOLDER = 'D:/Satellite_Data/Scatterometer Data' 
OUTPUT_FOLDER = 'daily_nc_files.py'
MIN_LAT, MAX_LAT = -30, 30
MIN_LON, MAX_LON = 10, 100
GRID_RES = 0.1 

def create_daily_composites():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    files_by_day = defaultdict(list)
    all_files = glob.glob(os.path.join(INPUT_FOLDER, '*.h5'))
    
    # Regex to find date (YYYYDDD)
    date_pattern = re.compile(r'(20\d{2})(\d{3})') 

    print("Grouping files...")
    for f in all_files:
        match = date_pattern.search(os.path.basename(f))
        if match:
            files_by_day[match.group(0)].append(f)

    sorted_days = sorted(files_by_day.keys())
    print(f"Found {len(sorted_days)} days.")

    # Grid setup
    grid_lats = np.arange(MIN_LAT, MAX_LAT, GRID_RES)
    grid_lons = np.arange(MIN_LON, MAX_LON, GRID_RES)
    grid_shape = (len(grid_lats), len(grid_lons))

    for day in sorted_days:
        day_files = files_by_day[day]
        
        # Create empty grid (NaNs)
        daily_grid = np.full(grid_shape, np.nan)
        data_found_for_day = False # Flag to track if we found ANY data

        for file_path in day_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'science_data/Latitude' not in f: continue
                    
                    lats = f['science_data/Latitude'][:]
                    lons = f['science_data/Longitude'][:]
                    
                    # --- SCALING CHECK (Auto-Fix) ---
                    # If lats are integers like 3000, divide by 100
                    if np.max(np.abs(lats)) > 90: 
                        lats = lats * 0.01
                    if np.max(np.abs(lons)) > 360:
                        lons = lons * 0.01

                    # Select best wind variable
                    if 'science_data/Wind_speed_selection' in f:
                        wind = f['science_data/Wind_speed_selection'][:]
                    elif 'science_data/Rain_Corrected_Wind_Speed' in f:
                        wind = f['science_data/Rain_Corrected_Wind_Speed'][:]
                    else:
                        wind = f['science_data/Wind_speed'][:, :, 0]

                    # MASKING
                    mask = (lats >= MIN_LAT) & (lats < MAX_LAT) & \
                           (lons >= MIN_LON) & (lons < MAX_LON)
                    
                    if np.sum(mask) == 0: continue

                    # If we get here, we found data!
                    data_found_for_day = True
                    
                    lats_c = lats[mask]
                    lons_c = lons[mask]
                    wind_c = wind[mask]

                    lat_idx = ((lats_c - MIN_LAT) / GRID_RES).astype(int)
                    lon_idx = ((lons_c - MIN_LON) / GRID_RES).astype(int)
                    
                    lat_idx = np.clip(lat_idx, 0, grid_shape[0] - 1)
                    lon_idx = np.clip(lon_idx, 0, grid_shape[1] - 1)

                    daily_grid[lat_idx, lon_idx] = wind_c

            except Exception as e:
                print(f"Error on file: {e}")

        # --- FINAL CHECK BEFORE SAVING ---
        # "Why process them if not in desired region?" -> We won't save them.
        if not data_found_for_day or np.all(np.isnan(daily_grid)):
            print(f"Day {day}: No data in Indian Ocean. SKIPPING.")
        else:
            output_name = os.path.join(OUTPUT_FOLDER, f"scatterometer_{day}.nc")
            ds = xr.Dataset(
                data_vars={'wind_speed': (('lat', 'lon'), daily_grid)},
                coords={'lat': grid_lats, 'lon': grid_lons}
            )
            ds.to_netcdf(output_name)
            print(f"Day {day}: Success! Saved.")

if __name__ == "__main__":
    create_daily_composites()