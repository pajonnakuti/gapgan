import xarray as xr
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- CONFIGURATION ---
INPUT_FOLDER = 'daily_nc_files.py'
OUTPUT_FILE = 'master_filled_composite.nc'

def accumulate_data_to_fill_gaps():
    # 1. Get all files
    files = sorted(glob.glob(os.path.join(INPUT_FOLDER, '*.nc')))
    if not files:
        print("No .nc files found!")
        return

    print(f"Found {len(files)} files to accumulate.")

    # 2. Create the "Duplicate/Dummy Array" (The Master Canvas)
    # We open the first file just to get the shape and coordinates
    first_ds = xr.open_dataset(files[0])
    
    # Create a master array filled with NaNs (Empty)
    master_grid = np.full_like(first_ds['wind_speed'].values, np.nan)
    lats = first_ds['lat'].values
    lons = first_ds['lon'].values

    # 3. The Loop: Fill the Gaps
    for i, f in enumerate(files):
        ds = xr.open_dataset(f)
        day_data = ds['wind_speed'].values

        # LOGIC: "Where is the Master empty AND the New Day has data?"
        # mask_fill = (Master is NaN) AND (New Data is NOT NaN)
        mask_fill = np.isnan(master_grid) & ~np.isnan(day_data)
        
        # Fill those specific gaps
        master_grid[mask_fill] = day_data[mask_fill]
        
        # Calculate how full the array is
        percent_full = (np.sum(~np.isnan(master_grid)) / master_grid.size) * 100
        print(f"Processed {os.path.basename(f)} -> Master Array is {percent_full:.2f}% full")

        # Stop if 100% full (optional)
        if percent_full >= 99.9:
            print("Array is fully filled! Stopping early.")
            break

    # 4. Save the Result
    print("Saving the Master Filled Map...")
    ds_out = xr.Dataset(
        data_vars={'wind_speed': (('lat', 'lon'), master_grid)},
        coords={'lat': lats, 'lon': lons}
    )
    ds_out.to_netcdf(OUTPUT_FILE)
    
    # 5. Plot to Verify
    plot_result(ds_out)

def plot_result(ds):
    # Quick visualization logic (simplified from previous chat)
    data = ds['wind_speed']
    if data.max() > 100: data = data * 0.01
    
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='black')
    ax.add_feature(cfeature.COASTLINE)
    
    data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='jet', vmin=0, vmax=20, cbar_kwargs={'label': ''})
    ax.set_extent([30, 100, -10, 30])
    plt.title("Master Filled Array (Accumulated)")
    plt.show()

if __name__ == "__main__":
    accumulate_data_to_fill_gaps()