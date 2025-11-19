import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import os

# --- 1. CONFIGURATION ---
# --- Change this to the file you want to view ---
NC_FILE_TO_VIEW = 'filled_all_days/scatterometer_2023002.nc' 
# NC_FILE_TO_VIEW = './filled_daily_files_v2/scatterometer_2023002.nc'
# NC_FILE_TO_VIEW = 'master_filled_composite.nc'

VARIABLE_NAME = 'wind_speed' # This is the variable inside the .nc file

def visualize_nc_file():
    if not os.path.exists(NC_FILE_TO_VIEW):
        print(f"Error: File not found: {NC_FILE_TO_VIEW}")
        return

    # 1. Open the NetCDF file
    ds = xr.open_dataset(NC_FILE_TO_VIEW)
    
    if VARIABLE_NAME not in ds:
        print(f"Error: Variable '{VARIABLE_NAME}' not in this file.")
        print(f"Available variables are: {list(ds.data_vars)}")
        return

    wind_data = ds[VARIABLE_NAME]
    
    # 2. Fix Scaling (if data is 3000 instead of 30)
    if wind_data.max() > 100:
        print("Data is scaled. Dividing by 100.")
        wind_data = wind_data * 0.01

    # 3. Setup Plot
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # 4. Add Map Features (White Land & Black Coastlines)
    ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='black', linewidth=0.8, zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black', zorder=2)
    
    # 5. Plot the Wind Data
    wind_data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='jet',                # Use 'jet' colormap
        vmin=0, vmax=20,           # Set color limits
        add_colorbar=True,
        cbar_kwargs={'label': 'Wind Speed (m/s)', 'pad': 0.02}
    )
    
    # 6. Set Map Extent (Zoom)
    ax.set_extent([30, 100, -10, 30], crs=ccrs.PlateCarree())

    # 7. Set Specific Axis Ticks (Labels)
    gl = ax.gridlines(draw_labels=True, linewidth=0, color='none')
    gl.top_labels = False
    gl.right_labels = False
    
    # Force the exact ticks you want
    gl.xlocator = mticker.FixedLocator([30, 40, 50, 60, 70, 80, 90, 100])
    gl.ylocator = mticker.FixedLocator([-10, -5, 0, 5, 10, 15, 20, 25, 30])
    
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}

    plt.title(f"Viewer: {os.path.basename(NC_FILE_TO_VIEW)}")
    plt.show()

if __name__ == "__main__":
    visualize_nc_file()