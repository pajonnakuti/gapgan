# File: view_mask.py
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import os

# --- CONFIGURATION ---
MASK_FILE = 'geo_mask.nc' 

def visualize_mask():
    if not os.path.exists(MASK_FILE):
        print(f"File not found: {MASK_FILE}")
        return

    # 1. Open the Data
    ds = xr.open_dataset(MASK_FILE)
    
    # 2. Get the CORRECT variable: 'geo_mask'
    mask_data = ds['geo_mask']

    # 3. Setup Plot
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # 4. Add Land & Coastlines
    ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='black', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black', zorder=2)
    
    # 5. Plot Data
    # We plot the 'geo_mask' variable.
    # It will be 0 (dark blue) for "outside" and 1 (yellow) for "inside"
    mask_data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='viridis', # 'viridis' is good for 0-1 masks
        add_colorbar=True,
        cbar_kwargs={'label': 'Mask Value (1=Valid, 0=Outside)'}
    )
    
    # 6. Set Extents & Ticks
    ax.set_extent([30, 100, -10, 30], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0, color='none')
    gl.top_labels = False; gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([30, 40, 50, 60, 70, 80, 90, 100])
    gl.ylocator = mticker.FixedLocator([-10, -5, 0, 5, 10, 15, 20, 25, 30])
    
    plt.title("Static Geographical Mask", fontweight='bold')
    plt.show()

if __name__ == "__main__":
    visualize_mask()