# fill_all_days_v2.py
import os
import glob
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FOLDER = 'daily_nc_files' 
OUTPUT_FOLDER = 'filled_all_days' # New output folder for new results
MODEL_PATH = 'checkpoints/latest_checkpoint.pth'  
GEO_MASK_FILE = 'geo_mask.nc' # Path to the static geographical mask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- RE-DEFINE MODEL ARCHITECTURE (Must match training script) ---
class GeneratorUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def down(in_c, out_c): return nn.Sequential(nn.Conv2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2))
        def up(in_c, out_c): return nn.Sequential(nn.ConvTranspose2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.ReLU())
        
        self.d1 = down(2, 64)
        self.d2 = down(64, 128)
        self.d3 = down(128, 256)
        self.d4 = down(256, 512)
        self.u1 = up(512, 256)
        self.u2 = up(256+256, 128)
        self.u3 = up(128+128, 64)
        self.final = nn.Sequential(nn.ConvTranspose2d(64+64, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3)
        u1 = self.u1(d4)
        if u1.size() != d3.size(): u1 = F.interpolate(u1, size=d3.shape[2:])
        u2 = self.u2(torch.cat([u1, d3], 1))
        if u2.size() != d2.size(): u2 = F.interpolate(u2, size=d2.shape[2:])
        u3 = self.u3(torch.cat([u2, d2], 1))
        if u3.size() != d1.size(): u3 = F.interpolate(u3, size=d1.shape[2:])
        out = self.final(torch.cat([u3, d1], 1))
        if out.size() != img.size(): out = F.interpolate(out, size=img.shape[2:])
        return out

def fill_every_day_v2():
    # 1. Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run the training script first to create the model.")
        return

    # 2. Load Model
    print(f"Loading Model from {MODEL_PATH}...")
    G = GeneratorUNet().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    G.load_state_dict(checkpoint['G_state_dict'])
    G.eval() 

    # 3. Load Static Geographical Mask
    if not os.path.exists(GEO_MASK_FILE):
        print(f"Error: Geographical mask not found at {GEO_MASK_FILE}")
        print("Please run 'prepare_static_mask.py' first to create it.")
        return
    ds_geo_mask = xr.open_dataset(GEO_MASK_FILE)
    geo_mask_np = ds_geo_mask['geo_mask'].values
    print(f"Loaded geographical mask from {GEO_MASK_FILE}.")

    # 4. Get all Daily Files
    files = sorted(glob.glob(os.path.join(INPUT_FOLDER, '*.nc')))
    if not files:
        print(f"Error: No .nc files found in {INPUT_FOLDER}")
        return
    print(f"Found {len(files)} daily files to fill.")

    # 5. Loop
    for f in tqdm(files):
        try:
            # A. Load Data
            ds = xr.open_dataset(f)
            raw_data = ds['wind_speed'].values
            
            # B. Preprocess
            if np.nanmax(raw_data) > 100: 
                raw_data = raw_data * 0.01
                
            # C. Create mask for GAN input (1=data, 0=gap/outside)
            mask_np = (~np.isnan(raw_data)).astype(np.float32)
            input_np = np.nan_to_num(raw_data, nan=0.0) # Fill NaNs with 0 for GAN
            norm_input = np.clip(input_np, 0, 30) / 30.0
            
            # D. Convert to Tensor
            img_t = torch.tensor(norm_input).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            mask_t = torch.tensor(mask_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            # E. GAN Inference
            with torch.no_grad():
                pred_t = G(img_t, mask_t)
            
            # F. Post-Process
            pred_np = pred_t.squeeze().cpu().numpy()
            pred_scaled = pred_np * 30.0 
            
            # G. Combine (Keep Real Data, Fill Gaps with GAN)
            final_filled = raw_data.copy()
            gap_indices = np.isnan(raw_data)
            final_filled[gap_indices] = pred_scaled[gap_indices]
            
            # H. Apply the geographical mask: set outside-map areas to NaN
            final_filled[geo_mask_np == 0] = np.nan 
            
            # I. Save
            output_filename = os.path.join(OUTPUT_FOLDER, os.path.basename(f))
            ds_out = xr.Dataset(
                data_vars={'wind_speed': (('lat', 'lon'), final_filled)},
                coords={'lat': ds['lat'], 'lon': ds['lon']}
            )
            ds_out.to_netcdf(output_filename)
            
        except Exception as e:
            print(f"Error processing {f}: {e}")

    print("All files filled successfully!")

if __name__ == "__main__":
    fill_every_day_v2()