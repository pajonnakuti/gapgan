import os
import glob
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models # For VGG
from tqdm import tqdm

# ================= CONFIGURATION =================
MASTER_FILE = 'master_filled_composite.nc' 
CHECKPOINT_DIR = 'checkpoints' # New folder for this new model
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

RESUME_FILE = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pth')

BATCH_SIZE = 16   
PATCH_SIZE = 64
EPOCHS = 530
LR = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 1. THE DATASET =================
# (This is the correct dataset, no changes)
class MasterInpaintingDataset(Dataset):
    def __init__(self, master_file, patch_size=64, samples_per_epoch=10000):
        ds_master = xr.open_dataset(master_file)
        self.data = ds_master['wind_speed'].values
        
        if np.nanmax(self.data) > 100: self.data *= 0.01
        self.data = np.nan_to_num(self.data, nan=0.0)
        self.max_val = 30.0
        self.data = np.clip(self.data, 0, self.max_val) / self.max_val
        
        self.h, self.w = self.data.shape
        self.patch_size = patch_size
        self.samples = samples_per_epoch

    def __len__(self): return self.samples

    def __getitem__(self, idx):
        y = np.random.randint(0, self.h - self.patch_size)
        x = np.random.randint(0, self.w - self.patch_size)
        target_patch = self.data[y:y+self.patch_size, x:x+self.patch_size]
        
        mask = np.ones_like(target_patch)
        hole_h = np.random.randint(self.patch_size // 4, self.patch_size // 2)
        hole_w = np.random.randint(self.patch_size // 4, self.patch_size // 2)
        y1 = np.random.randint(self.patch_size // 4, self.patch_size - hole_h)
        x1 = np.random.randint(self.patch_size // 4, self.patch_size - hole_w)
        mask[y1 : y1 + hole_h, x1 : x1 + hole_w] = 0.0
        
        input_patch = target_patch * mask
        
        input_tensor = torch.tensor(input_patch).float().unsqueeze(0)
        mask_tensor = torch.tensor(mask).float().unsqueeze(0)
        target_tensor = torch.tensor(target_patch).float().unsqueeze(0)
        
        return input_tensor, mask_tensor, target_tensor

# ================= 2. MODELS =================
# (Generator and Discriminator are unchanged)
class GeneratorUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def down(in_c, out_c): return nn.Sequential(nn.Conv2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2))
        def up(in_c, out_c): return nn.Sequential(nn.ConvTranspose2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.ReLU())
        self.d1 = down(2, 64); self.d2 = down(64, 128); self.d3 = down(128, 256); self.d4 = down(256, 512)
        self.u1 = up(512, 256); self.u2 = up(256+256, 128); self.u3 = up(128+128, 64)
        self.final = nn.Sequential(nn.ConvTranspose2d(64+64, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3)
        u1 = self.u1(d4); u2 = self.u2(torch.cat([F.interpolate(u1, size=d3.shape[2:]), d3], 1))
        u3 = self.u3(torch.cat([F.interpolate(u2, size=d2.shape[2:]), d2], 1))
        out = self.final(torch.cat([F.interpolate(u3, size=d1.shape[2:]), d1], 1))
        return F.interpolate(out, size=img.shape[2:])

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c): return nn.Sequential(nn.Conv2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2))
        self.model = nn.Sequential(block(1, 64), block(64, 128), block(128, 256), nn.Conv2d(256, 1, 4, 1, 0), nn.Sigmoid())
    def forward(self, x): return self.model(x)

# ================= 3. VGG PERCEPTUAL LOSS =================
# (Unchanged)
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.features = nn.Sequential(*[vgg19[i] for i in range(9)]).eval().to(DEVICE)
        for param in self.features.parameters():
            param.requires_grad = False 

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.features(x)

# ================= 4. SAVE / LOAD FUNCTIONS =================
# (Unchanged)
def save_checkpoint(epoch, G, D, opt_G, opt_D, path):
    print(f"\n--- Saving checkpoint for epoch {epoch+1} to {path} ---")
    torch.save({'epoch': epoch, 'G_state_dict': G.state_dict(), 'D_state_dict': D.state_dict(),
                'opt_G_state_dict': opt_G.state_dict(), 'opt_D_state_dict': opt_D.state_dict()}, path)

def load_checkpoint(path, G, D, opt_G, opt_D):
    if os.path.isfile(path):
        print(f"Resuming training from {path}...")
        checkpoint = torch.load(path, map_location=DEVICE)
        G.load_state_dict(checkpoint['G_state_dict'])
        D.load_state_dict(checkpoint['D_state_dict'])
        opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
        opt_D.load_state_dict(checkpoint['opt_D_state_dict'])
        return checkpoint['epoch'] + 1
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0

# ================= 5. TRAINING LOOP (With HYBRID Loss) =================
def train_gan():
    print(f"\n--- 🚀 STARTING HYBRID-LOSS TRAINING ON: {DEVICE} 🚀 ---\n")
    
    dataset = MasterInpaintingDataset(MASTER_FILE, patch_size=PATCH_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    G = GeneratorUNet().to(DEVICE)
    D = Discriminator().to(DEVICE)
    VGG = VGGFeatureExtractor().to(DEVICE)
    
    opt_G = optim.Adam(G.parameters(), lr=LR)
    opt_D = optim.Adam(D.parameters(), lr=LR)
    
    bce_loss = nn.BCELoss() # For GAN loss
    l1_loss = nn.L1Loss()  # For Pixel AND Perceptual loss

    start_epoch = load_checkpoint(RESUME_FILE, G, D, opt_G, opt_D)
    last_saved_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, EPOCHS):
            loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
            
            for masked_imgs, masks, real_imgs in loop:
                masked_imgs, masks, real_imgs = masked_imgs.to(DEVICE), masks.to(DEVICE), real_imgs.to(DEVICE)
                
                # --- Train Generator ---
                fake_imgs = G(masked_imgs, masks)
                
                # 1. Adversarial Loss (The "Warden")
                g_pred = D(fake_imgs)
                g_adv_loss = bce_loss(g_pred, torch.ones_like(g_pred))
                
                # 2. Perceptual Loss (The "Artist")
                fake_features = VGG(fake_imgs)
                real_features = VGG(real_imgs)
                g_perceptual_loss = l1_loss(fake_features, real_features)
                
                # 3. L1 Pixel Loss (The "Mathematician")
                #    We re-introduce this to fight the blur!
                g_pixel_loss = l1_loss(fake_imgs, real_imgs)
                
                # --- NEW HYBRID LOSS ---
                # This is the industry-standard "Inpainting Loss"
                # We balance all three judges.
                g_loss = (
                    0.1 * g_adv_loss +    # 10% "Look real"
                    1.0 * g_perceptual_loss + # 100% "Have the right style"
                    10.0 * g_pixel_loss      # 1000% "Be mathematically accurate"
                )
                
                opt_G.zero_grad(); g_loss.backward(); opt_G.step()
                
                # --- Train Discriminator ---
                real_pred = D(real_imgs)
                fake_pred = D(fake_imgs.detach())
                d_loss = (bce_loss(real_pred, torch.ones_like(real_pred)) + bce_loss(fake_pred, torch.zeros_like(fake_pred))) / 2
                
                opt_D.zero_grad(); d_loss.backward(); opt_D.step()
                
                loop.set_postfix({"G_Loss": g_loss.item(), "D_Loss": d_loss.item()})

            if (epoch + 1) % 50 == 0 or (epoch + 1) == EPOCHS:
                save_checkpoint(epoch, G, D, opt_G, opt_D, RESUME_FILE)
                last_saved_epoch = epoch

    except KeyboardInterrupt:
        print("\n\nTraining Stopped Manually (Ctrl+C).")
        if last_saved_epoch != epoch:
            save_checkpoint(epoch, G, D, opt_G, opt_D, RESUME_FILE)

    if (EPOCHS - 1) > last_saved_epoch:
         print(f"Training Complete! Saving final model...")
         save_checkpoint(EPOCHS - 1, G, D, opt_G, opt_D, RESUME_FILE)

if __name__ == "__main__":
    train_gan()