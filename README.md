SRGAN-OceanEnhance (SRGAN + Ocean Data)
SRGAN-Based Gap Filling of Wind Scatterometry and Super-Resolution Enhancement of Satellite SST for High-Resolution Ocean Analysis

This repository presents a deep learning–based framework for reconstructing missing wind scatterometry data and enhancing low-resolution Sea Surface Temperature (SST) satellite imagery using Super-Resolution Generative Adversarial Networks (SRGAN).

The project focuses on improving the quality and usability of oceanographic satellite datasets by filling spatial gaps caused by sensor limitations and cloud interference, and by upscaling coarse-resolution SST grids into high-resolution representations suitable for detailed ocean analysis.

Features

SRGAN-based gap filling (inpainting) for wind scatterometry NetCDF datasets.

Super-resolution enhancement of satellite SST images (60×48 → 512×512).

Progressive upscaling pipeline for stable and realistic reconstruction.

Quantitative evaluation using error metrics such as PSNR, SSIM, and RMSE.

End-to-end workflow for preprocessing, model training, inference, and visualization.

Example notebooks for experimentation and demonstration.

Getting Started

Clone the repository:

git clone https://github.com/yourusername/SRGAN-OceanEnhance.git
cd SRGAN-OceanEnhance


Install dependencies:

pip install -r requirements.txt


Run the demo notebooks:

jupyter notebook notebooks/

Directory Structure
data/              # Wind scatterometry and SST satellite datasets (NetCDF / images)
notebooks/         # Jupyter notebooks for training, testing, and visualization
src/               # SRGAN models, preprocessing scripts, and utilities
checkpoints/       # Trained model weights
results/           # Reconstructed and super-resolved outputs
requirements.txt   # Python dependencies
README.md          # Project documentation

Methodology Overview

Wind datasets with missing regions are reconstructed using an SRGAN inpainting framework.

Low-resolution SST datasets are progressively enhanced through multi-stage super-resolution using SRGAN.

The generator learns realistic spatial patterns while the discriminator enforces physical and visual consistency.

Model performance is validated using PSNR, SSIM, and RMSE against reference datasets.

# 📊 GapGAN Model Outputs – Arabian Sea (64x64)

**Region:** Arabian Sea  
**Resolution:** 64 × 64  
**Training:** 300 Epochs  
**Models Compared:** Bicubic Interpolation, SRCNN, SRGAN  

---

# 🌊 1️⃣ Year 2004 – Arabian Sea

<img width="503" height="336" alt="Screenshot 2026-02-17 212535" src="https://github.com/user-attachments/assets/fcc8e7cd-9780-43a6-be79-f9b8cb657112" />


### 📈 SRGAN Performance
- PSNR: 32.15
- SSIM: 0.929

---

# 🌊 2️⃣ Year 2009 – Arabian Sea

<img width="473" height="357" alt="Screenshot 2026-02-17 212547" src="https://github.com/user-attachments/assets/d5fa3fcd-b38e-459b-bec7-9784d9814e16" />



### 📈 SRGAN Performance
- PSNR: 34.12
- SSIM: 0.952

---

# 🌊 3️⃣ Year 2016 – Arabian Sea

<img width="477" height="369" alt="Screenshot 2026-02-17 212602" src="https://github.com/user-attachments/assets/057287b8-0833-49d6-8163-176d514ede0b" />

### 📈 SRGAN Performance
- PSNR: 32.38
- SSIM: 0.932

---

# 🌊 4️⃣ Year 2017 – Arabian Sea

<img width="465" height="339" alt="Screenshot 2026-02-17 212617" src="https://github.com/user-attachments/assets/16609a5a-24e5-40a1-a06e-858b12fdffa6" />



### 📈 SRGAN Performance
- PSNR: 31.05
- SSIM: 0.931

---

# 🌊 5️⃣ Year 2018 – Arabian Sea

<img width="459" height="322" alt="Screenshot 2026-02-17 212626" src="https://github.com/user-attachments/assets/f9b96be8-be0f-464c-be27-9e32166b0eba" />



### 📈 SRGAN Performance
- PSNR: 31.41
- SSIM: 0.931

---

# 🔬 Comparative Analysis

Across multiple years of Arabian Sea scatterometer data:

- SRGAN consistently outperforms Bicubic interpolation.
- Structural similarity (SSIM) remains above 0.92 across all years.
- PSNR values demonstrate stable high-quality reconstruction.
- Spatial continuity and oceanic structure patterns are preserved.
- The model generalizes effectively across temporal variations.

---

# 🚀 Conclusion

The GapGAN (SRGAN-based) model demonstrates strong robustness and consistency in reconstructing missing satellite



# 🚀 Applications

- High-resolution ocean surface analysis

- Climate and monsoon studies

- Potential Fishing Zone (PFZ) support

- Environmental monitoring and forecasting

- Research in remote sensing and oceanography

# Citation

If you use this work in your research, please cite:

Kandula Sohan et al., SRGAN-Based Gap Filling of Wind Scatterometry and Super-Resolution Enhancement of Satellite SST for High-Resolution Ocean Analysis, INCOIS Internship Project, 2025.

# License

MIT License
