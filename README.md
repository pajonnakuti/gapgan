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

Applications

High-resolution ocean surface analysis

Climate and monsoon studies

Potential Fishing Zone (PFZ) support

Environmental monitoring and forecasting

Research in remote sensing and oceanography

Citation

If you use this work in your research, please cite:

Kandula Sohan et al., SRGAN-Based Gap Filling of Wind Scatterometry and Super-Resolution Enhancement of Satellite SST for High-Resolution Ocean Analysis, INCOIS Internship Project, 2025.

License

MIT License
