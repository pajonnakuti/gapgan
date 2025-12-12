import os
import sys
import time

# --- Setup: Import Local DINCAE Module ---
# This ensures the script finds the 'DINCAE' folder in your repository
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DINCAE_PATH = os.path.join(CURRENT_DIR, "DINCAE")

# Append the parent directory to path so we can import DINCAE as a module
if DINCAE_PATH not in sys.path:
    sys.path.append(os.path.dirname(DINCAE_PATH))

try:
    import DINCAE
except ImportError:
    print(f"CRITICAL ERROR: Could not import DINCAE. Please check that the 'DINCAE' folder exists at: '{DINCAE_PATH}'")
    sys.exit(1)

# --- 1. Configuration: File Paths ---

# sector and file name here for easy changing
SECTOR_FOLDER = "3_Goa"
NC_FILE_NAME = "3R_3_SST_DATA.nc"

# Construct relative paths based on the Git repository structure
# Expected Data Path: ./model_data/3_Goa/3R_3_SST_DATA.nc
BASE_DATA_PATH = os.path.join(CURRENT_DIR, "model_data", SECTOR_FOLDER)

# The location of your NetCDF data file
FILENAME = os.path.join(BASE_DATA_PATH, NC_FILE_NAME)

# The variable name inside the NetCDF file to use
VARNAME = 'SST'

# The directory where model checkpoints and outputs will be saved
# We save it back into the same sector folder in 'model_data'
OUTDIR = BASE_DATA_PATH

#if you want to resume training from pretrained checkpoints
LOAD_MODEL_CHECKPOINT = None

# --- 2. Training Hyperparameters ---
# Total number of training cycles to run
EPOCHS = 1002
# How fast the model learns. 1e-4 is a good starting point.
LEARNING_RATE = 1e-4
# How often to save a reconstructed .nc file (e.g., every 50 epochs)
SAVE_EACH = 200
# How often to save a model checkpoint .ckpt file (e.g., every 100 epochs)
SAVE_MODEL_EACH = 200                                    

# --- 3. Model Architecture ---
# These parameters MUST be the same when you run inference later.
BATCH_SIZE = 16

NTIME_WIN = 3
ENC_NFILTER_INTERNAL = [16, 32, 64, 128]
SKIP_CONNECTIONS = [1, 2, 3, 4]
FRAC_DENSE_LAYER = [0.2]

# --- Regularization and augmentation ---
JITTER_STD = 0.05
# A technique to prevent overfitting
DROPOUT_RATE_TRAIN = 0.3

start_time = time.time()

# --- Main Execution ---
def main():
    # 1. Validation Check
    if not os.path.exists(FILENAME):
        print(f"\n❌ ERROR: Training file not found.")
        print(f"   Looked for: {FILENAME}")
        print(f"   Please make sure you have placed '{NC_FILE_NAME}' inside the 'model_data/{SECTOR_FOLDER}/' folder.\n")
        return

    if LOAD_MODEL_CHECKPOINT:
        print(f'Loading the model from {LOAD_MODEL_CHECKPOINT}')
    else:
        print("Starting new training run")

    """Sets up and starts the DINCAE training process."""
    print("--- Starting New Training Run ---")
    print(f"Input data: {FILENAME}")
    print(f"Output directory: {OUTDIR}")
    print(f"Total epochs: {EPOCHS}")
    print("-" * 35)

    # Ensure the output directory exists
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
        print(f"Created output directory: {OUTDIR}")

    # We call the main training function from the DINCAE library.
    # The `load_model` parameter is intentionally left out to start training from scratch.
    DINCAE.reconstruct_gridded_nc(
        filename=FILENAME,
        varname=VARNAME,
        outdir=OUTDIR,

        load_model=LOAD_MODEL_CHECKPOINT,
        
        # Pass training hyperparameters
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        save_each=SAVE_EACH,
        save_model_each=SAVE_MODEL_EACH,

        # clip_grad = 1.0,
        
        # Pass model architecture
        ntime_win=NTIME_WIN,
        enc_nfilter_internal=ENC_NFILTER_INTERNAL,
        skipconnections=SKIP_CONNECTIONS,
        frac_dense_layer=FRAC_DENSE_LAYER, 

        # regularization and augmentation 
        # jitter_std=JITTER_STD,
        # dropout_rate_train=DROPOUT_RATE_TRAIN
    )

    print("--- Training Finished ---")
    end_time = time.time()
    duration = end_time - start_time
    minutes = duration / 60
    seconds = duration % 60
    print(f'The duration of the Training: {minutes:.1f} minutes and {seconds:.1f} seconds')

# This makes the script runnable from the command line
if __name__ == '__main__':
    main()