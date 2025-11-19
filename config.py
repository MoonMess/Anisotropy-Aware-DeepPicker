import torch
import os

# --- Data and Path Parameters ---
is_local = os.path.exists("/home/momessao/")
if is_local:
    # Path to the output folder of prepare_czi_data.py
    CZI_DATA_ROOT = "/home/momessao/Documents/data/czii-cryo-et-object-identification/" # Training data
    # Path to the public CZI evaluation dataset (after preparation)
    CZI_PUBLIC_EVAL_DATA_ROOT = "/home/momessao/Documents/data/czii-cryo-et-object-identification/public_sample"
    # NEW: Root path for checkpoints, logs, and results
    OUTPUT_ROOT = "/home/momessao/Documents/code/deepfinders/deepfinder2.1/"
else:
    # Adapt this path for your server environment
    CZI_DATA_ROOT = "../../storage/data/czii-cryo-et-object-identification/" # Training data
    # Path to the public CZI evaluation dataset (after preparation)
    CZI_PUBLIC_EVAL_DATA_ROOT = "../../storage/data/czi/public_dataset/10445/"
    # NEW: Root path for checkpoints, logs, and results
    OUTPUT_ROOT = "../../storage/outputs/deepfinder2.1/"

# Path to the output folder of `prepare_mask_czi_multitask.py`.
# This folder should contain the 'segmentations_...' and 'centroids_...' subfolders.
LABEL_DIR = CZI_DATA_ROOT + "mask/"
RADIUS_FRACTION = 0.5 # Fraction of the particle radius to generate masks.

# Percentiles pour la normalisation globale par défaut
NORM_LOWER_PERCENTILE = 5.0
NORM_UPPER_PERCENTILE = 99.0


# --- Paramètres du Modèle ---
NUM_CLASSES = 6 # Number of particle types (excluding background)
IN_CHANNELS = 1 # Grayscale tomograms

# Class names for logging. The order MUST match the IDs in particle_config.py
CLASS_NAMES = [
    'apo-ferritin',
    'beta-amylase',
    'beta-galactosidase',
    'ribosome',
    'thyroglobulin',
    'virus-like-particle',
]

# Tomogram type to use for VALIDATION.
TOMO_TYPE = "denoised"

# List of tomogram types to use for AUGMENTATION during training.
# The DataLoader will randomly choose from these types for each patch.
TRAINING_TOMO_TYPES = ["denoised", "isonetcorrected"]#, "ctfdeconvolved", "wbp"]

CLASS_LOSS_WEIGHTS = [
    1.0,  # background (classe 0)
    10.0,  # apo-ferritin
    10.0,  # beta-amylase
    20.0,  # beta-galactosidase
    10.0,  # ribosome
    20.0,  # thyroglobulin
    10.0,  # virus-like-particle
]

# --- Paramètres de l'Entraînement ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Training Parameters
LEARNING_RATE = 1e-3 # Initial learning rate.
# Enable/disable Test-Time Augmentation for validation.
USE_TTA_VALIDATION = False
NUM_WORKERS = 4
# Enable/disable Mixup augmentation during training.
USE_MIXUP = False

MIXUP_PROB = 1.0 # Probability of applying Mixup on a batch
MIXUP_BETA = 1.0 # Beta distribution parameter for Mixup, uniform distribution between 0 and 1 if the value is 1

# Default augmentation level. Options: 'none', 'base', 'advanced'
AUGMENTATION_LEVEL = 'base'


# Probability and maximum amplitude for mean/standard deviation augmentation.
# The actual amplitude will be randomly chosen between 0 and this value.
AUG_MEAN_STD_SHIFT_PROB = 0.5
AUG_MEAN_STD_SHIFT_MAX_AMOUNT = 0.5 # e.g., 0.05 for a maximum variation of +/- 5%

# Accumulate gradients over N batches before performing an optimizer step.
# Useful for simulating larger batches on GPUs with limited memory.
# Set to 1 to disable.
GRADIENT_ACCUMULATION_STEPS = 1
# Frequency of logging training metrics (in number of steps/batches).
# Must be less than the number of batches per epoch (TRAIN_STEPS_PER_EPOCH / BATCH_SIZE).
LOG_EVERY_N_STEPS = 10
BATCH_SIZE = 4
# Batch size for sliding window inference during validation.
# A larger batch size speeds up validation if GPU memory allows.
VALIDATION_SW_BATCH_SIZE = 16
EPOCHS = 50
# Number of steps (batches) per training epoch.
# If set to `None`, the number of steps will be equal to the total number of
# annotated particles in the training set, which means that on average,
# each particle is seen once per epoch.
TRAIN_STEPS_PER_EPOCH = None # Set to an integer (e.g., 1000) for a fixed number of steps.
# Validation frequency. A full validation will be performed every N epochs.
# Increasing this value speeds up the overall training.
VALIDATE_EVERY_N_EPOCHS = 1
# Overlap for validation patches (0.5 = 50%, 0.25 = 25%).
# A smaller overlap speeds up validation.
VALIDATION_OVERLAP_FRACTION = 0.5
# The patch size must be a multiple of the model's total reduction factor.
# For ResUNet with 3 pooling levels (stride 2), each dimension must be divisible by 2^3 = 8.
PATCH_SIZE = [128, 128, 128]

# --- CZI Evaluation and Post-Processing Parameters ---
VOXEL_SPACING = 10.012444 # Voxel size in Angstroms
CZI_EVAL_BETA = 4      # Beta parameter for calculating the F-beta score

# --- Parameters for validation during training ---
# Confidence threshold for peak detection. (peak local max only)
VALIDATION_CONF_THRESHOLD = 0.1
# Radius fraction for NMS.
VALIDATION_NMS_RADIUS_FRACTION = 0.8
# Post-processing method for validation ('peak_local_max' or 'cc3d').
# 'peak_local_max' is often faster and sufficient for monitoring.
VALIDATION_POSTPROC_METHOD = 'peak_local_max_gpu'

# --- General Post-Processing Parameters ---
# Default post-processing method for inference and evaluation.
DEFAULT_POSTPROC_METHOD = 'peak_local_max_gpu'
# Downsampling factor of the probability map before detection.
POSTPROC_DOWNSAMPLING_FACTOR = 1

# --- 'cc3d' Method Specific Parameters ---
# Default volume fraction threshold for the connected components size filter.
# A value of 1/7 means a component must have at least 1/7 of the volume
# of a perfect sphere to be considered a detection.
CC3D_VOL_FRAC_THRESHOLD = 1/20
# Search space for volume fraction optimization for cc3d and meanshift methods.
VOL_FRAC_GRID_SEARCH = [1.2, 1, 0.8, 1/3, 1/7, 1/15, 1/20, 1/25, 1/30]
# Connectivity for the cc3d algorithm. 6, 18, or 26.
# 18 is a good compromise to avoid merging particles that only touch at a corner.
CC3D_CONNECTIVITY = 26

# --- Debugging Parameters ---
# Enable debug mode for post-processing (can be overridden by command line).
POSTPROC_DEBUG_MODE = False
POSTPROC_DEBUG_DIR = os.path.join("debug", "postproc_debug")

#model parameters
norm_type = 'batch'

UNET_CHANNELS = (48, 64, 80, 80)
UNET_STRIDES = (2, 2, 2)
UNET_NUM_RES_UNITS = 2

UNET_CHANNELS_DF1 = (32, 48, 64, 64)
UNET_STRIDES_DF1 = (2, 2, 2)
UNET_NUM_RES_UNITS_DF1 = 2

dropout_rate = 0.2 # Dropout in the U-Net model
act = 'leakyrelu' # Activation function

# --- SWA (Stochastic Weight Averaging) Parameters ---
# Learning rate for SWA.
SWA_LRS = 1e-4
SWA_EPOCH_START = 0.5 # Percentage of epochs before starting SWA (e.g., 0.8 for the last 20% of epochs)

# --- EMA (Exponential Moving Average) Parameters ---
EMA_DECAY = 0.999 # Decay factor for EMA

# --- Loss Type ---
# Loss type to use: 'ce' (simple CrossEntropy) or 'focal_tversky' (hybrid).
# 'ce' is now the default.
LOSS_TYPE = 'ce'
# Parameters for Tversky loss (if used)
TVERSKY_ALPHA = 0.3 # Penalizes False Positives (FP)
TVERSKY_BETA = 0.7  # Penalizes False Negatives (FN)

# --- Learning Rate Scheduler Parameters ---
USE_SCHEDULER = True
SCHEDULER_TYPE = 'cosine'  # Choose the scheduler type: 'cosine' or 'plateau'

# Parameters for ReduceLROnPlateau (if SCHEDULER_TYPE = 'plateau')
PLATEAU_PATIENCE = 10  # Patience in epochs. Must be < EARLY_STOPPING_PATIENCE.
PLATEAU_FACTOR = 0.5   # LR reduction factor (new_lr = lr * factor)