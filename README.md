
# ETFinder: Anisotropic-Aware Test-Time Augmentation for Cryo-ET Particle Detection

This repository contains the code for **ETFinder**, a novel approach for detecting macromolecular particles in cryo-electron tomograms (cryo-ET). ETFinder enhances a standard U-Net baseline by incorporating an **anisotropic-aware Test-Time Augmentation (TTA)** strategy, which significantly improves detection accuracy by accounting for the inherent anisotropy of cryo-ET data.

The project is designed to facilitate research in cryo-ET, particularly within the context of challenges like the CZI Cryo-ET Object Identification Challenge.

## Table of Contents

1.  [Getting Started](#1-getting-started)
2.  [Setup and Installation](#2-setup-and-installation)
3.  [Data Preparation](#3-data-preparation)
    *   [Generate Training Segmentation Masks](#31-generate-training-segmentation-masks)
    *   [Prepare Public Evaluation Annotations](#32-prepare-public-evaluation-annotations)
4.  [Model Training](#4-model-training)
    *   [Basic Training Command](#41-basic-training-command)
    *   [K-Fold Cross-Validation](#42-k-fold-cross-validation)
    *   [Key Training Parameters](#43-key-training-parameters)
5.  [Model Evaluation](#5-model-evaluation)
    *   [Single Model / Simple Ensemble Evaluation](#51-single-model--simple-ensemble-evaluation)
    *   [Campaign Evaluation for Multiple Experiments](#52-campaign-evaluation-for-multiple-experiments)
    *   [Key Evaluation Parameters](#53-key-evaluation-parameters)
6.  [Configuration Files](#6-configuration-files)

---

## 1. Getting Started

To get started with ETFinder, follow these general steps:

1.  **Set up your environment**: Install the required dependencies.
2.  **Prepare your data**: Generate segmentation masks for training and convert annotations for evaluation.
3.  **Configure your run**: Adjust `config.py` and `particle_config.py` as needed.
4.  **Train your models**: Use `train.py` to train U-Net models, potentially with K-Fold cross-validation.
5.  **Evaluate your models**: Use `eval_public_czi.py` for individual evaluations or `run_experiment_evaluation.py` for comprehensive campaigns.

## 2. Setup and Installation

First, clone the repository:

```bash
git clone https://github.com/your_repo/etfinder.git
cd etfinder
```

It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`
```

Install the necessary Python packages. A `requirements.txt` file is assumed to be present. If not, you might need to install the following manually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Adjust for your CUDA version
pip install pytorch-lightning numpy pandas zarr monai scipy scikit-image scikit-learn cc3d wandb mrcfile
```

## 3. Data Preparation

Before training or evaluating, you need to prepare your dataset.

### 3.1. Generate Training Segmentation Masks

The `prepare_mask_czi.py` script generates 3D segmentation masks (`.npy` files) from particle annotations (`.json` files) and tomogram data (`.zarr` files). These masks serve as ground truth for semantic segmentation training.

```bash
python prepare_mask_czi.py \
    --data_root /path/to/your/czi_dataset \
    --output_dir /path/to/your/czi_dataset/mask_data \
    --radius_fraction 0.5 \
    --tomo_type denoised \
    --save_plots
```

**Parameters:**

*   `--data_root`: Path to the root directory of your CZI dataset (e.g., containing `train/static/ExperimentRuns`).
*   `--output_dir`: Directory where the `segmentations_X.X` subdirectories will be created. Default is `data_root/mask`.
*   `--tomo_type`: Type of tomogram to process (e.g., `denoised`, `isonetcorrected`). This specifies which `.zarr` file to use.
*   `--radius_fraction`: Fraction of the particle's configured radius to use for generating spherical masks. A value of `0.5` means the mask will have half the particle's physical radius.
*   `--single_pixel_mask`: If set, generates a single-pixel mask at the centroid instead of a sphere. Overrides `--radius_fraction`.
*   `--save_plots`: If set, generates and saves a preview image for each mask, showing segmentation contours and GT annotations. Requires `matplotlib` and `scikit-image`.

### 3.2. Prepare Public Evaluation Annotations

The `prepare_fullpublic_eval_annotations.py` (or `convert_public_picks_to_czi_format.py`) script converts `.ndjson` annotations, typically found in public evaluation datasets, into the structured `.json` format expected by the CZI Challenge evaluation. It also handles the conversion of pixel coordinates to Angstroms based on voxel spacing.

```bash
python prepare_fullpublic_eval_annotations.py \
    --data_dir /path/to/public_evaluation_dataset
```

**Parameters:**

*   `--data_dir`: Path to the root directory containing tomogram subdirectories (e.g., `TS_...`). The `.json` files will be created directly within this structure.

## 4. Model Training

The `train.py` script handles model training using PyTorch Lightning. It supports various encoder backbones, K-Fold cross-validation, and advanced training techniques.

### 4.1. Basic Training Command (Single Fold)

This example trains a `resnet50`-based U-Net on a single validation tomogram (the first one found by default).

```bash
python train.py \
    --encoder_name resnet50 \
    --fold 0 \
    --comment "my_first_resnet50_run" \
    --use_ema \
    --augmentation_level advanced \
    --loss_type ce_tversky
```

### 4.2. K-Fold Cross-Validation

To perform K-Fold cross-validation, specify the folds you want to run. If `--run_specific_folds` is omitted, all available folds will be executed.

```bash
python train.py \
    --encoder_name deepfinder2 \
    --run_specific_folds 0 1 2 \
    --comment "deepfinder2_kfold_experiment" \
    --use_swa \
    --batch_size 2 \
    --epochs 100
```

### 4.3. Key Training Parameters

**Model & Architecture:**

*   `--encoder_name`: Specifies the U-Net encoder. Choices: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `deepfinder2`.
*   `--norm_type`: Type of normalization layer. Choices: `batch`, `instance`, `mix`. (`mix` uses instance norm for early layers, batch norm for deeper).
*   `--patch_size D H W`: Dimensions of the 3D patches used for training (e.g., `128 128 128`). Overrides `config.PATCH_SIZE`.

**Training Process:**

*   `--resume_from_checkpoint`: Path to a `.ckpt` file to resume training.
*   `--epochs`: Number of training epochs. Overrides `config.EPOCHS`.
*   `--batch_size`: Training batch size. Overrides `config.BATCH_SIZE`.
*   `--lr`: Initial learning rate. Overrides `config.LEARNING_RATE`.
*   `--use_scheduler`: Activates the learning rate scheduler defined in `config.py`.
*   `--use_swa`: Enables Stochastic Weight Averaging (SWA).
*   `--use_ema`: Enables Exponential Moving Average (EMA) of model weights.
*   `--use_gradnorm`: Activates GradNorm for dynamic balancing of hybrid losses (e.g., `ce_tversky`). Requires a hybrid loss type.
*   `--gradnorm_lr`: Learning rate for the GradNorm optimizer. Default: `0.001`.

**Loss Function:**

*   `--loss_type`: Type of loss function. Choices: `ce`, `focal_tversky`, `ce_tversky`, `tversky`, `focal`, `focal_tversky_pp`. Overrides `config.LOSS_TYPE`.
*   `--class_loss_weights`: List of weights for the loss, one per class (including background). Example: `1.0 10.0 10.0 ...`.

**Data & Augmentation:**

*   `--data_root`: Path to the CZI data root. Overrides `config.CZI_DATA_ROOT`.
*   `--training_tomo_types`: List of tomogram types to use for training (e.g., `denoised isonetcorrected`). Overrides `config.TRAINING_TOMO_TYPES`.
*   `--augmentation_level`: Level of augmentation: `none`, `base` (flips/rotations), `advanced` (base + affine/noise).
*   `--augmentation_mode`: Augmentation mode: `anisotropic` (Z-axis treated differently) or `isotropic` (all axes treated equally).
*   `--use_mixup`: Activates Mixup augmentation.
*   `--mixup_prob`: Probability of applying Mixup if enabled. Overrides `config.MIXUP_PROB`.
*   `--norm_percentiles LOWER UPPER`: Percentiles for global normalization (e.g., `0 95`). Overrides `config.NORM_LOWER_PERCENTILE`, `config.NORM_UPPER_PERCENTILE`.
*   `--disable_global_norm`: Disables global percentile normalization during training.
*   `--radius_fraction`: Fraction of the particle radius to use for masks. Overrides `config.RADIUS_FRACTION`.
*   `--single_pixel_mask`: Uses a single-pixel mask at the centroid for training. Overrides `--radius_fraction`.

**Logging & Output:**

*   `--comment`: Additional comment for W&B logs and output directory naming.
*   `--fold`: Runs a single specific fold by its index.
*   `--run_specific_folds`: List of fold indices to execute (e.g., `0 2 5`). If not specified, all folds are run.

## 5. Model Evaluation

DeepFinder 2.1 provides two main scripts for evaluation: `eval_public_czi.py` for evaluating a single model or simple ensemble, and `run_experiment_evaluation.py` for orchestrating evaluation campaigns across multiple experiments and strategies.

### 5.1. Single Model / Simple Ensemble Evaluation (`eval_public_czi.py`)

This script evaluates a single model checkpoint or a directory of checkpoints (treated as a simple ensemble) on a specified dataset. It includes options for hyperparameter search for post-processing.

**Example: Evaluate a single model with fixed hyperparameters**

```bash
python eval_public_czi.py \
    --model_path checkpoints/my_experiment_name/fold_0-best-fscore-epoch=X-f4=0.XXX.ckpt \
    --data_root /path/to/public_evaluation_dataset \
    --postprocessing_method peak_local_max_gpu \
    --fixed_plm_gpu_peak_thresh 0.15 \
    --fixed_nms_radius_fraction 0.8 \
    --use_tta \
    --precision bf16
```

**Example: Perform hyperparameter search for `watershed` method (global strategy)**

```bash
python eval_public_czi.py \
    --model_path checkpoints/my_experiment_name/fold_0-best-fscore-epoch=X-f4=0.XXX.ckpt \
    --data_root /path/to/public_evaluation_dataset \
    --postprocessing_method watershed \
    --hp_search_strategy global \
    --ws_cluster_rad_fracs 0.3 0.5 0.8 \
    --ws_vol_frac_thresholds 0.05 0.1 0.2 \
    --limit_tomos_by_id TS_001 # Limit HP search to a single tomogram
```

### 5.2. Campaign Evaluation for Multiple Experiments (`run_experiment_evaluation.py`)

This script automates the evaluation of multiple trained models or experiments. It supports different evaluation strategies and integrates hyperparameter search.

**Example: Evaluate best single models from multiple K-Fold experiments**

```bash
python run_experiment_evaluation.py \
    --experiments_dir checkpoints \
    --experiment_names df1_se_20251008_135757 df2_se_20251009_100000 \
    --evaluation_strategy best_single \
    --postprocessing_method watershed \
    --hp_search_strategy per_class \
    --use_tta \
    --comment "comparison_deepfinder_versions"
```

**Example: Evaluate a K-Fold ensemble**

```bash
python run_experiment_evaluation.py \
    --experiments_dir checkpoints \
    --experiment_names df1_se_20251008_135757 \
    --evaluation_strategy kfold_ensemble \
    --postprocessing_method meanshift \
    --hp_search_strategy global \
    --ms_cluster_rad_fracs 0.8 1.0 1.2 \
    --ms_min_vol_fracs 0.1 0.2 0.3
```

### 5.3. Key Evaluation Parameters

**General Evaluation Settings:**

*   `--experiments_dir`: Directory containing the results of your training runs (e.g., `checkpoints/`). Default: `config.OUTPUT_ROOT/checkpoints/`. (Used by `run_experiment_evaluation.py`)
*   `--experiment_names`: List of base names of the experiments to evaluate. (Used by `run_experiment_evaluation.py`)
*   `--model_path`: Path to a single model checkpoint (`.ckpt`) or a directory containing multiple checkpoints for ensemble inference. (Used by `eval_public_czi.py`)
*   `--data_root`: Path to the dataset for evaluation. For HP search, this should be your training data root. For final evaluation, it should be your public evaluation data root.
*   `--output_dir`: Main output directory for evaluation results.
*   `--tomo_type`: Type of tomogram to use for inference (e.g., `denoised`).
*   `--use_tta`: Activates Test-Time Augmentation (flip + rotation) for improved precision.
*   `--precision`: Inference precision: `bf16`, `fp16`, or `fp32`.
*   `--force_rerun`: Forces re-execution even if intermediate results exist.
*   `--limit_tomos_by_id ID1 ID2 ...`: Limits evaluation to specific tomograms by their IDs. (Used by `eval_public_czi.py`)
*   `--limit_final_eval_tomos N`: Quick test mode: Limits final evaluation to N tomograms. (Used by `run_experiment_evaluation.py`)
*   `--sw_batch_size`: Batch size for MONAI's sliding window inference. Overrides `config.VALIDATION_SW_BATCH_SIZE`.
*   `--blend_mode`: Blending mode for `sliding_window_inference` (`gaussian` or `constant`).
*   `--norm_percentiles LOWER UPPER`: Percentiles for global normalization (e.g., `5 99`).
*   `--use_global_norm`: Forces activation of global percentile normalization.
*   `--training_radius_fraction`: Internal use: Specifies the radius fraction used during training, automatically detected by `run_experiment_evaluation.py`.

**Evaluation Strategies (for `run_experiment_evaluation.py`):**

*   `--evaluation_strategy`: Strategy to use. Choices:
    *   `best_single`: Evaluates the best checkpoint from each K-Fold individually.
    *   `ensemble_specialists`: Creates an ensemble of "specialist" models (one per particle class) for each fold.
    *   `kfold_ensemble`: Creates a single ensemble from the best generalist models of all folds.

**Post-Processing & Hyperparameter Search:**

*   `--postprocessing_method`: Post-processing method to use. Choices: `cc3d`, `meanshift`, `peak_local_max_gpu`, `watershed`.
*   `--postproc_internal_downsample`: Additional internal downsampling factor applied before particle detection, mainly for benchmarking.
*   `--hp_search_strategy`: Hyperparameter search strategy: `per_class` (one set of HPs per particle type) or `global` (a single set of HPs for all).
*   `--per_class_hp_path`: Path to a JSON file containing per-class hyperparameters. Disables grid search.

**Fixed Hyperparameters (to skip grid search):**

*   `--fixed_conf_threshold`: Fixed confidence threshold.
*   `--fixed_nms_radius_fraction`: Fixed NMS radius fraction.
*   `--fixed_vol_frac_threshold`: Fixed volume fraction threshold (for `cc3d` and `watershed`).
*   `--fixed_cluster_radius_fraction`: Fixed clustering radius fraction (for `meanshift` and `watershed`).
*   `--fixed_min_cluster_vol_frac`: Fixed minimum cluster volume fraction (for `meanshift`).
*   `--fixed_plm_gpu_peak_thresh`: Fixed peak threshold (for `peak_local_max_gpu`).
*   `--fixed_ws_cluster_rad_frac`: Fixed clustering radius fraction specific to `watershed`.

**Hyperparameter Search Spaces:**

*   `--conf_thresholds`: List of confidence thresholds to test.
*   `--nms_radius_fractions`: List of NMS radius fractions to test.
*   `--vol_frac_thresholds`: List of volume fractions to test (for `cc3d`).
*   `--cluster_radius_fractions`: List of clustering radius fractions to test (for `meanshift`).
*   `--min_cluster_vol_fracs`: List of minimum cluster volume fractions to test (for `meanshift`).
*   `--plm_gpu_peak_threshs`: List of peak thresholds to test (for `peak_local_max_gpu`).
*   `--ms_cluster_rad_fracs`: Search space for `cluster_radius_fraction` specific to `meanshift`.
*   `--ms_min_vol_fracs`: Search space for `min_cluster_vol_frac` specific to `meanshift`.
*   `--ws_cluster_rad_fracs`: Search space for `cluster_radius_fraction` specific to `watershed`.
*   `--ws_vol_frac_thresholds`: Search space for `vol_frac_threshold` specific to `watershed`.

## 6. Configuration Files

The project uses two main configuration files:

*   `config.py`: Contains global parameters for data paths, model architecture, training, validation, and general post-processing. You should adjust paths like `CZI_DATA_ROOT` and `CZI_PUBLIC_EVAL_DATA_ROOT` to match your local setup.
*   `particle_config.py`: Defines properties for each particle type, such as their physical radius, evaluation weight, and a unique ID. This file is crucial for consistent data generation and evaluation.

---

**Note:** This README provides a comprehensive overview. For detailed usage and advanced configurations, please refer to the comments and docstrings within each Python script.
```