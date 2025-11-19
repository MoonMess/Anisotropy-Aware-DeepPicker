import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import json
import traceback
import torch.optim as optim
from pathlib import Path

try:
    import zarr
except ImportError:
    print("WARNING: 'zarr' is not installed. Validation tomogram reconstruction will not work. Install with 'pip install zarr'.")
    zarr = None

try:
    from skimage.feature import peak_local_max
except ImportError:
    print("WARNING: 'scikit-image' is not installed. Centroid evaluation will not work. Install with 'pip install scikit-image'.")
    peak_local_max = None

try:
    import mrcfile
except ImportError:
    print("WARNING: 'mrcfile' is not installed. Reading .mrc tomograms for visualization will not work. Install with 'pip install mrcfile'.")
    mrcfile = None

try:
    from pytorch_lightning.loggers import WandbLogger
except ImportError:
    WandbLogger = None

# Local imports from the project structure
import config
from particle_config import CZI_PARTICLE_CONFIG
from models.unet import UNet
from models.resnet_encoder_unet import ResNetEncoderUNet
from utils.czi_eval import run_czi_evaluation 
from utils.postprocessing import process_predictions_to_df
from utils.loss import Loss
from utils.inference import run_inference_on_tomogram

# --- Global Flags ---
_TTA_MODE_LOGGED_GLOBALLY = False

class SegWrapper(nn.Module):
    """Wraps a model to return a dictionary {'seg': output}, for consistency."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return {'seg': self.model(x)}

class Mixup(nn.Module):
    """
    Implements Mixup data augmentation for segmentation and regression tasks.
    """
    def __init__(self, mix_beta):
        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y_seg):
        bs = X.shape[0]
        perm = torch.randperm(bs)

        # Generate a lambda coefficient for each sample in the batch
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        # Adapt the shape of the coefficients for broadcasting
        x_coeffs = coeffs.view((-1,) + (1,) * (X.ndim - 1))
        y_coeffs = coeffs.view((-1,) + (1,) * (Y_seg.ndim - 1))

        # Apply Mixup
        X_mixed = x_coeffs * X + (1 - x_coeffs) * X[perm]
        Y_seg_mixed = y_coeffs * Y_seg + (1 - y_coeffs) * Y_seg[perm]

        return X_mixed, Y_seg_mixed

class UnetSystem(pl.LightningModule):
    """
    Encapsulates the model, loss function, and training logic.
    """
    def __init__(self, learning_rate, num_classes, patch_size, use_scheduler, encoder_name, norm_type, use_mixup, mixup_prob, mixup_beta, loss_type, use_ema=False, use_global_norm=True, use_gradnorm=False, gradnorm_lr=0.025, unet_channels=(48, 64, 80, 80), unet_strides=(2, 2, 1), unet_num_res_units=1):
        super().__init__()
        self.save_hyperparameters()

        self.validation_step_outputs = []
        self.czi_particle_config = CZI_PARTICLE_CONFIG
        self.class_names = config.CLASS_NAMES

        if self.hparams.encoder_name == 'deepfinder2':
            print("INFO: Initializing Deepfinder2 (UNet).")
            unet = UNet(
                spatial_dims=3,
                in_channels=config.IN_CHANNELS,
                out_channels=self.hparams.num_classes + 1,
                channels=config.UNET_CHANNELS_DF1,
                strides=config.UNET_STRIDES_DF1,
                num_res_units=config.UNET_NUM_RES_UNITS_DF1,
                norm=self.hparams.norm_type,
                act=config.act,
                dropout=config.dropout_rate,
                bias=False
            )
            self.model = SegWrapper(unet)
        else:
            print(f"INFO: Initializing a ResNetEncoderUNet with encoder '{self.hparams.encoder_name}'.")
            res_unet = ResNetEncoderUNet(
                input_dim=config.IN_CHANNELS,
                num_classes=self.hparams.num_classes + 1, # +1 for background class
                encoder_name=self.hparams.encoder_name,
                norm_type=self.hparams.norm_type,
                filters=list(self.hparams.unet_channels),
                downsample_strides=tuple(self.hparams.unet_strides),
            )
            self.model = SegWrapper(res_unet)

        class_weights = None
        if hasattr(config, 'CLASS_LOSS_WEIGHTS') and config.CLASS_LOSS_WEIGHTS is not None:
            class_weights = config.CLASS_LOSS_WEIGHTS
            print(f"INFO: Using class weights for CrossEntropy loss: {class_weights}")
            # Consistency check
            if len(class_weights) != (self.hparams.num_classes + 1):
                raise ValueError(f"config.CLASS_LOSS_WEIGHTS must have {self.hparams.num_classes + 1} elements (1 for background + {self.hparams.num_classes} classes), but has {len(class_weights)}.")

        self.loss_fn = Loss(
            loss_type=self.hparams.loss_type,
            tversky_alpha=config.TVERSKY_ALPHA, 
            tversky_beta=config.TVERSKY_BETA,
            class_weights=class_weights,
        )

        # Initialize Mixup
        self.mixup = None
        if self.hparams.use_mixup:
            print(f"INFO: Using Mixup with beta={self.hparams.mixup_beta} and probability={self.hparams.mixup_prob}")
            self.mixup = Mixup(mix_beta=self.hparams.mixup_beta)

        # --- GradNorm Configuration ---
        self.automatic_optimization = not self.hparams.use_gradnorm
        if self.hparams.use_gradnorm:
            print("INFO: GradNorm is enabled. Manual optimization will be used.")
            if 'tversky' not in self.hparams.loss_type:
                raise ValueError("GradNorm is currently only implemented for hybrid losses containing 'tversky' (e.g., 'ce_tversky').")
            # Learnable weights for the two loss components
            self.loss_weights = nn.Parameter(torch.ones(2, requires_grad=True))
            self.register_buffer('initial_losses', None)
            # Alpha hyperparameter for GradNorm's pulling force
            self.gradnorm_alpha = 1.5

    def forward(self, x):
        """Forward pass for inference."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        tomo = batch["tomo"]
        seg_target = batch["seg_target"]
        batch_size = tomo.shape[0]

        # Apply Mixup if enabled and in training mode
        if self.training and self.mixup is not None and batch_size > 1 and torch.rand(1).item() < self.hparams.mixup_prob:
            # Mixup requires "soft" (one-hot) segmentation targets
            num_classes = self.hparams.num_classes + 1
            # Convert targets to one-hot for mixup
            seg_target_one_hot = F.one_hot(seg_target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
            tomo, seg_target = self.mixup(tomo, seg_target_one_hot)

        if self.automatic_optimization:
            # --- Standard training logic (without GradNorm) ---
            predictions = self.model(tomo)
            # The loss function now returns a dictionary
            loss_dict = self.loss_fn(predictions, seg_target)

            # Manual combination of losses (previous behavior)
            if 'ce' in loss_dict and 'tversky' in loss_dict:
                total_loss = loss_dict['ce'] * 2 + loss_dict['tversky']
                self.log("train/loss_ce", loss_dict['ce'].detach(), on_step=False, on_epoch=True, logger=True, batch_size=batch_size)
                self.log("train/loss_tversky", loss_dict['tversky'].detach(), on_step=False, on_epoch=True, logger=True, batch_size=batch_size)
            else:
                total_loss = sum(loss_dict.values())

            lr = self.optimizers().param_groups[0]['lr']
            self.log('train/learning_rate', lr, on_step=True, on_epoch=False, logger=True)
            self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
            return total_loss
        else:
            # --- Manual training logic with GradNorm (reorganized to avoid 'inplace' error) ---
            opt_model, opt_weights = self.optimizers()

            # Zero out gradients for both optimizers
            opt_model.zero_grad()
            opt_weights.zero_grad()

            # Forward pass to get losses
            predictions = self.model(tomo)
            unweighted_losses = self.loss_fn(predictions, seg_target)

            loss_1 = unweighted_losses.get('ce', unweighted_losses.get('focal'))
            loss_2 = unweighted_losses['tversky']

            weights = F.softmax(self.loss_weights, dim=-1) * len(unweighted_losses)

            # --- GradNorm loss calculation (must be done before any weight update) ---
            if self.initial_losses is None:
                self.initial_losses = torch.stack([loss_1.detach(), loss_2.detach()])

            last_layer = None
            model_instance = self.model.model
            if isinstance(model_instance, ResNetEncoderUNet):
                last_layer = model_instance.seg_head
            elif isinstance(model_instance, UNet):
                up_path = model_instance.model[-1]
                if model_instance.num_res_units > 0:
                    # up_path is a Sequential(Convolution, ResidualUnit)
                    res_unit = up_path[-1]
                    # res_unit.block is a Sequential of Convolutions
                    last_conv_module = res_unit.block[-1]
                    last_layer = last_conv_module.conv
                else:
                    # up_path is a Convolution module
                    last_layer = up_path.conv
            if last_layer is None:
                raise TypeError(f"GradNorm could not find the last layer for model type {type(model_instance)}. The architecture may have changed.")

            grad_norms = []
            for i, loss_comp in enumerate([loss_1, loss_2]):
                grad = torch.autograd.grad(weights[i] * loss_comp, last_layer.parameters(), retain_graph=True, allow_unused=True, create_graph=True)
                if grad[0] is not None:
                    flat_grad = torch.cat([g.view(-1) for g in grad if g is not None])
                    grad_norms.append(torch.norm(flat_grad))
                else:
                    grad_norms.append(torch.tensor(0.0, device=self.device))

            grad_norms = torch.stack(grad_norms)
            avg_grad_norm = grad_norms.mean()

            loss_ratio = torch.stack([loss_1.detach(), loss_2.detach()]) / self.initial_losses
            relative_inverse_rates = loss_ratio / loss_ratio.mean()

            gradnorm_loss = torch.sum(torch.abs(grad_norms - avg_grad_norm.detach() * (relative_inverse_rates ** self.gradnorm_alpha)))

            # --- Backpropagation and weight update ---
            weighted_loss = weights[0] * loss_1 + weights[1] * loss_2
            self.manual_backward(weighted_loss, retain_graph=True)

            loss_weights_grads = torch.autograd.grad(gradnorm_loss, self.loss_weights)
            self.loss_weights.grad = loss_weights_grads[0]

            self.clip_gradients(opt_model, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            opt_model.step()
            opt_weights.step()

            # Logging
            self.log_dict({
                "train/total_loss": weighted_loss.detach(), "train/gradnorm_loss": gradnorm_loss.detach(),
                "train/weight_1": weights[0].detach(), "train/weight_2": weights[1].detach(),
                "train/grad_norm_1": grad_norms[0].detach(), "train/grad_norm_2": grad_norms[1].detach()
            }, batch_size=batch_size)

    @staticmethod
    def _mode_projection(label_map_3d: np.ndarray, num_total_classes: int) -> np.ndarray:
        """
        Projects a 3D segmentation mask to 2D by finding the most frequent
        particle class along the depth axis. The background (class 0) is
        chosen only if no other class is present.
        """
        if label_map_3d.size == 0:
            return np.array([[]], dtype=np.uint8)

        # Create a one-hot representation of the 3D segmentation slab.
        # The shape will be (D, H, W, num_total_classes).
        one_hot = np.eye(num_total_classes, dtype=np.uint8)[label_map_3d]

        # Count occurrences of each class along the depth axis.
        # The resulting shape is (H, W, num_total_classes).
        counts = one_hot.sum(axis=0)

        # Isolate particle counts (classes 1 and up)
        particle_counts = counts[:, :, 1:]

        # Find the most frequent particle class.
        # The index returned by argmax is from 0 to (num_classes-1), so we add 1 for the class ID.
        most_frequent_particle = np.argmax(particle_counts, axis=2) + 1

        # Determine where there is only background (sum of particle counts is zero).
        only_background_mask = (particle_counts.sum(axis=2) == 0)

        # The projected map is the most frequent particle, set to 0 where there is only background.
        projected_map_2d = np.where(only_background_mask, 0, most_frequent_particle)

        return projected_map_2d.astype(np.uint8)

    def _get_tta_predictions(self, patches):
        """
        Applies Test-Time Augmentation (TTA) by automatically detecting the mode
        (anisotropic vs isotropic) from the model's hyperparameters.
        """
        global _TTA_MODE_LOGGED_GLOBALLY
        # Automatically detect the augmentation mode used during training.
        # Default to 'anisotropic' for compatibility with older models.
        tta_mode = getattr(self.hparams, 'augmentation_mode', 'anisotropic')
        if not _TTA_MODE_LOGGED_GLOBALLY:
            print(f"INFO (TTA): Using '{tta_mode}' mode detected from model hyperparameters.")
            _TTA_MODE_LOGGED_GLOBALLY = True

        all_probs = []

        # 1. Original prediction
        all_probs.append(torch.softmax(self.model(patches)['seg'], dim=1))

        # 2. Flips (common to both modes)
        # Axes: 2=D, 3=H, 4=W for a tensor (N, C, D, H, W)
        for axis in [2, 3, 4]:
            flipped_patches = torch.flip(patches, [axis])
            flipped_preds = self.model(flipped_patches)['seg']
            # Undo the flip on the prediction
            back_preds = torch.flip(flipped_preds, [axis])
            all_probs.append(torch.softmax(back_preds, dim=1))

        # 3. Rotations around the Z-axis (XY plane) (common to both modes)
        # Rotation axes: [3, 4] -> (H, W)
        for k in range(1, 4): # k=1,2,3 pour 90, 180, 270 degrés
            rotated_patches = torch.rot90(patches, k, [3, 4])
            rotated_preds = self.model(rotated_patches)['seg']
            # Undo the rotation
            back_preds = torch.rot90(rotated_preds, -k, [3, 4])
            all_probs.append(torch.softmax(back_preds, dim=1))

        # 4. Additional rotations for isotropic mode
        if tta_mode == 'isotropic':
            # Rotations around the Y-axis (XZ plane)
            # Rotation axes: [2, 4] -> (D, W)
            for k in range(1, 4):
                rotated_patches = torch.rot90(patches, k, [2, 4])
                rotated_preds = self.model(rotated_patches)['seg']
                back_preds = torch.rot9ot(rotated_preds, -k, [2, 4])
                all_probs.append(torch.softmax(back_preds, dim=1))

            # Rotations around the X-axis (YZ plane)
            # Rotation axes: [2, 3] -> (D, H)
            for k in range(1, 4):
                rotated_patches = torch.rot90(patches, k, [2, 3])
                rotated_preds = self.model(rotated_patches)['seg']
                back_preds = torch.rot90(rotated_preds, -k, [2, 3])
                all_probs.append(torch.softmax(back_preds, dim=1))

        # Average of all collected predictions
        ensembled_probs = torch.stack(all_probs).mean(dim=0)
        return ensembled_probs


    def validation_step(self, batch, batch_idx):
        tomo = batch["tomo"]
        seg_target = batch["seg_target"]
        batch_size = tomo.shape[0]

        # Calculate and log the loss on the validation batch.
        # Full reconstruction will be done in on_validation_epoch_end.
        predictions = self.model(tomo)        
        loss_dict = self.loss_fn(predictions, seg_target)

        # Manual combination of losses for logging (previous behavior)
        if 'ce' in loss_dict and 'tversky' in loss_dict:
            total_loss = loss_dict['ce'] * 2 + loss_dict['tversky']
        else:
            total_loss = sum(loss_dict.values())

        self.log("val/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        # Log individual components
        for name, value in loss_dict.items():
            self.log(f"val/loss_{name}", value, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)

    def on_validation_epoch_end(self):
        main_metric_name = f"val/czi_f{config.CZI_EVAL_BETA}_score"

        if self.trainer.sanity_checking:
            self.log(main_metric_name, 0.0, prog_bar=True)
            # Log zeros for all per-class metrics to avoid errors
            for p_type in self.class_names:
                self.log(f"val_f{config.CZI_EVAL_BETA}_{p_type}", 0.0)
            return

        print("\n--- End-of-epoch CZI evaluation on the reconstructed tomogram ---")

        val_tomo_info = self.trainer.datamodule.val_dataset.file_list[0]
        val_tomo_id = val_tomo_info['run_id']

        # 1. Run inference on the full validation tomogram
        val_tomo_path = val_tomo_info['tomo_paths'].get('denoised')
        if not val_tomo_path:
            raise FileNotFoundError(f"'denoised' tomogram not found for validation of {val_tomo_id}")

        precision = 'bf16' if self.device.type == 'cuda' and torch.cuda.is_bf16_supported() else 'fp32'

        full_prob_map = run_inference_on_tomogram(
            models=self, # The function can take a LightningModule directly
            tomo_path=val_tomo_path,
            device=self.device,
            patch_size=self.hparams.patch_size,
            use_tta=config.USE_TTA_VALIDATION,
            precision=precision,
            overlap_fraction=config.VALIDATION_OVERLAP_FRACTION,
            sw_batch_size=self.trainer.datamodule.hparams.validation_sw_batch_size,
            progress=False, # No nested progress bar during training
            use_global_norm=self.hparams.use_global_norm
        )

        # 2. Extract predictions, process them (NMS), and get peaks for visualization
        submission_df = self._extract_and_process_predictions(full_prob_map, val_tomo_id)

        # 3. Load ground truth and run evaluation
        solution_df = self._load_validation_ground_truth(val_tomo_id)
        final_score = self._run_czi_evaluation_and_log(submission_df, solution_df)
        self.log(main_metric_name, final_score, prog_bar=True)

        # 4. Log visualizations to W&B
        if WandbLogger and isinstance(self.trainer.logger, WandbLogger):
            print("INFO: W&B logger detected. Attempting to log image.")
            self._log_visualizations(full_prob_map, val_tomo_info, solution_df, submission_df)
        else:
            print("INFO: W&B logger not detected or not imported. Visualization will be skipped.")


    def _extract_and_process_predictions(self, full_prob_map, val_tomo_id):
        """Extracts particles, formats predictions, and applies NMS for validation."""
        print("  Extracting and processing predictions via the unified pipeline...")

        # Parameters for validation are fixed and defined in config.py
        # or directly here for monitoring.
        submission_df = process_predictions_to_df(
            prediction_map=full_prob_map,
            tomo_id=val_tomo_id,
            particle_config=self.czi_particle_config,
            voxel_spacing=config.VOXEL_SPACING,
            conf_threshold=config.VALIDATION_CONF_THRESHOLD,
            nms_radius_fraction=config.VALIDATION_NMS_RADIUS_FRACTION,
            vol_fraction_threshold=config.CC3D_VOL_FRAC_THRESHOLD,
            downsampling_factor=config.POSTPROC_DOWNSAMPLING_FACTOR,
            postprocessing_method=config.VALIDATION_POSTPROC_METHOD
        )

        if submission_df.empty:
            print("  No particles detected.")
        else:
            print(f"  {len(submission_df)} detections found after post-processing.")

        return submission_df

    def _load_validation_ground_truth(self, val_tomo_id):
        """Loads the ground truth DataFrame for the validation tomogram."""
        print("  Loading ground truth for evaluation...")
        gt_points = []
        gt_root_path = Path(config.CZI_DATA_ROOT) / "train/overlay/ExperimentRuns"
        run_dir = gt_root_path / val_tomo_id
        if not run_dir.is_dir():
            print(f"WARNING: Ground truth directory not found for {val_tomo_id} at {run_dir}")
            return pd.DataFrame()

        picks_dir = run_dir / "Picks"
        if not picks_dir.is_dir():
            return pd.DataFrame()

        for p_name in self.czi_particle_config:
            json_path = picks_dir / f"{p_name}.json"
            if not json_path.exists(): continue
            with open(json_path, 'r') as f:
                data = json.load(f)
                for point in data.get('points', []):
                    loc = point.get('location')
                    if loc:
                        gt_points.append({'experiment': run_dir.name, 'particle_type': p_name, 'x': loc['x'], 'y': loc['y'], 'z': loc['z']})

        return pd.DataFrame(gt_points)

    def _run_czi_evaluation_and_log(self, submission_df, solution_df):
        """Runs the CZI evaluation and logs the metrics."""
        print("  Running CZI evaluation...")
        final_score = 0.0
        if not submission_df.empty and not solution_df.empty:
            score, results_df, _ = run_czi_evaluation(
                submission_df, solution_df, self.czi_particle_config, beta=config.CZI_EVAL_BETA
            )
            final_score = score

            # Log the F-beta score for each class
            for _, row in results_df.iterrows():
                p_type = row['particle_type']
                # Use a simple metric name for checkpointing
                self.log(f"val_f{config.CZI_EVAL_BETA}_{p_type}", row['f_beta'])
                self.log(f"val_czi/precision_{p_type}", row['precision'])
                self.log(f"val_czi/recall_{p_type}", row['recall'])
        else:
            print("INFO: No predictions or ground truth for CZI evaluation, logging zeros.")
            # Add checks for debugging
            print(f"Submission DataFrame is empty: {submission_df.empty}")
            print(f"Solution DataFrame is empty: {solution_df.empty}")

            # Log zeros for all metrics if evaluation fails
            for p_type in self.class_names:
                self.log(f"val_f{config.CZI_EVAL_BETA}_{p_type}", 0.0)
                self.log(f"val_czi/precision_{p_type}", 0.0)
                self.log(f"val_czi/recall_{p_type}", 0.0)

        return final_score

    def _log_visualizations(self, full_prob_map, val_tomo_info, solution_df, submission_df):
        """Generates and logs full tomogram visualizations to W&B."""
        print("  Generating full tomogram visualization for W&B...")
        try:
            val_tomo_path = val_tomo_info['tomo_paths'].get('denoised')
            if not val_tomo_path:
                return

            if val_tomo_path.endswith('.zarr'):
                with zarr.open(val_tomo_path, mode='r') as z:
                    tomo_data = z['0'] if isinstance(z, zarr.hierarchy.Group) else z
                    tomo_shape = tomo_data.shape
                    z_slice = tomo_shape[0] // 2
                    tomo_slice_data = tomo_data[z_slice, :, :]
            elif val_tomo_path.endswith('.mrc'):
                if mrcfile is None:
                    print("WARNING: mrcfile not installed, cannot generate visualization for .mrc tomogram.")
                    return
                with mrcfile.open(val_tomo_path) as mrc:
                    tomo_data = mrc.data
                    tomo_shape = tomo_data.shape
                    z_slice = tomo_shape[0] // 2
                    tomo_slice_data = tomo_data[z_slice, :, :]
            else:
                print(f"WARNING: Unsupported file format for visualization: {val_tomo_path}")
                return

            # Half-thickness of the slab for segmentation projection (11 pixels total).
            seg_tolerance = 5
            z_slice = tomo_shape[0] // 2
            z_start = max(0, z_slice - seg_tolerance)
            z_end = min(tomo_shape[0], z_slice + seg_tolerance + 1)

            num_total_classes = self.hparams.num_classes + 1

            # --- Ground Truth (GT) ---
            # Load the GT segmentation slab and project it to 2D using mode projection.
            gt_seg_map_slab = np.load(val_tomo_info['seg_path'], mmap_mode='r')[z_start:z_end, :, :]
            gt_seg_map_slice = self._mode_projection(gt_seg_map_slab, num_total_classes)

            # --- Prediction ---
            # Apply argmax in 3D on the probability slab, then project to 2D with mode.
            pred_prob_map_slab = full_prob_map[:, z_start:z_end, :, :]
            argmax_slab_np = torch.argmax(pred_prob_map_slab, dim=0).cpu().numpy()
            pred_seg_map_slice = self._mode_projection(argmax_slab_np, num_total_classes)

            # --- New figure layout (asymmetric 2x2) ---
            fig = plt.figure(figsize=(24, 12))
            gs = plt.GridSpec(2, 3, figure=fig)
            ax_tomo = fig.add_subplot(gs[0, 0])
            ax_gt = fig.add_subplot(gs[1, 0])
            ax_pred = fig.add_subplot(gs[:, 1:]) # The prediction panel takes up the full height

            fig.suptitle(f"Epoch {self.current_epoch+1}: Validation on {val_tomo_info['run_id']}", fontsize=16)

            # --- Panel 1: Original Tomogram ---
            ax_tomo.imshow(tomo_slice_data, cmap='gray', origin='lower')
            ax_tomo.set_title(f"Original Tomogram (Z-slice={z_slice})")
            ax_tomo.set_xticks([]); ax_tomo.set_yticks([])

            # --- Definition of a more distinct color palette ---
            # Colors from the 'Set1' palette + a black background.
            distinct_colors = [
                '#000000',  # 0: background (noir)
                '#e41a1c',  # 1: apo-ferritin (rouge)
                '#377eb8',  # 2: beta-amylase (bleu)
                '#4daf4a',  # 3: beta-galactosidase (vert)
                '#984ea3',  # 4: ribosome (violet)
                '#ff7f00',  # 5: thyroglobulin (orange)
                '#ffff33',  # 6: virus-like-particle (jaune)
            ]
            if len(distinct_colors) < num_total_classes:
                # If not enough colors are defined, fall back to a standard palette
                cmap = plt.get_cmap('tab10', num_total_classes)
            else:
                cmap = ListedColormap(distinct_colors[:num_total_classes])

            # --- Panel 2: Ground Truth (GT) ---
            ax_gt.imshow(gt_seg_map_slice, cmap=cmap, vmin=0, vmax=self.hparams.num_classes, origin='lower')
            ax_gt.set_title("GT Segmentation")
            ax_gt.set_xticks([]); ax_gt.set_yticks([])

            # --- Panel 3: Predictions (large) ---
            ax_pred.imshow(pred_seg_map_slice, cmap=cmap, vmin=0, vmax=self.hparams.num_classes, origin='lower')
            ax_pred.set_title("Prediction Segmentation + GT (circles) + Detections (crosses)")
            ax_pred.set_xticks([]); ax_pred.set_yticks([])

            # --- Filtering centroids with dynamic tolerance based on radius ---
            def get_dynamic_centroid_tolerance(particle_type: str) -> float:
                """Calculates a Z tolerance (in pixels) based on the particle radius."""
                radius_angstrom = self.czi_particle_config[particle_type]['radius']
                # The tolerance is the particle radius in pixels.
                # A minimum value is used to ensure that even small particles are visible
                # if they are close to the segmentation slab.
                radius_px = radius_angstrom / config.VOXEL_SPACING
                return max(seg_tolerance, radius_px)

            # Apply the dynamic filter to the ground truth
            if not solution_df.empty:
                gt_tolerances = solution_df['particle_type'].apply(get_dynamic_centroid_tolerance)
                gt_on_slice = solution_df[np.abs(solution_df['z'] / config.VOXEL_SPACING - z_slice) <= gt_tolerances]
            else:
                gt_on_slice = pd.DataFrame()

            # Apply the dynamic filter to the predictions
            if not submission_df.empty:
                pred_tolerances = submission_df['particle_type'].apply(get_dynamic_centroid_tolerance)
                preds_on_slice = submission_df[np.abs(submission_df['z'] / config.VOXEL_SPACING - z_slice) <= pred_tolerances]
            else:
                preds_on_slice = pd.DataFrame()
            class_id_map = {name: props['id'] for name, props in self.czi_particle_config.items()}

            # --- DEBUGGING ---
            print(f"    - Visualization: {len(gt_on_slice)} GT points and {len(preds_on_slice)} detections on this slice.")

            if not gt_on_slice.empty:
                # Group by particle type to assign colors to GT circles
                for p_type, group in gt_on_slice.groupby('particle_type'):
                    gt_x_px = group['x'].values / config.VOXEL_SPACING
                    gt_y_px = group['y'].values / config.VOXEL_SPACING
                    class_id = class_id_map.get(p_type)
                    color = cmap(class_id) if class_id is not None and class_id < cmap.N else 'fuchsia'
                    # Draw a black shadow/outline for better visibility
                    ax_pred.scatter(gt_x_px, gt_y_px, s=120, facecolors='none', edgecolors='black', linewidths=3.5, marker='o')
                    # Draw the colored circle on top
                    ax_pred.scatter(gt_x_px, gt_y_px, s=120, facecolors='none', edgecolors=color, linewidths=2, marker='o', label=f'GT ({p_type})')

            if not preds_on_slice.empty:
                # Group by particle type to assign colors
                for p_type, group in preds_on_slice.groupby('particle_type'):
                    pred_x_px = group['x'].values / config.VOXEL_SPACING
                    pred_y_px = group['y'].values / config.VOXEL_SPACING
                    class_id = class_id_map.get(p_type)
                    color = cmap(class_id) if class_id is not None and class_id < cmap.N else 'red'
                    # Draw a black shadow/outline for better visibility
                    ax_pred.scatter(pred_x_px, pred_y_px, s=120, c='black', marker='x', linewidths=3.5)
                    # Draw the colored cross on top
                    ax_pred.scatter(pred_x_px, pred_y_px, s=120, c=[color], marker='x', linewidths=2, label=f'Detection ({p_type})')

            if not gt_on_slice.empty or not preds_on_slice.empty:
                # Handle the legend to avoid duplicates
                handles, labels = ax_pred.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax_pred.legend(by_label.values(), by_label.keys())

            # Segmentation class legend
            legend_labels = ['background'] + self.class_names
            patches_list = [patches.Patch(color=cmap(i), label=label) 
                            for i, label in enumerate(legend_labels)]
            fig.legend(handles=patches_list, loc='center right', title="Segmentation Classes")

            plt.tight_layout(rect=[0, 0, 0.9, 0.95])
            self.trainer.logger.log_image(key="validation/full_reconstruction", images=[fig], caption=[f"Epoch {self.current_epoch + 1}"])
            plt.close('all')
            print("  ...visualization saved to W&B.")
        except Exception as e:
            print("\n" + "="*60)
            print(f"WARNING: Failed to generate full tomogram visualization.")
            print(f"Error: {e}")
            print("Full traceback:")
            traceback.print_exc()
            print("="*60 + "\n")

    def configure_optimizers(self):
        if self.hparams.use_gradnorm:
            # Optimizer for model weights
            model_optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
            # Separate optimizer for GradNorm loss weights
            weights_optimizer = optim.AdamW([self.loss_weights], lr=self.hparams.gradnorm_lr)
            return model_optimizer, weights_optimizer

        # Standard optimizer configuration
        optimizer = optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate,
        )

        if not self.hparams.use_scheduler:
            print("INFO: Learning rate scheduler is disabled.")
            return optimizer

        scheduler_type = getattr(config, 'SCHEDULER_TYPE', 'cosine')
        if scheduler_type == 'cosine':
            print("INFO: Using CosineAnnealingLR scheduler.")
            scheduler_config = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.trainer.estimated_stepping_batches,
                    eta_min=0,
                ),
                "interval": "step",
            }
        elif scheduler_type == 'plateau':
            print("INFO: Using ReduceLROnPlateau scheduler.")
            scheduler_config = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=getattr(config, 'PLATEAU_FACTOR', 0.2),
                    patience=getattr(config, 'PLATEAU_PATIENCE', 10),
                ),
                "monitor": 'val/total_loss',
                "interval": "epoch",
            }
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}