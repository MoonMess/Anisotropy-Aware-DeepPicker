#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified script for inference with a single model or an ensemble of U-Net models,
particle centroid extraction, and optimization of post-processing
hyperparameters via a grid search.
"""

#python -u run_experiment_evaluation.py --experiments_dir checkpoints --use_tta --experiment_names df1_se_20251008_135757

import argparse
import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import zarr
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from monai.inferers import sliding_window_inference
from monai.utils import BlendMode
import json
from typing import List, Dict, Union
import re

import config
# We load MultiTaskUnetSystem from its own module to be able to load the checkpoint
from models.system import UnetSystem
from utils.czi_eval import run_czi_evaluation 
from utils.postprocessing import process_predictions_to_df
from utils.inference import run_inference_on_tomogram
from particle_config import CZI_PARTICLE_CONFIG

def find_tomograms(data_root: str, tomo_type: str) -> List[Dict]:
	"""
	Finds tomogram files (.zarr or .mrc).
	Handles both the training data structure and the public evaluation set structure.
	"""
	tomo_list = []
	base_dir = Path(data_root)
	if not base_dir.is_dir():
		raise FileNotFoundError(f"Data directory '{data_root}' not found.")

	# --- Detect directory structure ---
	# 1. Training data structure (contains 'train/static/ExperimentRuns')
	training_runs_dir = base_dir / "train/static/ExperimentRuns"
	if training_runs_dir.is_dir():
		print("INFO: Detecting training data structure for tomograms.")
		run_ids = sorted([d.name for d in training_runs_dir.iterdir() if d.is_dir() and d.name.startswith("TS_")])
		for run_id in run_ids:
			tomo_base_path = training_runs_dir / run_id / "VoxelSpacing10.000"
			if tomo_base_path.is_dir():
				tomo_path_zarr = tomo_base_path / f"{tomo_type}.zarr"
				tomo_path_mrc = tomo_base_path / f"{tomo_type}.mrc"
				
				tomo_path_to_use = None
				if tomo_path_zarr.exists():
					tomo_path_to_use = tomo_path_zarr
				elif tomo_path_mrc.exists():
					tomo_path_to_use = tomo_path_mrc

				if tomo_path_to_use:
					tomo_list.append({"id": run_id, "path": str(tomo_path_to_use)})
		return tomo_list

	# 2. Public evaluation data structure (contains 'TS_.../Reconstructions')
	print("INFO: Detecting public evaluation data structure for tomograms.")
	run_ids = sorted([d.name for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("TS_")])
	for run_id in run_ids:
		# The path is fixed for the public evaluation set
		tomo_base_path = base_dir / run_id / "Reconstructions" / "VoxelSpacing10.012" / "Tomograms" / "102"
		
		tomo_path_zarr = tomo_base_path / f"{run_id}.zarr"
		tomo_path_mrc = tomo_base_path / f"{run_id}.mrc"
		
		tomo_path_to_use = None
		if tomo_path_zarr.exists():
			tomo_path_to_use = tomo_path_zarr
		elif tomo_path_mrc.exists():
			tomo_path_to_use = tomo_path_mrc

		if tomo_path_to_use:
			tomo_list.append({"id": run_id, "path": str(tomo_path_to_use)})
	return tomo_list

def _load_models(model_path_arg: Path, device: torch.device, dtype: torch.dtype) -> tuple[Union[List[UnetSystem], Dict[str, UnetSystem]], tuple]:
	"""Loads one or more models from a path and returns the models and patch size."""
	if not model_path_arg.exists():
		raise FileNotFoundError(f"Model path or directory not found: {model_path_arg}")

	models_to_infer: Union[List[UnetSystem], Dict[str, UnetSystem]]
	if model_path_arg.is_dir():
		model_files = list(model_path_arg.rglob('*.ckpt'))
		if not model_files:
			raise FileNotFoundError(f"No checkpoint files (.ckpt) found in the directory {model_path_arg}")

		class_names_str = "|".join(config.CLASS_NAMES)
		specialist_pattern = re.compile(rf"-best-({class_names_str})-")
		
		is_specialist_run = any(specialist_pattern.search(p.name) for p in model_files)

		if is_specialist_run:
			print(f"INFO: Detected a set of specialist models. Loading {len(model_files)} models from: {model_path_arg}")
			specialist_models: Dict[str, UnetSystem] = {}
			for path in model_files:
				match = specialist_pattern.search(path.name)
				if not match:
					print(f"WARNING: File {path.name} ignored because it does not match the expected specialist format in this directory.")
					continue
				
				class_name = match.group(1)
				model = UnetSystem.load_from_checkpoint(path, map_location=device, weights_only=False, strict=False)
				checkpoint = torch.load(path, map_location=device, weights_only=False)
				if 'ema_state_dict' in checkpoint:
					print(f"  -> INFO: EMA weights found in {path.name}. Loading automatically for specialist {class_name}.")
					model.load_state_dict(checkpoint['ema_state_dict'])
				elif getattr(model.hparams, 'use_ema', False):
					print(f"  -> WARNING: The model was trained with EMA but 'ema_state_dict' is not found in {path.name}. Using standard weights.")

				specialist_models[class_name] = model.eval().to(dtype=dtype)
			models_to_infer = specialist_models
		else:
			print(f"INFO: Detected a set of generalist models. Loading {len(model_files)} models for ensemble inference from: {model_path_arg}")
			generalist_models: List[UnetSystem] = []
			for path in model_files:
				model = UnetSystem.load_from_checkpoint(path, map_location=device, weights_only=False, strict=False)
				checkpoint = torch.load(path, map_location=device, weights_only=False)
				if 'ema_state_dict' in checkpoint:
					print(f"  -> INFO: EMA weights found in {path.name}. Loading automatically.")
					model.load_state_dict(checkpoint['ema_state_dict'])
				elif getattr(model.hparams, 'use_ema', False):
					print(f"  -> WARNING: The model was trained with EMA but 'ema_state_dict' is not found in {path.name}. Using standard weights.")
				generalist_models.append(model.eval().to(dtype=dtype))
			models_to_infer = generalist_models

	elif model_path_arg.is_dir() and "kfold_ensemble" in model_path_arg.name: # New logic for K-Fold ensemble
		model_files = list(model_path_arg.rglob('*.ckpt'))
		if not model_files:
			raise FileNotFoundError(f"No checkpoint files (.ckpt) found in the directory {model_path_arg} for the K-Fold ensemble.")
		print(f"INFO: K-Fold ensemble detected. Loading {len(model_files)} models from: {model_path_arg}")
		kfold_ensemble_models: List[UnetSystem] = []
		for path in model_files:
			kfold_ensemble_models.append(_load_single_model_from_path(path, device, dtype))
		models_to_infer = kfold_ensemble_models
	elif model_path_arg.is_file():
		print(f"Loading a single model from: {model_path_arg}")
		model = UnetSystem.load_from_checkpoint(model_path_arg, map_location=device, weights_only=False, strict=False)
		checkpoint = torch.load(model_path_arg, map_location=device, weights_only=False)
		if 'ema_state_dict' in checkpoint:
			print("  -> INFO: EMA weights found. Loading automatically.")
			model.load_state_dict(checkpoint['ema_state_dict'])
		elif getattr(model.hparams, 'use_ema', False):
			print("  -> WARNING: The model was trained with EMA but 'ema_state_dict' is not found. Using standard weights.")
		models_to_infer = [model.eval().to(dtype=dtype)]

	print("Model(s) loaded and ready for inference.")

	patch_size = None
	if isinstance(models_to_infer, list) and models_to_infer:
		patch_size = models_to_infer[0].hparams.patch_size
	elif isinstance(models_to_infer, dict) and models_to_infer:
		patch_size = next(iter(models_to_infer.values())).hparams.patch_size
	
	if patch_size is None:
		raise ValueError("Could not determine patch size from loaded models.")

	return models_to_infer, patch_size

def _load_single_model_from_path(path: Path, device: torch.device, dtype: torch.dtype) -> UnetSystem:
	model = UnetSystem.load_from_checkpoint(path, map_location=device, weights_only=False, strict=False)
	print("Model(s) loaded and ready for inference.")

	patch_size = None
	if isinstance(models_to_infer, list) and models_to_infer:
		patch_size = models_to_infer[0].hparams.patch_size
	elif isinstance(models_to_infer, dict) and models_to_infer:
		patch_size = next(iter(models_to_infer.values())).hparams.patch_size
	
	if patch_size is None:
		raise ValueError("Could not determine patch size from loaded models.")

	return model.eval().to(dtype=dtype)

def _load_ground_truth(data_root: str) -> pd.DataFrame:
	"""Loads and returns the ground truth DataFrame."""
	gt_points = []
	gt_root_path = Path(data_root)

	gt_runs_base_dir = gt_root_path
	training_gt_dir = gt_root_path / "train/overlay/ExperimentRuns"
	if training_gt_dir.is_dir():
		print("INFO: Detecting training data GT structure.")
		gt_runs_base_dir = training_gt_dir
	else:
		print("INFO: Detecting public evaluation data GT structure.")

	if not gt_runs_base_dir.is_dir():
		print(f"WARNING: Ground truth directory '{gt_runs_base_dir}' not found. Evaluation will be skipped.")
		return pd.DataFrame()

	for run_dir in sorted([d for d in gt_runs_base_dir.iterdir() if d.is_dir() and d.name.startswith("TS_")]):
		picks_dir = run_dir / "Picks"
		if not picks_dir.is_dir(): continue
		for p_name in CZI_PARTICLE_CONFIG:
			json_path = picks_dir / f"{p_name}.json"
			if not json_path.exists(): continue
			with open(json_path, 'r') as f:
				data = json.load(f)
				for point in data.get('points', []):
					loc = point.get('location')
					if loc:
						gt_points.append({
							'experiment': run_dir.name, 
							'particle_type': p_name, 
							'x': loc['x'], 
							'y': loc['y'], 
							'z': loc['z']
						})
	solution_df = pd.DataFrame(gt_points)

	if not solution_df.empty:
		max_coord_val = solution_df[['x', 'y', 'z']].max().max()
		if max_coord_val < 2000:
			print("\n" + "="*80)
			print("WARNING: The maximum value of the ground truth coordinates is very low.")
			print(f"({max_coord_val:.2f}). It is very likely that the coordinates are in pixels instead of Angstroms.")
			print("="*80 + "\n")
	
	return solution_df

def _run_grid_search(
	models_to_infer: Union[List[UnetSystem], Dict[str, UnetSystem]],
	tomos_to_process: List[Dict],
	solution_df: pd.DataFrame,
	patch_size: tuple,
	device: torch.device,
	args: argparse.Namespace,
	use_global_norm: bool,
	training_radius_fraction: float
) -> dict:
	"""Runs the hyperparameter search and returns the best parameters."""
	start_time = pd.Timestamp.now()
	hp_search_strategy = getattr(args, 'hp_search_strategy', 'per_class') # Get the strategy

	print("\n" + "="*80)
	print(f"Mode: HYPERPARAMETER SEARCH (Strategy: {hp_search_strategy})")
	print("="*80)

	if len(tomos_to_process) > 1:
		print(f"WARNING: HP search is designed for a single tomogram, but {len(tomos_to_process)} were provided. Only the first one will be used.")
		tomos_to_process = tomos_to_process[:1]

	# --- Inference (run only once) ---
	tomo_info = tomos_to_process[0]
	print(f"\n--- Exécution de l'inférence pour : {tomo_info['id']} ---")
	# Specify which model is used for inference
	if isinstance(models_to_infer, list):
		model_name = Path(args.model_path).name if len(models_to_infer) == 1 else f"Ensemble de {len(models_to_infer)} modèles"
		print(f"--- Model used: {model_name} ---")
	elif isinstance(models_to_infer, dict):
		 print(f"--- Models used: Ensemble of {len(models_to_infer)} specialists ---")

	blend_mode_str = getattr(args, 'blend_mode', 'gaussian')
	blend_mode = BlendMode.GAUSSIAN if blend_mode_str == 'gaussian' else BlendMode.CONSTANT
	print(f"INFO: Using blend mode for inference: {blend_mode_str}")

	prediction_map = run_inference_on_tomogram(
		models=models_to_infer, tomo_path=tomo_info['path'], device=device,
		patch_size=patch_size, use_tta=args.use_tta, precision=args.precision,
		overlap_fraction=config.VALIDATION_OVERLAP_FRACTION,
		sw_batch_size=args.sw_batch_size,
		blend_mode=blend_mode,
		progress_desc=f"Inférence (Grid Search) pour {tomo_info['id']}",
		use_global_norm=use_global_norm,
		norm_percentiles=getattr(args, 'norm_percentiles', None)
	)

	# --- Define the search space for ALL HPs ---
	# Initialize all HP lists with dummy values.
	conf_thresholds = [0.0]
	nms_fractions = [0.0]
	vol_fractions = [0.0]
	cluster_radius_fractions = [0.0]
	min_cluster_vol_fracs = [0.0]
	plm_gpu_peak_threshs = [0.0]

	# Replace dummy lists with the real values from the search space
	# only for the selected post-processing method.
	if args.postprocessing_method == 'cc3d':
		vol_fractions = args.vol_frac_thresholds
		print("INFO: 'cc3d' method selected. HP search will focus on 'vol_frac_threshold'.")
	elif args.postprocessing_method == 'meanshift':
		cluster_radius_fractions = args.ms_cluster_rad_fracs # Uses the specific argument
		min_cluster_vol_fracs = args.ms_min_vol_fracs       # Uses the specific argument
		print("INFO: 'meanshift' method selected. HP search will focus on its specific HPs ('ms_cluster_rad_fracs', 'ms_min_vol_fracs').")
	elif args.postprocessing_method == 'watershed':
		cluster_radius_fractions = args.ws_cluster_rad_fracs # NEW: Uses the watershed-specific argument
		vol_fractions = args.ws_vol_frac_thresholds # NEW: Uses the watershed-specific argument
		print("INFO: 'watershed' method selected. HP search will focus on its specific HPs ('ws_cluster_rad_fracs', 'ws_vol_frac_thresholds').")
	elif args.postprocessing_method == 'peak_local_max_gpu':
		nms_fractions = args.nms_radius_fractions
		plm_gpu_peak_threshs = args.plm_gpu_peak_threshs
		print("INFO: 'peak_local_max_gpu' method selected. HP search will focus on 'plm_gpu_peak_thresh' and 'nms_radius_fraction'.")

	tomo_id_for_run = tomos_to_process[0]['id']
	solution_df_for_run = solution_df[solution_df['experiment'] == tomo_id_for_run]

	# --- Grid search loop (using the single prediction map) ---
	best_params = {}
	best_score_gs = -1.0

	if hp_search_strategy == 'global':
		print("\n--- Starting global grid search ---")
		results_summary = []
		# Dynamic calculation of the total number of iterations
		iter_counts = [len(conf_thresholds), len(nms_fractions), len(vol_fractions), 
					   len(cluster_radius_fractions), len(min_cluster_vol_fracs),
					   len(plm_gpu_peak_threshs)]
		total_iterations = np.prod([c for c in iter_counts if c > 0])
		pbar = tqdm(total=total_iterations, desc="Grid Search (Global)")

		for conf_thresh in conf_thresholds:
			for nms_frac in nms_fractions:
				for vol_frac in vol_fractions:
					for cluster_frac in cluster_radius_fractions:
						for min_vol_frac in min_cluster_vol_fracs:
							for peak_thresh in plm_gpu_peak_threshs:

								# NEW: Use specific keys for overlapping HPs
								current_params = {'conf': conf_thresh, 'nms_frac': nms_frac, 'vol_frac': vol_frac, 'plm_gpu_peak_thresh': peak_thresh}
								if args.postprocessing_method == 'watershed':
									current_params['ws_cluster_rad_frac'] = cluster_frac
								else: # For meanshift or others
									current_params['cluster_radius_frac'] = cluster_frac
								current_params['min_cluster_vol_frac'] = min_vol_frac

								hps_for_this_run = {p_name: current_params for p_name in CZI_PARTICLE_CONFIG.keys()}
								submission_df_gs = process_predictions_to_df(
									prediction_map=prediction_map, tomo_id=tomo_id_for_run, particle_config=CZI_PARTICLE_CONFIG,
									voxel_spacing=config.VOXEL_SPACING, per_class_hps=hps_for_this_run,
									downsampling_factor=config.POSTPROC_DOWNSAMPLING_FACTOR, postprocessing_method=args.postprocessing_method,
									internal_downsample_factor=args.postproc_internal_downsample,
									training_radius_fraction=training_radius_fraction
								)
								score_gs, _, _ = run_czi_evaluation(submission_df_gs, solution_df_for_run, CZI_PARTICLE_CONFIG, beta=config.CZI_EVAL_BETA)
								
								current_params['score'] = score_gs
								results_summary.append(current_params)
								if score_gs > best_score_gs:
									best_score_gs = score_gs
									best_params = {k: v for k, v in current_params.items() if k != 'score'}
								pbar.update(1)
		pbar.close()
		print("\n--- Global grid search summary ---")
		summary_df = pd.DataFrame(results_summary).sort_values(by=['conf', 'nms_frac', 'vol_frac'])
		print(summary_df.to_string(index=False))
		print("\n" + "="*50)
		print(f"Best global parameters found: {best_params} (Score: {best_score_gs:.6f})")
		print("="*50)

	elif hp_search_strategy == 'per_class':
		target_class_for_hp_tuning = getattr(args, 'target_class_for_hp_tuning', None)
		classes_to_optimize = [target_class_for_hp_tuning] if target_class_for_hp_tuning else CZI_PARTICLE_CONFIG.keys()

		for class_name in classes_to_optimize:
			print(f"\n--- Optimizing for class: {class_name} ---")
			best_score_for_class = -1.0
			best_params_for_class = {}
			results_summary = []
			
			iter_counts = [len(conf_thresholds), len(nms_fractions), len(vol_fractions), 
						   len(cluster_radius_fractions), len(min_cluster_vol_fracs),
						   len(plm_gpu_peak_threshs)]
			total_iterations = np.prod([c for c in iter_counts if c > 0])
			pbar = tqdm(total=total_iterations, desc=f"Grid Search ({class_name})")

			for conf_thresh in conf_thresholds:
				for nms_frac in nms_fractions:
					for vol_frac in vol_fractions:
						for cluster_frac in cluster_radius_fractions:
							for min_vol_frac in min_cluster_vol_fracs:
								for peak_thresh in plm_gpu_peak_threshs:

									hps_for_this_run = {}
									# HPs of other classes remain default, we only modify those of the class being optimized.
									current_params = {'conf': conf_thresh, 'nms_frac': nms_frac, 'vol_frac': vol_frac, 'plm_gpu_peak_thresh': peak_thresh}
									if args.postprocessing_method == 'watershed':
										current_params['ws_cluster_rad_frac'] = cluster_frac
									else:
										current_params['cluster_radius_frac'] = cluster_frac
									current_params['min_cluster_vol_frac'] = min_vol_frac
									hps_for_this_run[class_name] = current_params

									submission_df_gs = process_predictions_to_df(
										prediction_map=prediction_map, tomo_id=tomo_id_for_run, particle_config=CZI_PARTICLE_CONFIG,
										voxel_spacing=config.VOXEL_SPACING, per_class_hps=hps_for_this_run,
										downsampling_factor=config.POSTPROC_DOWNSAMPLING_FACTOR, postprocessing_method=args.postprocessing_method,
										internal_downsample_factor=args.postproc_internal_downsample,
										training_radius_fraction=training_radius_fraction
									)
									
									# MODIFICATION: Use the global F-beta score instead of the per-class score.
									# This optimizes the HPs of a class based on their impact on overall performance.
									score_gs, _, _ = run_czi_evaluation(submission_df_gs, solution_df_for_run, CZI_PARTICLE_CONFIG, beta=config.CZI_EVAL_BETA)
									
									summary_row = current_params.copy()
									summary_row['score'] = score_gs
									results_summary.append(summary_row)
									if score_gs > best_score_for_class:
										best_score_for_class = score_gs
										best_params_for_class = current_params
									pbar.update(1)
			pbar.close()
			
			print(f"  -> Best parameters for '{class_name}': {best_params_for_class} (Score: {best_score_for_class:.6f})")
			best_params[class_name] = best_params_for_class
		
		best_score_gs = -1 # Not applicable for per-class search, the score is per class.
	else:
		raise ValueError(f"Unrecognized HP search strategy: {hp_search_strategy}")

	return {
		"score": best_score_gs,
		"time_per_tomo": (pd.Timestamp.now() - start_time).total_seconds(),
		"num_tomos": len(tomos_to_process),
		"best_params": best_params,
		"per_tomo_scores": {tomo_id_for_run: best_score_gs} # The per-tomo score is the search score
	}

def _run_final_evaluation(
	models_to_infer: Union[List[UnetSystem], Dict[str, UnetSystem]],
	tomos_to_process: List[Dict],
	solution_df: pd.DataFrame,
	patch_size: tuple,
	device: torch.device,
	args: argparse.Namespace,
	per_class_hps_from_file: dict,
	use_global_norm: bool,
	training_radius_fraction: float
) -> dict:
	"""Runs the final evaluation with fixed HPs."""
	start_time = pd.Timestamp.now()
	print("\n" + "="*80)
	print("Mode: FINAL EVALUATION (serial processing)")
	print("="*80)

	blend_mode_str = getattr(args, 'blend_mode', 'gaussian')
	blend_mode = BlendMode.GAUSSIAN if blend_mode_str == 'gaussian' else BlendMode.CONSTANT
	print(f"INFO: Using blend mode for inference: {blend_mode_str}")

	all_tomo_scores = {}
	all_per_particle_results = [] # NEW: To collect detailed results for each tomogram
	exp_name = getattr(args, 'exp_name_for_saving', None)
	fold_idx = getattr(args, 'fold_idx_for_saving', None)
	per_tomo_csv_path = getattr(args, 'per_tomo_csv_path_for_saving', None)
	force_rerun = getattr(args, 'force_rerun', False)

	# --- Logic for resuming evaluation ---
	tomos_to_skip = set()
	if per_tomo_csv_path and Path(per_tomo_csv_path).exists() and not force_rerun:
		try:
			per_tomo_df = pd.read_csv(per_tomo_csv_path)
			# Filter for the current experiment and fold
			if exp_name is not None and fold_idx is not None:
				done_tomos_df = per_tomo_df[(per_tomo_df['experiment'] == exp_name) & (per_tomo_df['fold_idx'] == fold_idx)]
				tomos_to_skip = set(done_tomos_df['tomo_id'])
				# Pre-fill scores for already completed tomograms
				for _, row in done_tomos_df.iterrows():
					all_tomo_scores[row['tomo_id']] = row['f4_score']
				if tomos_to_skip:
					print(f"INFO: Resuming evaluation. {len(tomos_to_skip)} tomograms already evaluated for fold {fold_idx} will be skipped.")
		except Exception as e:
			print(f"WARNING: Could not read the per-tomogram scores file for resumption: {e}")

	for tomo_info in tqdm(tomos_to_process, desc="Evaluating tomograms"):
		tomo_id = tomo_info['id']
		if tomo_id in tomos_to_skip:
			tqdm.write(f"--- Skipped (already processed): {tomo_id} ---")
			continue
		tqdm.write(f"\n--- Processing: {tomo_id} ---")

		# 1. Inference
		seg_map = run_inference_on_tomogram(
			models=models_to_infer, tomo_path=tomo_info['path'], device=device,
			patch_size=patch_size, use_tta=args.use_tta, precision=args.precision,
			overlap_fraction=config.VALIDATION_OVERLAP_FRACTION,
			sw_batch_size=args.sw_batch_size,
			blend_mode=blend_mode,
			progress=False, # No progress bar for each tomo in tqdm
			progress_desc=f"Inférence pour {tomo_id}", use_global_norm=use_global_norm,
			norm_percentiles=getattr(args, 'norm_percentiles', None)
		)

		# 2. Post-processing
		submission_df_tomo = process_predictions_to_df(
			prediction_map=seg_map, tomo_id=tomo_id, particle_config=CZI_PARTICLE_CONFIG,
			voxel_spacing=config.VOXEL_SPACING, per_class_hps=per_class_hps_from_file,
			conf_threshold=args.fixed_conf_threshold, nms_radius_fraction=args.fixed_nms_radius_fraction,
			# NEW: Priority to the watershed-specific argument
			vol_fraction_threshold=getattr(args, 'fixed_vol_frac_threshold', None), ws_cluster_rad_frac=getattr(args, 'fixed_ws_cluster_rad_frac', None),
			cluster_radius_fraction=getattr(args, 'fixed_cluster_radius_fraction', None),
			min_cluster_vol_frac=getattr(args, 'fixed_min_cluster_vol_frac', None),
			plm_gpu_peak_thresh=getattr(args, 'fixed_plm_gpu_peak_thresh', None),
			downsampling_factor=config.POSTPROC_DOWNSAMPLING_FACTOR, postprocessing_method=args.postprocessing_method,
			internal_downsample_factor=args.postproc_internal_downsample,
			training_radius_fraction=training_radius_fraction
		)

		# 3. Evaluation
		solution_df_tomo = solution_df[solution_df['experiment'] == tomo_id]
		score_tomo, results_df_tomo, _ = run_czi_evaluation(
			submission_df_tomo, solution_df_tomo, CZI_PARTICLE_CONFIG, beta=config.CZI_EVAL_BETA,
			exp_name=exp_name, fold_idx=fold_idx, per_tomo_scores_csv_path=per_tomo_csv_path
		)
		all_tomo_scores[tomo_id] = score_tomo
		if results_df_tomo is not None and not results_df_tomo.empty:
			all_per_particle_results.append(results_df_tomo)
		tqdm.write(f"  -> Score for {tomo_id}: {score_tomo:.6f}")

	# 4. Aggregation of results
	# NEW: Calculate and save the average F4 and standard deviation per particle
	if all_per_particle_results:
		summary_df = pd.concat(all_per_particle_results, ignore_index=True)
		# Calculate the mean and standard deviation of the f_beta score for each particle type
		stats_df = summary_df.groupby('particle_type')['f_beta'].agg(['mean', 'std']).reset_index()
		stats_df.rename(columns={'mean': 'f4_mean', 'std': 'f4_std'}, inplace=True)
		
		# Save the summary in the main output directory to avoid path issues.
		# The args.output_dir directory (e.g., .../final_eval_fold_X) is not always created.
		main_output_dir = Path(args.output_dir).parent
		summary_filename = main_output_dir / f"{exp_name}_fold_{fold_idx}_f4_summary_by_particle.csv"
		try:
			stats_df.to_csv(str(summary_filename), index=False, float_format='%.6f')
			print("\n" + "="*50)
			print(f"F4 score summary per particle saved to: {summary_filename}")
			print(stats_df.to_string(index=False))
			print("="*50)
		except Exception as e:
			print(f"ERROR: Could not save the per-particle score summary: {e}")
	else:
		print("WARNING: No per-particle results were collected, the summary file will not be created.")

	final_score = np.mean(list(all_tomo_scores.values())) if all_tomo_scores else 0.0
	total_time = (pd.Timestamp.now() - start_time).total_seconds()
	time_per_tomo = total_time / len(tomos_to_process) if tomos_to_process else 0

	print("\n" + "="*50)
	print(f"Final average CZI F{config.CZI_EVAL_BETA} score obtained: {final_score:.6f}")
	if per_class_hps_from_file:
		print("With the provided per-class hyperparameters.")
	else:
		print(f"With fixed parameters: {args.fixed_conf_threshold=}, {args.fixed_nms_radius_fraction=}, {getattr(args, 'fixed_vol_frac_threshold', 'default')=}, ...")
	print(f"Total execution time: {total_time:.2f}s ({time_per_tomo:.2f}s/tomo)")
	print("="*50)

	return {
		"score": final_score,
		"time_per_tomo": time_per_tomo,
		"num_tomos": len(tomos_to_process),
		"best_params": {
			'conf': args.fixed_conf_threshold, 
			'nms_frac': args.fixed_nms_radius_fraction, 
			'vol_frac': getattr(args, 'fixed_vol_frac_threshold', None),
			'ws_cluster_rad_frac': getattr(args, 'fixed_ws_cluster_rad_frac', None),
			'cluster_radius_frac': getattr(args, 'fixed_cluster_radius_fraction', None),
			'min_cluster_vol_frac': getattr(args, 'fixed_min_cluster_vol_frac', None),
			'plm_gpu_peak_thresh': getattr(args, 'fixed_plm_gpu_peak_thresh', None),
		} if not per_class_hps_from_file else per_class_hps_from_file,
		"per_tomo_scores": all_tomo_scores
	}

def perform_evaluation(args: argparse.Namespace) -> dict:
	"""
	Main function that orchestrates the evaluation and returns the results.
	Takes a Namespace object (similar to argparse) as input.
	Returns a dictionary with 'score', 'time_per_tomo', 'num_tomos', 'best_params', 'per_tomo_scores'.
	"""
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")
	torch.backends.cudnn.benchmark = True

	if args.precision == 'bf16':
		dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
	elif args.precision == 'fp16':
		dtype = torch.float16
	else: # fp32
		dtype = torch.float32
	print(f"Requested precision: '{args.precision}', dtype used for inference: {dtype}")

	training_radius_fraction = getattr(args, 'training_radius_fraction', config.RADIUS_FRACTION)

	# 1. Loading models
	models_to_infer, patch_size = _load_models(Path(args.model_path), device, dtype)

	# --- NEW LOGIC TO DETERMINE NORMALIZATION ---
	# By default, normalization is disabled for older models that do not have this parameter.
	model_uses_norm = False 
	if isinstance(models_to_infer, list) and models_to_infer:
		# For generalist or single models
		model_uses_norm = getattr(models_to_infer[0].hparams, 'use_global_norm', False)
	elif isinstance(models_to_infer, dict) and models_to_infer:
		# For specialist ensembles
		first_model = next(iter(models_to_infer.values()))
		model_uses_norm = getattr(first_model.hparams, 'use_global_norm', False)

	# Determine if global normalization should be used for inference.
	# By default, the model's parameter is used for consistency.
	use_global_norm_for_inference = model_uses_norm
	
	# The command-line argument `--use_global_norm` has priority to enable it.
	if getattr(args, 'use_global_norm', False):
		use_global_norm_for_inference = True
		print(f"INFO: Global normalization forced to ON by the --use_global_norm argument (the model was trained with use_global_norm={model_uses_norm}).")
	else:
		print(f"INFO: The model's global normalization parameter (use_global_norm={model_uses_norm}) is used for inference.")

	# --- Determination of tomograms for grid search and final evaluation ---
	print(f"Searching for tomograms in {args.data_root}...")
	all_tomogram_files = find_tomograms(args.data_root, args.tomo_type)
	if not all_tomogram_files:
		print(f"ERROR: No tomograms found. Check the path '{args.data_root}' and the tomogram type '{args.tomo_type}'.")
		return {"score": 0.0, "time_per_tomo": 0, "num_tomos": 0, "best_params": {}}

	if args.limit_tomos_by_id:
		print(f"INFO: Filtering tomograms to keep only IDs: {args.limit_tomos_by_id}")
		all_tomogram_files = [t for t in all_tomogram_files if t['id'] in args.limit_tomos_by_id]
		if not all_tomogram_files:
			print(f"ERROR: No tomograms corresponding to IDs {args.limit_tomos_by_id} were found in {args.data_root}.")
			return {"score": 0.0, "time_per_tomo": 0, "num_tomos": 0, "best_params": {}}

	tomos_to_process = all_tomogram_files
	print(f"INFO: The evaluation will be performed on {len(tomos_to_process)} tomogram(s).")

	# 2. Loading the ground truth
	solution_df = _load_ground_truth(args.data_root)
	if not solution_df.empty:
		print("\n--- Ground Truth (GT) Summary ---")
		gt_counts = solution_df.groupby('experiment').size().reset_index(name='nombre_particules_gt')
		print(gt_counts.to_string(index=False))
		print("-" * 40)

	# 3. Mode determination and execution
	is_grid_search_active = False
	method = args.postprocessing_method
	if method == 'cc3d':
		if getattr(args, 'fixed_vol_frac_threshold', None) is None:
			is_grid_search_active = True
	elif method == 'meanshift':
		if getattr(args, 'fixed_cluster_radius_fraction', None) is None or \
		   getattr(args, 'fixed_min_cluster_vol_frac', None) is None:
			is_grid_search_active = True
	elif method == 'watershed':
		# NEW: Check the watershed-specific argument first
		is_rad_frac_fixed = getattr(args, 'fixed_ws_cluster_rad_frac', None) is not None or getattr(args, 'fixed_cluster_radius_fraction', None) is not None
		if not is_rad_frac_fixed or getattr(args, 'fixed_vol_frac_threshold', None) is None:
			is_grid_search_active = True
	elif method == 'peak_local_max_gpu':
		if getattr(args, 'fixed_nms_radius_fraction', None) is None or \
		   getattr(args, 'fixed_plm_gpu_peak_thresh', None) is None:
			is_grid_search_active = True

	per_class_hps_from_file = None
	if hasattr(args, 'per_class_hp_path') and args.per_class_hp_path:
		try:
			with open(args.per_class_hp_path, 'r') as f:
				per_class_hps_from_file = json.load(f)
			print(f"INFO: Per-class hyperparameters loaded from {args.per_class_hp_path}")
			# If a per-class HP file is provided, no grid search is performed.
			is_grid_search_active = False
		except Exception as e:
			print(f"ERROR: Could not load the per-class HP file: {e}. Aborting.")
			return {"score": 0.0, "time_per_tomo": 0, "num_tomos": 0, "best_params": {}}

	if is_grid_search_active:
		return _run_grid_search(
			models_to_infer=models_to_infer,
			tomos_to_process=tomos_to_process,
			solution_df=solution_df,
			patch_size=patch_size,
			device=device,
			args=args,
			use_global_norm=use_global_norm_for_inference,
			training_radius_fraction=training_radius_fraction
		)
	else:
		return _run_final_evaluation(
			models_to_infer=models_to_infer,
			tomos_to_process=tomos_to_process,
			solution_df=solution_df,
			patch_size=patch_size,
			device=device,
			args=args,
			per_class_hps_from_file=per_class_hps_from_file,
			use_global_norm=use_global_norm_for_inference,
			training_radius_fraction=training_radius_fraction
		)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Script d'inférence et d'évaluation pour un ensemble de modèles U-Net.")
	
	# --- Main Arguments ---
	parser.add_argument('--model_path', type=str, required=True, help="Chemin vers un checkpoint de modèle (.ckpt) ou un dossier contenant plusieurs checkpoints pour l'inférence en ensemble.")
	parser.add_argument('--data_root', type=str, default=config.CZI_DATA_ROOT, help="Répertoire racine du dataset CZI (contenant 'train' ou 'eval').")
	parser.add_argument('--output_dir', type=str, default="inference_results/", help='Dossier de sortie principal.')
	parser.add_argument('--tomo_type', type=str, default=config.TOMO_TYPE, help="Type de tomogramme à utiliser (ex: 'denoised').")
	parser.add_argument('--force_rerun', action='store_true', help="Forcer la ré-exécution de l'inférence même si un cache existe.")
	parser.add_argument('--use_tta', action='store_true', help="Activer le Test-Time Augmentation (flip + rotation) pour une meilleure précision.")
	# --- Grid Search Parameters ---
	parser.add_argument('--conf_thresholds', type=float, nargs='+', default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], help='Liste des seuils de confiance à tester pour la détection.')
	parser.add_argument('--nms_radius_fractions', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], help="Liste des fractions de rayon à utiliser pour la NMS.")
	# --- Parameters for evaluation with fixed values ---
	parser.add_argument('--limit_tomos_by_id', type=str, nargs='+', default=None, help="Limite l'évaluation aux tomogrammes avec les IDs spécifiés.")
	parser.add_argument('--fixed_conf_threshold', type=float, default=None, help='Si fourni, utilise ce seuil de confiance fixe et désactive la recherche par grille pour ce paramètre.')
	parser.add_argument('--fixed_nms_radius_fraction', type=float, default=None, help='Si fourni, utilise cette fraction de rayon NMS fixe et désactive la recherche par grille pour ce paramètre.')
	parser.add_argument('--vol_frac_thresholds', type=float, nargs='+', default=config.VOL_FRAC_GRID_SEARCH, help='Liste des fractions de volume à tester pour cc3d.')
	parser.add_argument('--fixed_vol_frac_threshold', type=float, default=None, help="Si fourni, utilise cette fraction de volume fixe pour cc3d et désactive la recherche par grille pour ce paramètre.")
	parser.add_argument('--per_class_hp_path', type=str, default=None, help="Chemin vers un fichier JSON contenant les hyperparamètres par classe. Désactive la recherche par grille.")
	# --- HPs for meanshift ---
	parser.add_argument('--cluster_radius_fractions', type=float, nargs='+', default=[0.5, 0.7, 1.0, 1.2, 1.5], help="Liste des fractions de rayon à utiliser pour le clustering MeanShift.")
	parser.add_argument('--min_cluster_vol_fracs', type=float, nargs='+', default=config.VOL_FRAC_GRID_SEARCH, help="Liste des fractions de volume minimales de cluster à tester pour le clustering.")
	parser.add_argument('--fixed_cluster_radius_fraction', type=float, default=None, help='Si fourni, utilise cette fraction de rayon de clustering fixe.')
	parser.add_argument('--fixed_min_cluster_vol_frac', type=float, default=None, help='Si fourni, utilise cette fraction de volume de cluster minimale fixe.')
	# --- NEW: Specific search space for meanshift ---
	parser.add_argument('--ms_cluster_rad_fracs', type=float, nargs='+', default=[0.5, 1.0, 1.5], help="Espace de recherche pour 'cluster_radius_fraction' spécifique à MeanShift.")
	parser.add_argument('--ms_min_vol_fracs', type=float, nargs='+', default=[0.01, 0.05, 0.1], help="Espace de recherche pour 'min_cluster_vol_frac' spécifique à MeanShift.") # --- NEW: Specific search space and fixed HP for watershed ---
	parser.add_argument('--ws_cluster_rad_fracs', type=float, nargs='+', default=[0.3, 0.5, 0.8, 1.0, 1.2], help="Espace de recherche pour 'cluster_radius_fraction' spécifique à Watershed.")
	parser.add_argument('--fixed_ws_cluster_rad_frac', type=float, default=None, help="HP fixe pour 'cluster_radius_fraction' spécifique à watershed.")
	parser.add_argument('--ws_vol_frac_thresholds', type=float, nargs='+', default=[0.05, 0.1, 0.2], help="Espace de recherche pour 'vol_frac_threshold' spécifique à Watershed.")
	# --- NEW: Specific search space for watershed ---
	# --- HPs for peak_local_max_gpu ---
	parser.add_argument('--plm_gpu_peak_threshs', type=float, nargs='+', default=[0.05, 0.1, 0.15], help="Seuils de pic finaux pour peak_local_max_gpu.")
	parser.add_argument('--fixed_plm_gpu_peak_thresh', type=float, default=None, help='Valeur fixe pour le seuil de pic de plm_gpu.')
	parser.add_argument(
		'--postproc_internal_downsample',
		type=int,
		default=1,
		help="Facteur de sous-échantillonnage interne supplémentaire à appliquer avant la détection de particules pour le benchmarking. Défaut: 1."
	)
	parser.add_argument(
		'--postprocessing_method',
		type=str,
		default=config.DEFAULT_POSTPROC_METHOD,
		choices=['cc3d', 'meanshift', 'peak_local_max_gpu', 'watershed'],
		help=f"Méthode de post-traitement à utiliser. Défaut: '{config.DEFAULT_POSTPROC_METHOD}'."
	)
	parser.add_argument('--target_class_for_hp_tuning', type=str, default=None, help="Usage interne: limite la recherche d'HP à une seule classe.")
	parser.add_argument(
		'--hp_search_strategy',
		type=str,
		default='per_class',
		choices=['per_class', 'global'],
		help="Stratégie de recherche d'HP: 'per_class' (un jeu d'HP par particule) ou 'global' (un seul jeu d'HP pour toutes)."
	)
	parser.add_argument(
		'--sw_batch_size',
		type=int, default=config.VALIDATION_SW_BATCH_SIZE,
		help="Taille du batch pour l'inférence par fenêtre glissante."
	)

	parser.add_argument(
		'--norm_percentiles', type=float, nargs=2, default=None,
		help="Percentiles (lower upper) pour la normalisation globale. Ex: 0 95. Remplace la valeur par défaut (5, 99)."
	)
	parser.add_argument(
		'--use_global_norm', action='store_true',
		help="Forcer l'activation de la normalisation globale par percentiles."
	)

	parser.add_argument(
		'--precision', 
		type=str, 
		default='bf16', 
		choices=['bf16', 'fp16', 'fp32'],
		help="Précision pour l'inférence: 'bf16' (défaut, fallback sur fp32), 'fp16', ou 'fp32'."
	)
	# This argument is not used directly by the user but is passed by run_experiment_evaluation.py
	parser.add_argument(
		'--training_radius_fraction', type=float, default=None,
		help="Usage interne: spécifie la fraction de rayon utilisée lors de l'entraînement."
	)
	args = parser.parse_args()

	# The direct call is now for command-line executions
	perform_evaluation(args)