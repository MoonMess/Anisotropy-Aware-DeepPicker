#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script unifié pour l'inférence avec un modèle unique ou un ensemble de modèles U-Net,
l'extraction de centroïdes de particules, et l'optimisation des hyperparamètres
de post-traitement via une recherche par grille.
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
# On charge MultiTaskUnetSystem depuis son propre module pour pouvoir charger le checkpoint
from models.system import UnetSystem
from utils.czi_eval import run_czi_evaluation 
from utils.postprocessing import process_predictions_to_df
from utils.inference import run_inference_on_tomogram
from particle_config import CZI_PARTICLE_CONFIG

def find_tomograms(data_root: str, tomo_type: str) -> List[Dict]:
	"""
	Trouve les fichiers tomogrammes (.zarr ou .mrc).
	Gère à la fois la structure des données d'entraînement et celle du jeu d'évaluation public.
	"""
	tomo_list = []
	base_dir = Path(data_root)
	if not base_dir.is_dir():
		raise FileNotFoundError(f"Répertoire de données '{data_root}' non trouvé.")

	# --- Détecter la structure du dossier ---
	# 1. Structure des données d'entraînement (contient 'train/static/ExperimentRuns')
	training_runs_dir = base_dir / "train/static/ExperimentRuns"
	if training_runs_dir.is_dir():
		print("INFO: Détection de la structure des données d'entraînement pour les tomogrammes.")
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

	# 2. Structure des données d'évaluation publiques (contient 'TS_.../Reconstructions')
	print("INFO: Détection de la structure des données d'évaluation publiques pour les tomogrammes.")
	run_ids = sorted([d.name for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("TS_")])
	for run_id in run_ids:
		# Le chemin est fixe pour le jeu d'évaluation public
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
	"""Charge un ou plusieurs modèles depuis un chemin et retourne les modèles et la taille des patchs."""
	if not model_path_arg.exists():
		raise FileNotFoundError(f"Chemin du modèle ou dossier non trouvé : {model_path_arg}")

	models_to_infer: Union[List[UnetSystem], Dict[str, UnetSystem]]
	if model_path_arg.is_dir():
		model_files = list(model_path_arg.rglob('*.ckpt'))
		if not model_files:
			raise FileNotFoundError(f"Aucun fichier de checkpoint (.ckpt) trouvé dans le dossier {model_path_arg}")

		class_names_str = "|".join(config.CLASS_NAMES)
		specialist_pattern = re.compile(rf"-best-({class_names_str})-")
		
		is_specialist_run = any(specialist_pattern.search(p.name) for p in model_files)

		if is_specialist_run:
			print(f"INFO: Détection d'un ensemble de modèles spécialistes. Chargement de {len(model_files)} modèles depuis : {model_path_arg}")
			specialist_models: Dict[str, UnetSystem] = {}
			for path in model_files:
				match = specialist_pattern.search(path.name)
				if not match:
					print(f"AVERTISSEMENT: Fichier {path.name} ignoré car il ne correspond pas au format spécialiste attendu dans ce dossier.")
					continue
				
				class_name = match.group(1)
				model = UnetSystem.load_from_checkpoint(path, map_location=device, weights_only=False, strict=False)
				checkpoint = torch.load(path, map_location=device, weights_only=False)
				if 'ema_state_dict' in checkpoint:
					print(f"  -> INFO: Poids EMA trouvés dans {path.name}. Chargement automatique pour le spécialiste {class_name}.")
					model.load_state_dict(checkpoint['ema_state_dict'])
				elif getattr(model.hparams, 'use_ema', False):
					print(f"  -> AVERTISSEMENT: Le modèle a été entraîné avec EMA mais 'ema_state_dict' n'est pas trouvé dans {path.name}. Utilisation des poids standards.")

				specialist_models[class_name] = model.eval().to(dtype=dtype)
			models_to_infer = specialist_models
		else:
			print(f"INFO: Détection d'un ensemble de modèles généralistes. Chargement de {len(model_files)} modèles pour l'inférence en ensemble depuis : {model_path_arg}")
			generalist_models: List[UnetSystem] = []
			for path in model_files:
				model = UnetSystem.load_from_checkpoint(path, map_location=device, weights_only=False, strict=False)
				checkpoint = torch.load(path, map_location=device, weights_only=False)
				if 'ema_state_dict' in checkpoint:
					print(f"  -> INFO: Poids EMA trouvés dans {path.name}. Chargement automatique.")
					model.load_state_dict(checkpoint['ema_state_dict'])
				elif getattr(model.hparams, 'use_ema', False):
					print(f"  -> AVERTISSEMENT: Le modèle a été entraîné avec EMA mais 'ema_state_dict' n'est pas trouvé dans {path.name}. Utilisation des poids standards.")
				generalist_models.append(model.eval().to(dtype=dtype))
			models_to_infer = generalist_models

	elif model_path_arg.is_dir() and "kfold_ensemble" in model_path_arg.name: # Nouvelle logique pour K-Fold ensemble
		model_files = list(model_path_arg.rglob('*.ckpt'))
		if not model_files:
			raise FileNotFoundError(f"Aucun fichier de checkpoint (.ckpt) trouvé dans le dossier {model_path_arg} pour l'ensemble K-Fold.")
		print(f"INFO: Détection d'un ensemble K-Fold. Chargement de {len(model_files)} modèles depuis : {model_path_arg}")
		kfold_ensemble_models: List[UnetSystem] = []
		for path in model_files:
			kfold_ensemble_models.append(_load_single_model_from_path(path, device, dtype))
		models_to_infer = kfold_ensemble_models
	elif model_path_arg.is_file():
		print(f"Chargement d'un modèle unique depuis : {model_path_arg}")
		model = UnetSystem.load_from_checkpoint(model_path_arg, map_location=device, weights_only=False, strict=False)
		checkpoint = torch.load(model_path_arg, map_location=device, weights_only=False)
		if 'ema_state_dict' in checkpoint:
			print("  -> INFO: Poids EMA trouvés. Chargement automatique.")
			model.load_state_dict(checkpoint['ema_state_dict'])
		elif getattr(model.hparams, 'use_ema', False):
			print("  -> AVERTISSEMENT: Le modèle a été entraîné avec EMA mais 'ema_state_dict' n'est pas trouvé. Utilisation des poids standards.")
		models_to_infer = [model.eval().to(dtype=dtype)]

	print("Modèle(s) chargé(s) et prêt(s) pour l'inférence.")

	patch_size = None
	if isinstance(models_to_infer, list) and models_to_infer:
		patch_size = models_to_infer[0].hparams.patch_size
	elif isinstance(models_to_infer, dict) and models_to_infer:
		patch_size = next(iter(models_to_infer.values())).hparams.patch_size
	
	if patch_size is None:
		raise ValueError("Impossible de déterminer la taille des patchs à partir des modèles chargés.")

	return models_to_infer, patch_size

def _load_single_model_from_path(path: Path, device: torch.device, dtype: torch.dtype) -> UnetSystem:
	model = UnetSystem.load_from_checkpoint(path, map_location=device, weights_only=False, strict=False)
	print("Modèle(s) chargé(s) et prêt(s) pour l'inférence.")

	patch_size = None
	if isinstance(models_to_infer, list) and models_to_infer:
		patch_size = models_to_infer[0].hparams.patch_size
	elif isinstance(models_to_infer, dict) and models_to_infer:
		patch_size = next(iter(models_to_infer.values())).hparams.patch_size
	
	if patch_size is None:
		raise ValueError("Impossible de déterminer la taille des patchs à partir des modèles chargés.")

	return model.eval().to(dtype=dtype)

def _load_ground_truth(data_root: str) -> pd.DataFrame:
	"""Charge et retourne le DataFrame de la vérité terrain."""
	gt_points = []
	gt_root_path = Path(data_root)

	gt_runs_base_dir = gt_root_path
	training_gt_dir = gt_root_path / "train/overlay/ExperimentRuns"
	if training_gt_dir.is_dir():
		print("INFO: Détection de la structure GT des données d'entraînement.")
		gt_runs_base_dir = training_gt_dir
	else:
		print("INFO: Détection de la structure GT des données d'évaluation publiques.")

	if not gt_runs_base_dir.is_dir():
		print(f"AVERTISSEMENT: Répertoire de vérité terrain '{gt_runs_base_dir}' non trouvé. L'évaluation sera ignorée.")
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
			print("AVERTISSEMENT: La valeur maximale des coordonnées de la vérité terrain est très faible.")
			print(f"({max_coord_val:.2f}). Il est très probable que les coordonnées soient en pixels au lieu d'Angstroms.")
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
	"""Exécute la recherche d'hyperparamètres et retourne les meilleurs paramètres."""
	start_time = pd.Timestamp.now()
	hp_search_strategy = getattr(args, 'hp_search_strategy', 'per_class') # Récupère la stratégie

	print("\n" + "="*80)
	print(f"Mode: RECHERCHE D'HYPERPARAMÈTRES (Stratégie: {hp_search_strategy})")
	print("="*80)

	if len(tomos_to_process) > 1:
		print(f"AVERTISSEMENT: La recherche d'HP est conçue pour un seul tomogramme, mais {len(tomos_to_process)} ont été fournis. Seul le premier sera utilisé.")
		tomos_to_process = tomos_to_process[:1]

	# --- Inférence (exécutée une seule fois) ---
	tomo_info = tomos_to_process[0]
	print(f"\n--- Exécution de l'inférence pour : {tomo_info['id']} ---")
	# Préciser quel modèle est utilisé pour l'inférence
	if isinstance(models_to_infer, list):
		model_name = Path(args.model_path).name if len(models_to_infer) == 1 else f"Ensemble de {len(models_to_infer)} modèles"
		print(f"--- Modèle utilisé : {model_name} ---")
	elif isinstance(models_to_infer, dict):
		 print(f"--- Modèles utilisés : Ensemble de {len(models_to_infer)} spécialistes ---")

	blend_mode_str = getattr(args, 'blend_mode', 'gaussian')
	blend_mode = BlendMode.GAUSSIAN if blend_mode_str == 'gaussian' else BlendMode.CONSTANT
	print(f"INFO: Utilisation du mode de fusion pour l'inférence : {blend_mode_str}")

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

	# --- Définir l'espace de recherche pour TOUS les HPs ---
	# Initialiser toutes les listes d'HP avec des valeurs factices.
	conf_thresholds = [0.0]
	nms_fractions = [0.0]
	vol_fractions = [0.0]
	cluster_radius_fractions = [0.0]
	min_cluster_vol_fracs = [0.0]
	plm_gpu_peak_threshs = [0.0]

	# Remplacer les listes factices par les vraies valeurs de l'espace de recherche
	# uniquement pour la méthode de post-traitement sélectionnée.
	if args.postprocessing_method == 'cc3d':
		vol_fractions = args.vol_frac_thresholds
		print("INFO: Méthode 'cc3d' sélectionnée. La recherche d'HP se concentrera sur 'vol_frac_threshold'.")
	elif args.postprocessing_method == 'meanshift':
		cluster_radius_fractions = args.ms_cluster_rad_fracs # Utilise l'argument spécifique
		min_cluster_vol_fracs = args.ms_min_vol_fracs       # Utilise l'argument spécifique
		print("INFO: Méthode 'meanshift' sélectionnée. La recherche d'HP se concentrera sur ses HP spécifiques ('ms_cluster_rad_fracs', 'ms_min_vol_fracs').")
	elif args.postprocessing_method == 'watershed':
		cluster_radius_fractions = args.ws_cluster_rad_fracs # NOUVEAU: Utilise l'argument spécifique à watershed
		vol_fractions = args.ws_vol_frac_thresholds # NOUVEAU: Utilise l'argument spécifique à watershed
		print("INFO: Méthode 'watershed' sélectionnée. La recherche d'HP se concentrera sur ses HP spécifiques ('ws_cluster_rad_fracs', 'ws_vol_frac_thresholds').")
	elif args.postprocessing_method == 'peak_local_max_gpu':
		nms_fractions = args.nms_radius_fractions
		plm_gpu_peak_threshs = args.plm_gpu_peak_threshs
		print("INFO: Méthode 'peak_local_max_gpu' sélectionnée. La recherche d'HP se concentrera sur 'plm_gpu_peak_thresh' et 'nms_radius_fraction'.")

	tomo_id_for_run = tomos_to_process[0]['id']
	solution_df_for_run = solution_df[solution_df['experiment'] == tomo_id_for_run]

	# --- Boucle de recherche par grille (utilisant la carte de prédiction unique) ---
	best_params = {}
	best_score_gs = -1.0

	if hp_search_strategy == 'global':
		print("\n--- Démarrage de la recherche par grille globale ---")
		results_summary = []
		# Calcul dynamique du nombre total d'itérations
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

								# NOUVEAU: Utiliser des clés spécifiques pour les HPs qui se chevauchent
								current_params = {'conf': conf_thresh, 'nms_frac': nms_frac, 'vol_frac': vol_frac, 'plm_gpu_peak_thresh': peak_thresh}
								if args.postprocessing_method == 'watershed':
									current_params['ws_cluster_rad_frac'] = cluster_frac
								else: # Pour meanshift ou autres
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
		print("\n--- Résumé de la recherche par grille globale ---")
		summary_df = pd.DataFrame(results_summary).sort_values(by=['conf', 'nms_frac', 'vol_frac'])
		print(summary_df.to_string(index=False))
		print("\n" + "="*50)
		print(f"Meilleurs paramètres globaux trouvés : {best_params} (Score: {best_score_gs:.6f})")
		print("="*50)

	elif hp_search_strategy == 'per_class':
		target_class_for_hp_tuning = getattr(args, 'target_class_for_hp_tuning', None)
		classes_to_optimize = [target_class_for_hp_tuning] if target_class_for_hp_tuning else CZI_PARTICLE_CONFIG.keys()

		for class_name in classes_to_optimize:
			print(f"\n--- Optimisation pour la classe : {class_name} ---")
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
									# Les HPs des autres classes restent par défaut, on ne modifie que ceux de la classe en cours d'optimisation.
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
									
									# MODIFICATION: Utiliser le score F-beta global au lieu du score par classe.
									# Cela optimise les HPs d'une classe en fonction de leur impact sur la performance globale.
									score_gs, _, _ = run_czi_evaluation(submission_df_gs, solution_df_for_run, CZI_PARTICLE_CONFIG, beta=config.CZI_EVAL_BETA)
									
									summary_row = current_params.copy()
									summary_row['score'] = score_gs
									results_summary.append(summary_row)
									if score_gs > best_score_for_class:
										best_score_for_class = score_gs
										best_params_for_class = current_params
									pbar.update(1)
			pbar.close()
			
			print(f"  -> Meilleurs paramètres pour '{class_name}': {best_params_for_class} (Score: {best_score_for_class:.6f})")
			best_params[class_name] = best_params_for_class
		
		best_score_gs = -1 # Non applicable pour la recherche par classe, le score est par classe.
	else:
		raise ValueError(f"Stratégie de recherche d'HP non reconnue : {hp_search_strategy}")

	return {
		"score": best_score_gs,
		"time_per_tomo": (pd.Timestamp.now() - start_time).total_seconds(),
		"num_tomos": len(tomos_to_process),
		"best_params": best_params,
		"per_tomo_scores": {tomo_id_for_run: best_score_gs} # Le score par tomo est le score de la recherche
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
	"""Exécute l'évaluation finale avec des HP fixes."""
	start_time = pd.Timestamp.now()
	print("\n" + "="*80)
	print("Mode: ÉVALUATION FINALE (traitement en série)")
	print("="*80)

	blend_mode_str = getattr(args, 'blend_mode', 'gaussian')
	blend_mode = BlendMode.GAUSSIAN if blend_mode_str == 'gaussian' else BlendMode.CONSTANT
	print(f"INFO: Utilisation du mode de fusion pour l'inférence : {blend_mode_str}")

	all_tomo_scores = {}
	all_per_particle_results = [] # NOUVEAU: Pour collecter les résultats détaillés de chaque tomogramme
	exp_name = getattr(args, 'exp_name_for_saving', None)
	fold_idx = getattr(args, 'fold_idx_for_saving', None)
	per_tomo_csv_path = getattr(args, 'per_tomo_csv_path_for_saving', None)
	force_rerun = getattr(args, 'force_rerun', False)

	# --- Logique pour la reprise de l'évaluation ---
	tomos_to_skip = set()
	if per_tomo_csv_path and Path(per_tomo_csv_path).exists() and not force_rerun:
		try:
			per_tomo_df = pd.read_csv(per_tomo_csv_path)
			# Filtrer pour l'expérience et le fold actuels
			if exp_name is not None and fold_idx is not None:
				done_tomos_df = per_tomo_df[(per_tomo_df['experiment'] == exp_name) & (per_tomo_df['fold_idx'] == fold_idx)]
				tomos_to_skip = set(done_tomos_df['tomo_id'])
				# Pré-remplir les scores pour les tomogrammes déjà terminés
				for _, row in done_tomos_df.iterrows():
					all_tomo_scores[row['tomo_id']] = row['f4_score']
				if tomos_to_skip:
					print(f"INFO: Reprise de l'évaluation. {len(tomos_to_skip)} tomogrammes déjà évalués pour le fold {fold_idx} seront ignorés.")
		except Exception as e:
			print(f"AVERTISSEMENT: Impossible de lire le fichier des scores par tomogramme pour la reprise : {e}")

	for tomo_info in tqdm(tomos_to_process, desc="Évaluation des tomogrammes"):
		tomo_id = tomo_info['id']
		if tomo_id in tomos_to_skip:
			tqdm.write(f"--- Ignoré (déjà traité) : {tomo_id} ---")
			continue
		tqdm.write(f"\n--- Traitement de : {tomo_id} ---")

		# 1. Inférence
		seg_map = run_inference_on_tomogram(
			models=models_to_infer, tomo_path=tomo_info['path'], device=device,
			patch_size=patch_size, use_tta=args.use_tta, precision=args.precision,
			overlap_fraction=config.VALIDATION_OVERLAP_FRACTION,
			sw_batch_size=args.sw_batch_size,
			blend_mode=blend_mode,
			progress=False, # Pas de barre de progression pour chaque tomo dans tqdm
			progress_desc=f"Inférence pour {tomo_id}", use_global_norm=use_global_norm,
			norm_percentiles=getattr(args, 'norm_percentiles', None)
		)

		# 2. Post-traitement
		submission_df_tomo = process_predictions_to_df(
			prediction_map=seg_map, tomo_id=tomo_id, particle_config=CZI_PARTICLE_CONFIG,
			voxel_spacing=config.VOXEL_SPACING, per_class_hps=per_class_hps_from_file,
			conf_threshold=args.fixed_conf_threshold, nms_radius_fraction=args.fixed_nms_radius_fraction,
			# NOUVEAU: Priorité à l'argument spécifique à watershed
			vol_fraction_threshold=getattr(args, 'fixed_vol_frac_threshold', None), ws_cluster_rad_frac=getattr(args, 'fixed_ws_cluster_rad_frac', None),
			cluster_radius_fraction=getattr(args, 'fixed_cluster_radius_fraction', None),
			min_cluster_vol_frac=getattr(args, 'fixed_min_cluster_vol_frac', None),
			plm_gpu_peak_thresh=getattr(args, 'fixed_plm_gpu_peak_thresh', None),
			downsampling_factor=config.POSTPROC_DOWNSAMPLING_FACTOR, postprocessing_method=args.postprocessing_method,
			internal_downsample_factor=args.postproc_internal_downsample,
			training_radius_fraction=training_radius_fraction
		)

		# 3. Évaluation
		solution_df_tomo = solution_df[solution_df['experiment'] == tomo_id]
		score_tomo, results_df_tomo, _ = run_czi_evaluation(
			submission_df_tomo, solution_df_tomo, CZI_PARTICLE_CONFIG, beta=config.CZI_EVAL_BETA,
			exp_name=exp_name, fold_idx=fold_idx, per_tomo_scores_csv_path=per_tomo_csv_path
		)
		all_tomo_scores[tomo_id] = score_tomo
		if results_df_tomo is not None and not results_df_tomo.empty:
			all_per_particle_results.append(results_df_tomo)
		tqdm.write(f"  -> Score pour {tomo_id}: {score_tomo:.6f}")

	# 4. Agrégation des résultats
	# NOUVEAU: Calculer et sauvegarder le F4 moyen et l'écart-type par particule
	if all_per_particle_results:
		summary_df = pd.concat(all_per_particle_results, ignore_index=True)
		# Calculer la moyenne et l'écart-type du score f_beta pour chaque type de particule
		stats_df = summary_df.groupby('particle_type')['f_beta'].agg(['mean', 'std']).reset_index()
		stats_df.rename(columns={'mean': 'f4_mean', 'std': 'f4_std'}, inplace=True)
		
		# Sauvegarder le résumé dans le dossier de sortie principal pour éviter les problèmes de chemin.
		# Le dossier args.output_dir (ex: .../final_eval_fold_X) n'est pas toujours créé.
		main_output_dir = Path(args.output_dir).parent
		summary_filename = main_output_dir / f"{exp_name}_fold_{fold_idx}_f4_summary_by_particle.csv"
		try:
			stats_df.to_csv(str(summary_filename), index=False, float_format='%.6f')
			print("\n" + "="*50)
			print(f"Résumé du score F4 par particule sauvegardé dans : {summary_filename}")
			print(stats_df.to_string(index=False))
			print("="*50)
		except Exception as e:
			print(f"ERREUR: Impossible de sauvegarder le résumé des scores par particule : {e}")
	else:
		print("AVERTISSEMENT: Aucun résultat par particule n'a été collecté, le fichier de résumé ne sera pas créé.")

	final_score = np.mean(list(all_tomo_scores.values())) if all_tomo_scores else 0.0
	total_time = (pd.Timestamp.now() - start_time).total_seconds()
	time_per_tomo = total_time / len(tomos_to_process) if tomos_to_process else 0

	print("\n" + "="*50)
	print(f"Score final F{config.CZI_EVAL_BETA} CZI moyen obtenu : {final_score:.6f}")
	if per_class_hps_from_file:
		print("Avec les hyperparamètres par classe fournis.")
	else:
		print(f"Avec les paramètres fixes : {args.fixed_conf_threshold=}, {args.fixed_nms_radius_fraction=}, {getattr(args, 'fixed_vol_frac_threshold', 'default')=}, ...")
	print(f"Temps total d'exécution: {total_time:.2f}s ({time_per_tomo:.2f}s/tomo)")
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
	Fonction principale qui orchestre l'évaluation et retourne les résultats.
	Prend un objet Namespace (similaire à argparse) en entrée.
	Retourne un dictionnaire avec 'score', 'time_per_tomo', 'num_tomos', 'best_params', 'per_tomo_scores'.
	"""
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Utilisation du device: {device}")
	torch.backends.cudnn.benchmark = True

	if args.precision == 'bf16':
		dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
	elif args.precision == 'fp16':
		dtype = torch.float16
	else: # fp32
		dtype = torch.float32
	print(f"Précision demandée: '{args.precision}', dtype utilisé pour l'inférence: {dtype}")

	training_radius_fraction = getattr(args, 'training_radius_fraction', config.RADIUS_FRACTION)

	# 1. Chargement des modèles
	models_to_infer, patch_size = _load_models(Path(args.model_path), device, dtype)

	# --- NOUVELLE LOGIQUE POUR DÉTERMINER LA NORMALISATION ---
	# Par défaut, on désactive la normalisation pour les anciens modèles qui n'ont pas ce paramètre.
	model_uses_norm = False 
	if isinstance(models_to_infer, list) and models_to_infer:
		# Pour les modèles généralistes ou uniques
		model_uses_norm = getattr(models_to_infer[0].hparams, 'use_global_norm', False)
	elif isinstance(models_to_infer, dict) and models_to_infer:
		# Pour les ensembles de spécialistes
		first_model = next(iter(models_to_infer.values()))
		model_uses_norm = getattr(first_model.hparams, 'use_global_norm', False)

	# Déterminer si la normalisation globale doit être utilisée pour l'inférence.
	# Par défaut, on utilise le paramètre du modèle pour la cohérence.
	use_global_norm_for_inference = model_uses_norm
	
	# L'argument en ligne de commande `--use_global_norm` a la priorité pour l'activer.
	if getattr(args, 'use_global_norm', False):
		use_global_norm_for_inference = True
		print(f"INFO: Normalisation globale forcée à ON par l'argument --use_global_norm (le modèle a été entraîné avec use_global_norm={model_uses_norm}).")
	else:
		print(f"INFO: Le paramètre de normalisation globale du modèle (use_global_norm={model_uses_norm}) est utilisé pour l'inférence.")

	# --- Détermination des tomogrammes pour la recherche par grille et l'évaluation finale ---
	print(f"Recherche des tomogrammes dans {args.data_root}...")
	all_tomogram_files = find_tomograms(args.data_root, args.tomo_type)
	if not all_tomogram_files:
		print(f"ERREUR: Aucun tomogramme trouvé. Vérifiez le chemin '{args.data_root}' et le type de tomogramme '{args.tomo_type}'.")
		return {"score": 0.0, "time_per_tomo": 0, "num_tomos": 0, "best_params": {}}

	if args.limit_tomos_by_id:
		print(f"INFO: Filtrage des tomogrammes pour ne garder que les IDs: {args.limit_tomos_by_id}")
		all_tomogram_files = [t for t in all_tomogram_files if t['id'] in args.limit_tomos_by_id]
		if not all_tomogram_files:
			print(f"ERREUR: Aucun tomogramme correspondant aux IDs {args.limit_tomos_by_id} n'a été trouvé dans {args.data_root}.")
			return {"score": 0.0, "time_per_tomo": 0, "num_tomos": 0, "best_params": {}}

	tomos_to_process = all_tomogram_files
	print(f"INFO: L'évaluation sera effectuée sur {len(tomos_to_process)} tomogramme(s).")

	# 2. Chargement de la vérité terrain
	solution_df = _load_ground_truth(args.data_root)
	if not solution_df.empty:
		print("\n--- Résumé de la Vérité Terrain (GT) ---")
		gt_counts = solution_df.groupby('experiment').size().reset_index(name='nombre_particules_gt')
		print(gt_counts.to_string(index=False))
		print("-" * 40)

	# 3. Détermination du mode et exécution
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
		# NOUVEAU: Vérifier l'argument spécifique à watershed en premier
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
			print(f"INFO: Hyperparamètres par classe chargés depuis {args.per_class_hp_path}")
			# Si un fichier d'HP par classe est fourni, on ne fait pas de recherche par grille.
			is_grid_search_active = False
		except Exception as e:
			print(f"ERREUR: Impossible de charger le fichier d'HP par classe : {e}. Annulation.")
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
	
	# --- Arguments Principaux ---
	parser.add_argument('--model_path', type=str, required=True, help="Chemin vers un checkpoint de modèle (.ckpt) ou un dossier contenant plusieurs checkpoints pour l'inférence en ensemble.")
	parser.add_argument('--data_root', type=str, default=config.CZI_DATA_ROOT, help="Répertoire racine du dataset CZI (contenant 'train' ou 'eval').")
	parser.add_argument('--output_dir', type=str, default="inference_results/", help='Dossier de sortie principal.')
	parser.add_argument('--tomo_type', type=str, default=config.TOMO_TYPE, help="Type de tomogramme à utiliser (ex: 'denoised').")
	parser.add_argument('--force_rerun', action='store_true', help="Forcer la ré-exécution de l'inférence même si un cache existe.")
	parser.add_argument('--use_tta', action='store_true', help="Activer le Test-Time Augmentation (flip + rotation) pour une meilleure précision.")
	# --- Paramètres pour la recherche par grille ---
	parser.add_argument('--conf_thresholds', type=float, nargs='+', default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], help='Liste des seuils de confiance à tester pour la détection.')
	parser.add_argument('--nms_radius_fractions', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], help="Liste des fractions de rayon à utiliser pour la NMS.")
	# --- Paramètres pour l'évaluation avec des valeurs fixes ---
	parser.add_argument('--limit_tomos_by_id', type=str, nargs='+', default=None, help="Limite l'évaluation aux tomogrammes avec les IDs spécifiés.")
	parser.add_argument('--fixed_conf_threshold', type=float, default=None, help='Si fourni, utilise ce seuil de confiance fixe et désactive la recherche par grille pour ce paramètre.')
	parser.add_argument('--fixed_nms_radius_fraction', type=float, default=None, help='Si fourni, utilise cette fraction de rayon NMS fixe et désactive la recherche par grille pour ce paramètre.')
	parser.add_argument('--vol_frac_thresholds', type=float, nargs='+', default=config.VOL_FRAC_GRID_SEARCH, help='Liste des fractions de volume à tester pour cc3d.')
	parser.add_argument('--fixed_vol_frac_threshold', type=float, default=None, help="Si fourni, utilise cette fraction de volume fixe pour cc3d et désactive la recherche par grille pour ce paramètre.")
	parser.add_argument('--per_class_hp_path', type=str, default=None, help="Chemin vers un fichier JSON contenant les hyperparamètres par classe. Désactive la recherche par grille.")
	# --- HPs pour meanshift ---
	parser.add_argument('--cluster_radius_fractions', type=float, nargs='+', default=[0.5, 0.7, 1.0, 1.2, 1.5], help="Liste des fractions de rayon à utiliser pour le clustering MeanShift.")
	parser.add_argument('--min_cluster_vol_fracs', type=float, nargs='+', default=config.VOL_FRAC_GRID_SEARCH, help="Liste des fractions de volume minimales de cluster à tester pour le clustering.")
	parser.add_argument('--fixed_cluster_radius_fraction', type=float, default=None, help='Si fourni, utilise cette fraction de rayon de clustering fixe.')
	parser.add_argument('--fixed_min_cluster_vol_frac', type=float, default=None, help='Si fourni, utilise cette fraction de volume de cluster minimale fixe.')
	# --- NOUVEAU: Espace de recherche spécifique pour meanshift ---
	parser.add_argument('--ms_cluster_rad_fracs', type=float, nargs='+', default=[0.5, 1.0, 1.5], help="Espace de recherche pour 'cluster_radius_fraction' spécifique à MeanShift.")
	parser.add_argument('--ms_min_vol_fracs', type=float, nargs='+', default=[0.01, 0.05, 0.1], help="Espace de recherche pour 'min_cluster_vol_frac' spécifique à MeanShift.") # --- NOUVEAU: Espace de recherche et HP fixe spécifique pour watershed ---
	parser.add_argument('--ws_cluster_rad_fracs', type=float, nargs='+', default=[0.3, 0.5, 0.8, 1.0, 1.2], help="Espace de recherche pour 'cluster_radius_fraction' spécifique à Watershed.")
	parser.add_argument('--fixed_ws_cluster_rad_frac', type=float, default=None, help="HP fixe pour 'cluster_radius_fraction' spécifique à watershed.")
	parser.add_argument('--ws_vol_frac_thresholds', type=float, nargs='+', default=[0.05, 0.1, 0.2], help="Espace de recherche pour 'vol_frac_threshold' spécifique à Watershed.")
	# --- NOUVEAU: Espace de recherche spécifique pour watershed ---
	# --- HPs pour peak_local_max_gpu ---
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
	# Cet argument n'est pas utilisé directement par l'utilisateur mais est passé par run_experiment_evaluation.py
	parser.add_argument(
		'--training_radius_fraction', type=float, default=None,
		help="Usage interne: spécifie la fraction de rayon utilisée lors de l'entraînement."
	)
	args = parser.parse_args()

	# L'appel direct est maintenant pour les exécutions en ligne de commande
	perform_evaluation(args)