#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lance une campagne d'évaluation pour comparer plusieurs architectures de modèles.

Ce script automatise l'évaluation de modèles de segmentation sur des données de
cryo-tomographie électronique, en utilisant différentes stratégies et en optimisant
les hyperparamètres de post-traitement.

Stratégies supportées :
1.  'best_single': Évalue le meilleur checkpoint de chaque fold K-Fold individuellement.
2.  'kfold_ensemble': Crée un ensemble avec les meilleurs checkpoints de tous les folds
    et l'évalue comme un modèle unique.
3.  'ensemble_specialists': Pour chaque fold, crée un ensemble de modèles "spécialistes"
    (un par classe de particule) et l'évalue.

Le script peut effectuer une recherche par grille pour trouver les hyperparamètres
optimaux de post-traitement sur un tomogramme de validation avant de lancer
l'évaluation finale sur l'ensemble du jeu de données.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import re
import numpy as np
import shutil
from typing import List, Dict, Union, Optional, Tuple
from datetime import datetime
import json

# --- Imports des modules du projet ---
from eval_public_czi import perform_evaluation
from argparse import Namespace
from utils.dataloader import CryoETDataModule
from utils.utils import find_best_generalist_checkpoint
import config as project_config
from particle_config import CZI_PARTICLE_CONFIG

# --- Configuration & Constantes ---
# Colonnes pour les fichiers de résultats
INTERMEDIATE_COLS = ['experiment', 'fold_idx', 'score', 'time_per_tomo', 'num_tomos', 'params_json']
PER_TOMO_COLS = ['experiment', 'fold_idx', 'tomo_id', 'f4_score']

# --- Fonctions de Configuration et de Setup ---

def setup_arg_parser() -> argparse.ArgumentParser:
    """Configure et retourne l'analyseur d'arguments pour le script."""
    parser = argparse.ArgumentParser(description="Lance une campagne d'évaluation pour comparer plusieurs architectures de modèles.")
    
    # --- Arguments principaux ---
    parser.add_argument(
        '--experiments_dir', type=str, default=None,
        help="Dossier contenant les résultats des entraînements. Par défaut: <OUTPUT_ROOT>/checkpoints/."
    )
    parser.add_argument(
        '--experiment_names', type=str, nargs='+', required=True,
        help="Liste des noms de base des expériences à évaluer."
    )
    parser.add_argument(
        '--training_data_root', type=str, default=None,
        help="Chemin vers les données d'entraînement (pour la recherche d'HP). Par défaut: config.CZI_DATA_ROOT."
    )
    parser.add_argument(
        '--final_eval_data_root', type=str, default=None,
        help="Chemin vers les données d'évaluation finale. Par défaut: config.CZI_PUBLIC_EVAL_DATA_ROOT."
    )
    parser.add_argument(
        '--comment', type=str,
        help="Ajoute un commentaire personnalisé au nom du dossier de sortie."
    )

    # --- Arguments de Stratégie et d'Évaluation ---
    parser.add_argument(
        '--evaluation_strategy', type=str, default='best_single',
        choices=['best_single', 'ensemble_specialists', 'kfold_ensemble'],
        help="Stratégie d'évaluation à utiliser."
    )
    parser.add_argument(
        '--use_tta', action='store_true',
        help="Activer le Test-Time Augmentation pour l'inférence."
    )
    parser.add_argument(
        '--blend_mode', type=str, default='gaussian', choices=['gaussian', 'constant'],
        help="Mode de fusion pour sliding_window_inference ('gaussian' ou 'constant')."
    )
    parser.add_argument(
        '--precision', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'],
        help="Précision pour l'inférence."
    )
    parser.add_argument(
        '--force_rerun', action='store_true',
        help="Forcer la ré-exécution même si des résultats intermédiaires existent."
    )
    parser.add_argument(
        '--limit_final_eval_tomos', type=int, default=None,
        help="Mode test rapide : Limite l'évaluation finale à N tomogrammes."
    )

    # --- Arguments de Post-Traitement et Recherche d'HP ---
    parser.add_argument(
        '--postprocessing_method', type=str, default=project_config.DEFAULT_POSTPROC_METHOD,
        choices=['cc3d', 'meanshift', 'peak_local_max_gpu', 'watershed'],
        help=f"Méthode de post-traitement. Défaut: '{project_config.DEFAULT_POSTPROC_METHOD}'."
    )
    parser.add_argument(
        '--hp_search_strategy', type=str, default='global', choices=['per_class', 'global'],
        help="Stratégie de recherche d'HP ('per_class' ou 'global')."
    )
    parser.add_argument(
        '--sw_batch_size', type=int, default=None,
        help="Taille du batch pour l'inférence par fenêtre glissante. Remplace la valeur de config.py."
    )
    parser.add_argument(
		'--postproc_internal_downsample', type=int, default=1,
		help="Facteur de sous-échantillonnage interne pour le post-traitement. Défaut: 1."
	)
    
    # --- Arguments pour la Normalisation ---
    parser.add_argument(
        "--norm_percentiles", type=float, nargs=2, default=None,
        help="Percentiles (lower upper) pour la normalisation globale. Ex: 5 99."
    )
    parser.add_argument(
        "--use_global_norm", action="store_true",
        help="Forcer l'activation de la normalisation globale par percentiles."
    )

    # --- Arguments pour les HP fixes (pour sauter la recherche par grille) ---
    parser.add_argument('--fixed_vol_frac_threshold', type=float, default=None, help="HP fixe pour 'cc3d' et 'watershed'.")
    parser.add_argument('--fixed_cluster_radius_fraction', type=float, default=None, help="HP fixe pour 'meanshift' et 'watershed'.")
    parser.add_argument('--fixed_min_cluster_vol_frac', type=float, default=None, help="HP fixe pour 'meanshift'.")
    parser.add_argument('--fixed_plm_gpu_peak_thresh', type=float, default=None, help="HP fixe pour 'peak_local_max_gpu'.")
    # NOUVEAU: HP fixe spécifique pour watershed
    parser.add_argument('--fixed_ws_cluster_rad_frac', type=float, default=None, help="HP fixe pour 'cluster_radius_fraction' spécifique à watershed.")
    parser.add_argument('--fixed_nms_radius_fraction', type=float, default=None, help="HP fixe pour 'peak_local_max_gpu'.")

    # --- Arguments pour l'espace de recherche des HP ---
    parser.add_argument('--vol_frac_thresholds', type=float, nargs='+', default=project_config.VOL_FRAC_GRID_SEARCH)
    parser.add_argument('--cluster_radius_fractions', type=float, nargs='+', default=[0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2])
    parser.add_argument('--min_cluster_vol_fracs', type=float, nargs='+', default=project_config.VOL_FRAC_GRID_SEARCH)
    parser.add_argument('--plm_gpu_peak_threshs', type=float, nargs='+', default=np.round(np.arange(0.01, 1.01, 0.05), 3).tolist())
    parser.add_argument('--nms_radius_fractions', type=float, nargs='+', default=[0.3, 0.50, 0.80, 1.0, 1.2, 1.5, 2])
    # --- NOUVEAU: Arguments spécifiques pour l'espace de recherche de MeanShift ---
    parser.add_argument('--ms_cluster_rad_fracs', type=float, nargs='+', default=[0.8, 1.0, 1.2], help="Espace de recherche pour 'cluster_radius_fraction' spécifique à MeanShift.")
    parser.add_argument('--ms_min_vol_fracs', type=float, nargs='+', default=[0.1, 0.2, 0.3], help="Espace de recherche pour 'min_cluster_vol_frac' spécifique à MeanShift.")
    parser.add_argument('--ws_cluster_rad_fracs', type=float, nargs='+', default=[0.1, 0.2, 0.3], help="Espace de recherche pour 'cluster_radius_fraction' spécifique à Watershed.")
    parser.add_argument('--ws_vol_frac_thresholds', type=float, nargs='+', default=[0.8, 1.0, 1.2], help="Espace de recherche pour 'vol_frac_threshold' spécifique à Watershed.")


    # --- Arguments de débogage ---
    parser.add_argument('--debug_mode', action='store_true', help="Activer le mode de débogage (génère de faux résultats).")
    
    return parser

def resolve_paths(args: argparse.Namespace) -> Tuple[str, str, str]:
    """Résout et valide les chemins principaux (données, checkpoints)."""
    try:
        # Utiliser les chemins de config.py comme base
        default_experiments_dir = str(Path(project_config.OUTPUT_ROOT) / 'checkpoints')
        default_training_data_root = project_config.CZI_DATA_ROOT
        default_final_eval_data_root = project_config.CZI_PUBLIC_EVAL_DATA_ROOT
    except (ImportError, AttributeError):
        print("ERREUR: Le fichier 'config.py' est introuvable ou incomplet. Des chemins par défaut sont manquants.")
        sys.exit(1)

    experiments_dir = args.experiments_dir or default_experiments_dir
    training_data_root = args.training_data_root or default_training_data_root
    final_eval_data_root = args.final_eval_data_root or default_final_eval_data_root

    if not Path(experiments_dir).is_dir():
        print(f"ERREUR: Le dossier d'expériences '{experiments_dir}' n'existe pas.")
        sys.exit(1)

    return experiments_dir, training_data_root, final_eval_data_root

def setup_output_directory(args: argparse.Namespace) -> Path:
    """Crée le dossier de sortie pour l'exécution et retourne son chemin."""
    # --- Construire les composants du nom ---
    strategy_str = args.evaluation_strategy
    postproc_str = args.postprocessing_method
    hp_search_str = f"_{args.hp_search_strategy}" if args.hp_search_strategy == 'per_class' else ""
    tta_str = "_tta" if args.use_tta else ""
    comment_str = f"_{args.comment}" if args.comment else ""

    if len(args.experiment_names) == 1:
        base_name = args.experiment_names[0]
        # Nouveau format: {base_name}_{strategy}_{postproc}{hp_search}{_tta}{_comment}
        output_dirname = f"results/{base_name}_{strategy_str}_{postproc_str}{hp_search_str}{tta_str}{comment_str}"
        print(f"INFO: L'évaluation porte sur une seule expérience. Le dossier de résultats sera '{output_dirname}'.")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "comparison_DEBUG" if args.debug_mode else "comparison"
        # Nouveau format: {prefix}_{strategy}_{postproc}{hp_search}{_tta}{_comment}_{timestamp}
        output_dirname = f"results/{prefix}_{strategy_str}_{postproc_str}{hp_search_str}{tta_str}{comment_str}_{timestamp}"
    
    output_dir_path = Path(output_dirname)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"INFO: Le dossier de sortie pour cette exécution est : {output_dir_path}")
    return output_dir_path

def save_run_parameters(output_dir_path: Path, args: argparse.Namespace, training_data_root: str, final_eval_data_root: str):
    """Sauvegarde les paramètres de l'exécution dans un fichier JSON au début du run."""
    run_info = {
        "timestamp_start": datetime.now().isoformat(),
        "evaluation_strategy": args.evaluation_strategy,
        "postprocessing_method": args.postprocessing_method,
        "hp_search_strategy": args.hp_search_strategy,
        "compared_experiments": args.experiment_names,
        "training_data_root": training_data_root,
        "final_eval_data_root": final_eval_data_root,
        "use_tta": args.use_tta,
        "precision": args.precision,
        "full_command": " ".join(sys.argv),
    }
    info_path = output_dir_path / "run_parameters.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(run_info, f, indent=4, ensure_ascii=False)
    print(f"INFO: Paramètres de l'exécution sauvegardés dans : {info_path}")

def initialize_results_files(output_dir_path: Path, force_rerun: bool) -> Tuple[Path, Path]:
    """Crée ou réinitialise les fichiers CSV pour les résultats."""
    intermediate_results_path = output_dir_path / "intermediate_results.csv"
    if not intermediate_results_path.exists() or force_rerun:
        pd.DataFrame(columns=INTERMEDIATE_COLS).to_csv(intermediate_results_path, index=False)
        print("INFO: Fichier de résultats intermédiaires créé.")

    per_tomo_results_path = output_dir_path / "per_tomogram_scores.csv"
    if not per_tomo_results_path.exists() or force_rerun:
        pd.DataFrame(columns=PER_TOMO_COLS).to_csv(per_tomo_results_path, index=False)
        print("INFO: Fichier de scores par tomogramme créé.")
    
    print(f"INFO: Les résultats intermédiaires seront sauvegardés dans : {intermediate_results_path.name}")
    return intermediate_results_path, per_tomo_results_path

def get_tomo_id_mapping(training_data_root: str) -> List[str]:
    """Récupère la liste ordonnée des IDs de tomogrammes d'entraînement pour mapper les folds."""
    print("INFO: Recherche des tomogrammes dans le jeu de données d'entraînement pour mapper les folds...")
    try:
        temp_dm = CryoETDataModule(
            czi_data_root=training_data_root, label_dir=Path(training_data_root) / "mask",
            patch_size=project_config.PATCH_SIZE, batch_size=1, validation_sw_batch_size=1, num_workers=0,
            val_tomo_id="dummy", tomo_type="dummy", train_steps_per_epoch=1,
            radius_fraction=project_config.RADIUS_FRACTION, augmentation_level='none'
        )
        all_training_files = temp_dm._create_file_list()
        all_tomo_ids = sorted([f['run_id'] for f in all_training_files])
    except Exception as e:
        print(f"ERREUR: Impossible de charger les données d'entraînement pour mapper les folds. Erreur: {e}")
        sys.exit(1)

    if not all_tomo_ids:
        print(f"ERREUR: Aucun tomogramme trouvé dans '{training_data_root}'. Impossible de mapper les folds.")
        sys.exit(1)
    
    print(f"INFO: Tomogrammes d'entraînement trouvés pour le mapping des folds: {all_tomo_ids}")
    return all_tomo_ids

def get_models_for_evaluation(exp_path: Path, strategy: str) -> Dict[Union[int, str], Path]:
    """
    Trouve les checkpoints à évaluer pour une expérience donnée, selon la stratégie.

    Retourne:
        Un dictionnaire mappant un identifiant de fold (int) ou un nom ('ensemble')
        au chemin du modèle (Path) à évaluer.
    """
    all_ckpts = sorted(list(exp_path.glob("*.ckpt")))
    fold_groups = {}
    for ckpt_path in all_ckpts:
        match = re.search(r"fold_(\d+)-", ckpt_path.name)
        if match:
            fold_idx = int(match.group(1))
            if fold_idx not in fold_groups:
                fold_groups[fold_idx] = []
            fold_groups[fold_idx].append(str(ckpt_path))

    if not fold_groups:
        print(f"AVERTISSEMENT: Aucun checkpoint de fold trouvé pour l'expérience {exp_path.name}.")
        return {}

    models_to_evaluate = {}
    if strategy == 'best_single':
        print("\nINFO: Sélection du meilleur checkpoint unique pour chaque fold...")
        for fold_idx in sorted(fold_groups.keys()):
            fold_ckpt_paths_str = fold_groups.get(fold_idx, [])
            fold_ckpt_paths = [Path(p) for p in fold_ckpt_paths_str]
            result = find_best_generalist_checkpoint(fold_ckpt_paths)
            
            if result:
                best_ckpt_path, metric_type, metric_value = result
                print(f"  - Fold {fold_idx}: Meilleur checkpoint -> {best_ckpt_path.name} ({metric_type}={metric_value:.4f})")
                models_to_evaluate[fold_idx] = best_ckpt_path
            else:
                print(f"  AVERTISSEMENT: Impossible de trouver le meilleur checkpoint pour le fold {fold_idx}.")

    elif strategy == 'kfold_ensemble':
        print("\nINFO: Préparation de l'ensemble K-Fold (meilleurs modèles généralistes de chaque fold)...")
        kfold_ensemble_paths = []
        for fold_idx in sorted(fold_groups.keys()):
            fold_ckpt_paths_str = fold_groups.get(fold_idx, [])
            fold_ckpt_paths = [Path(p) for p in fold_ckpt_paths_str]
            result = find_best_generalist_checkpoint(fold_ckpt_paths)
            
            if result:
                best_ckpt_path, metric_type, metric_value = result
                print(f"  - Fold {fold_idx}: Inclus -> {best_ckpt_path.name} ({metric_type}={metric_value:.4f})")
                kfold_ensemble_paths.append(best_ckpt_path)
    
        if kfold_ensemble_paths:
            kfold_ensemble_dir = exp_path / "kfold_ensemble_temp_dir"
            if kfold_ensemble_dir.exists(): shutil.rmtree(kfold_ensemble_dir)
            kfold_ensemble_dir.mkdir(exist_ok=True)
            for model_path in kfold_ensemble_paths:
                shutil.copy(model_path, kfold_ensemble_dir)
            models_to_evaluate['ensemble'] = kfold_ensemble_dir
            print(f"INFO: L'ensemble K-Fold sera évalué comme un seul modèle depuis {kfold_ensemble_dir.name}.")

    elif strategy == 'ensemble_specialists':
        print("\nINFO: Sélection des modèles spécialistes pour chaque fold...")
        for fold_idx in sorted(fold_groups.keys()):
            specialist_ensemble_dir = exp_path / f"specialist_ensemble_fold_{fold_idx}"
            if specialist_ensemble_dir.exists(): shutil.rmtree(specialist_ensemble_dir)
            specialist_ensemble_dir.mkdir(exist_ok=True)
            
            found_specialists = False
            for class_name in project_config.CLASS_NAMES:
                specialist_ckpts = list(exp_path.glob(f"fold_{fold_idx}-best-{class_name}-*.ckpt"))
                if specialist_ckpts:
                    shutil.copy(specialist_ckpts[0], specialist_ensemble_dir)
                    found_specialists = True
            
            if found_specialists:
                print(f"  - Fold {fold_idx}: Trouvé des modèles spécialistes, copiés dans {specialist_ensemble_dir.name}.")
                models_to_evaluate[fold_idx] = specialist_ensemble_dir

    return models_to_evaluate

def is_grid_search_needed(args: argparse.Namespace) -> bool:
    """Détermine si une recherche d'HP est nécessaire en fonction des arguments fixes."""
    method = args.postprocessing_method
    if method == 'cc3d':
        # La recherche est nécessaire si le HP spécifique à cc3d n'est pas fixé.
        return getattr(args, 'fixed_vol_frac_threshold', None) is None
    elif method == 'meanshift':
        # La recherche est nécessaire si l'un des HPs de meanshift n'est pas fixé.
        return getattr(args, 'fixed_cluster_radius_fraction', None) is None or \
               getattr(args, 'fixed_min_cluster_vol_frac', None) is None
    elif method == 'watershed':
        # La recherche est nécessaire si l'un des HPs de watershed n'est pas fixé.
        return (getattr(args, 'fixed_ws_cluster_rad_frac', None) is None and getattr(args, 'fixed_cluster_radius_fraction', None) is None) or \
                getattr(args, 'fixed_vol_frac_threshold', None) is None
    elif method == 'peak_local_max_gpu':
        # La recherche est nécessaire si l'un des HPs de peak_local_max_gpu n'est pas fixé.
        return getattr(args, 'fixed_nms_radius_fraction', None) is None or \
               getattr(args, 'fixed_plm_gpu_peak_thresh', None) is None
    
    # Par défaut, si la méthode est inconnue, on suppose qu'aucune recherche n'est nécessaire.
    return False

def run_hp_search_for_model(model_path: Path, val_tomo_id: str, output_dir_path: Path, args: argparse.Namespace) -> Dict:
    """Exécute la recherche d'HP pour un modèle donné et retourne les meilleurs paramètres."""
    print(f"\n--- Recherche d'HP sur le tomogramme de validation '{val_tomo_id}' ---")
    
    args_hp = Namespace(
        model_path=str(model_path),
        data_root=args.training_data_root,
        output_dir=str(output_dir_path),
        limit_tomos_by_id=[val_tomo_id],
        use_tta=args.use_tta,
        precision=args.precision,
        force_rerun=args.force_rerun,
        tomo_type='denoised',
        # Espace de recherche
        vol_frac_thresholds=args.vol_frac_thresholds,
        cluster_radius_fractions=args.cluster_radius_fractions,
        min_cluster_vol_fracs=args.min_cluster_vol_fracs,
        plm_gpu_peak_threshs=args.plm_gpu_peak_threshs,
        nms_radius_fractions=args.nms_radius_fractions,
        # Nouveaux arguments spécifiques
        ms_cluster_rad_fracs=args.ms_cluster_rad_fracs,
        ms_min_vol_fracs=args.ms_min_vol_fracs,
        ws_vol_frac_thresholds=args.ws_vol_frac_thresholds,
        ws_cluster_rad_fracs=args.ws_cluster_rad_fracs,
        # Paramètres fixes (doivent être None pour la recherche)
        fixed_conf_threshold=None, fixed_nms_radius_fraction=None, fixed_vol_frac_threshold=None, fixed_ws_cluster_rad_frac=None,
        fixed_cluster_radius_fraction=None, fixed_min_cluster_vol_frac=None, fixed_plm_gpu_peak_thresh=None,
        # Autres paramètres
        target_class_for_hp_tuning=None, # Géré par la stratégie 'ensemble_specialists' si besoin
        per_class_hp_path=None,
        postprocessing_method=args.postprocessing_method,
        norm_percentiles=args.norm_percentiles,
        use_global_norm=args.use_global_norm,
        hp_search_strategy=args.hp_search_strategy,
        postproc_internal_downsample=args.postproc_internal_downsample,
        sw_batch_size=args.sw_batch_size or project_config.VALIDATION_SW_BATCH_SIZE,
        blend_mode=args.blend_mode,
        training_radius_fraction=getattr(args, 'training_radius_fraction', project_config.RADIUS_FRACTION)
        )
    
    result_hp = perform_evaluation(args_hp)
    best_params = result_hp.get('best_params', {})
    
    if best_params:
        print(f"--- Meilleurs HP trouvés : {best_params} ---")
    else:
        print("AVERTISSEMENT: Échec de la recherche d'HP.")
        
    return best_params

def run_final_evaluation_for_model(
    model_path: Path,
    best_params: Dict,
    output_dir_path: Path,
    exp_name: str,
    fold_id: Union[int, str],
    per_tomo_results_path: Path,
    args: argparse.Namespace
) -> Dict:
    """Exécute l'évaluation finale pour un modèle avec des HP fixes."""
    print(f"\n--- Évaluation finale du modèle pour le fold '{fold_id}' ---")
    
    final_per_class_hp_path = None
    
    # Si les meilleurs paramètres sont structurés par classe
    if args.hp_search_strategy == 'per_class' and best_params and isinstance(next(iter(best_params.values()), None), dict):
        per_class_hp_path = output_dir_path / f"hps_{exp_name}_fold_{fold_id}.json"
        with open(per_class_hp_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"  -> Fichier d'HP par classe sauvegardé : {per_class_hp_path.name}")
        final_per_class_hp_path = str(per_class_hp_path)
        # Les HP globaux sont mis à None car le fichier par classe a la priorité
        hps_global = {
            'vol_frac': None, 'nms_frac': None, 'cluster_radius_frac': None,
            'min_cluster_vol_frac': None, 'plm_gpu_peak_thresh': None, 'ws_cluster_rad_frac': None
        }
    else: # HP globaux
        hps_global = {
            'vol_frac': best_params.get('vol_frac'),
            'nms_frac': best_params.get('nms_frac'),
            'cluster_radius_frac': best_params.get('cluster_radius_frac'),
            'min_cluster_vol_frac': best_params.get('min_cluster_vol_frac'),
            'plm_gpu_peak_thresh': best_params.get('plm_gpu_peak_thresh'),
            'ws_cluster_rad_frac': best_params.get('ws_cluster_rad_frac')
        }

    args_final = Namespace(
        model_path=str(model_path),
        data_root=args.final_eval_data_root,
        output_dir=str(output_dir_path / f"final_eval_fold_{fold_id}"),
        limit_tomos_by_id=args.final_eval_tomo_ids_limit,
        use_tta=args.use_tta,
        precision=args.precision,
        force_rerun=args.force_rerun,
        tomo_type='denoised',
        # HP fixes
        fixed_conf_threshold=None,
        fixed_vol_frac_threshold=hps_global['vol_frac'],
        fixed_nms_radius_fraction=hps_global['nms_frac'], # Cet HP n'est pas utilisé par watershed mais on le passe quand même
        fixed_ws_cluster_rad_frac=hps_global['ws_cluster_rad_frac'],
        fixed_cluster_radius_fraction=hps_global['cluster_radius_frac'],
        fixed_min_cluster_vol_frac=hps_global['min_cluster_vol_frac'],
        fixed_plm_gpu_peak_thresh=hps_global['plm_gpu_peak_thresh'],
        # Chemin HP par classe
        per_class_hp_path=final_per_class_hp_path,
        # Autres paramètres
        postprocessing_method=args.postprocessing_method,
        norm_percentiles=args.norm_percentiles,
        use_global_norm=args.use_global_norm,
        postproc_internal_downsample=args.postproc_internal_downsample,
        sw_batch_size=args.sw_batch_size or project_config.VALIDATION_SW_BATCH_SIZE,
        # Paramètres pour la sauvegarde incrémentale
        exp_name_for_saving=exp_name,
        fold_idx_for_saving=fold_id,
        per_tomo_csv_path_for_saving=str(per_tomo_results_path),
        blend_mode=args.blend_mode,
        training_radius_fraction=getattr(args, 'training_radius_fraction', project_config.RADIUS_FRACTION)
    )    
    
    return perform_evaluation(args_final)

def aggregate_and_save_final_results(intermediate_results_path: Path, output_dir_path: Path, evaluation_strategy: str):
    """Agrège les résultats intermédiaires et sauvegarde le résumé final."""
    if not intermediate_results_path.exists():
        print("AVERTISSEMENT: Fichier de résultats intermédiaires non trouvé. Impossible de générer le résumé.")
        return

    final_df = pd.read_csv(intermediate_results_path, keep_default_na=False, na_values=[''])
    if final_df.empty:
        print("AVERTISSEMENT: Aucun résultat à agréger.")
        return

    # --- Aplatir les HP depuis la colonne JSON (version optimisée) ---
    # L'ancienne méthode avec df.apply() était très lente. Cette nouvelle approche
    # vectorisée est beaucoup plus performante.
    if 'params_json' in final_df.columns:
        # Remplacer les valeurs non valides par une chaîne JSON vide et parser
        params_series = final_df['params_json'].fillna('{}').astype(str)
        params_list = [json.loads(p) if p and p.strip() else {} for p in params_series]

        if params_list:
            # Déterminer si les HPs sont globaux ou par classe
            first_valid_param = next((p for p in params_list if p), None)
            is_per_class = False
            if first_valid_param and isinstance(first_valid_param, dict):
                is_per_class = any(isinstance(v, dict) for v in first_valid_param.values())

            if is_per_class:
                # Gérer la structure imbriquée {"class_name": {"hp_name": hp_val}}
                flat_params_list = []
                for params_dict in params_list:
                    row_params = {}
                    if isinstance(params_dict, dict):
                        for class_name, hps in params_dict.items():
                            if isinstance(hps, dict):
                                for hp_name, hp_val in hps.items():
                                    row_params[f'{class_name}_{hp_name}'] = hp_val
                    flat_params_list.append(row_params)
                params_df = pd.DataFrame(flat_params_list, index=final_df.index)
            else:
                # Gérer la structure plate {"hp_name": hp_val} avec json_normalize
                params_df = pd.json_normalize(params_list)
                params_df.index = final_df.index
            
            # Concaténer le nouveau DataFrame et supprimer l'ancienne colonne
            final_df = pd.concat([final_df, params_df], axis=1)

    final_df.drop(columns=['params_json'], inplace=True, errors='ignore')

    all_results = []
    for exp_name, group in final_df.groupby('experiment'):
        scores = group['score'].tolist()
        
        num_tomos_evaluated = int(group['num_tomos'].iloc[0]) if not group.empty else 0

        result_row: Dict[str, Union[str, float, int]] = {
            'Architecture': exp_name, 
            'Mean F-beta Score': np.mean(scores), 
            'Std Dev F-beta': np.std(scores), 
            'Mean Time/Tomo (s)': np.mean(group['time_per_tomo']),
            'Num Tomos Evaluated': num_tomos_evaluated,
            'Num Folds': len(scores)
        }
        
        if not group.empty:
            best_fold_row = group.loc[group['score'].idxmax()]
            result_row['Best Score'] = best_fold_row['score']
            result_row['Best Fold'] = int(best_fold_row['fold_idx'])

        # Ajouter les scores et HP de chaque fold
        for _, row in group.sort_values('fold_idx').iterrows():
            fold_idx = int(row['fold_idx'])
            result_row[f'Fold_{fold_idx}_Score'] = row['score']
            for col, val in row.items():
                if col not in INTERMEDIATE_COLS:
                    result_row[f'Fold_{fold_idx}_{col}'] = val
        all_results.append(result_row)

    # --- Sauvegarde finale ---
    if not all_results:
        print("AVERTISSEMENT: Aucun résultat agrégé à sauvegarder.")
        return

    results_df = pd.DataFrame(all_results).fillna('')
    
    # Organisation des colonnes pour la lisibilité
    ideal_cols = ['Architecture', 'Mean F-beta Score', 'Std Dev F-beta', 'Best Score', 'Best Fold', 'Mean Time/Tomo (s)', 'Num Tomos Evaluated', 'Num Folds']
    main_cols = [col for col in ideal_cols if col in results_df.columns]
    fold_cols = sorted([c for c in results_df.columns if c.startswith('Fold_')], key=lambda c: (int(c.split('_')[1]), c))
    final_cols = main_cols + fold_cols
    results_df = results_df[final_cols]

    print("\n\n" + "="*80 + "\nRÉCAPITULATIF FINAL DE LA COMPARAISON\n" + "="*80)
    print(results_df.to_markdown(index=False, floatfmt=".6f"))

    output_csv_path = output_dir_path / "comparison_summary.csv"
    results_df.to_csv(output_csv_path, index=False, float_format="%.6f")
    print(f"\nRésumé final sauvegardé dans : {output_csv_path}")

    # --- NOUVELLE LOGIQUE: Agrégation des scores par particule sur tous les folds ---
    all_particle_summary_files = list(output_dir_path.glob("*_f4_summary_by_particle.csv"))
    if all_particle_summary_files:
        print("\n--- Agrégation des scores F4 par particule sur l'ensemble des folds ---")
        all_particle_dfs = []
        for f in all_particle_summary_files:
            # Extraire le nom de l'expérience et le fold_idx du nom de fichier
            match = re.search(r"(.+)_fold_(\d+)_f4_summary_by_particle.csv", f.name)
            if match:
                exp_name = match.group(1)
                fold_idx = int(match.group(2))
                try:
                    df = pd.read_csv(f)
                    if not df.empty:
                        df['experiment'] = exp_name
                        df['fold_idx'] = fold_idx
                        all_particle_dfs.append(df)
                except pd.errors.EmptyDataError:
                    print(f"AVERTISSEMENT: Le fichier de résumé par particule {f.name} est vide et sera ignoré.")

        if all_particle_dfs:
            full_particle_summary_df = pd.concat(all_particle_dfs, ignore_index=True)
            
            agg_particle_df = full_particle_summary_df.groupby(['experiment', 'particle_type'])['f4_mean'].agg(['mean', 'std']).reset_index()
            agg_particle_df.rename(columns={'mean': 'mean_f4_across_folds', 'std': 'std_f4_across_folds'}, inplace=True)
            
            final_particle_summary_path = output_dir_path / "comparison_summary_by_particle.csv"
            agg_particle_df.to_csv(final_particle_summary_path, index=False, float_format="%.6f")
            
            print(f"\nRésumé des scores par particule sur tous les folds sauvegardé dans : {final_particle_summary_path}")
            print(agg_particle_df.to_string(index=False))

# --- Fonction Principale ---

def main():
    """Orchestre le processus d'évaluation."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    args.experiments_dir, args.training_data_root, args.final_eval_data_root = resolve_paths(args)
    
    # --- Mode test rapide ---
    args.final_eval_tomo_ids_limit = None
    if args.limit_final_eval_tomos is not None:
        print(f"INFO: Mode test rapide activé. Limite à {args.limit_final_eval_tomos} tomogrammes.")
        all_ids = sorted([d.name for d in Path(args.final_eval_data_root).iterdir() if d.is_dir() and d.name.startswith("TS_")])
        args.final_eval_tomo_ids_limit = all_ids[:args.limit_final_eval_tomos]
        print(f"INFO: Tomogrammes pour le test : {args.final_eval_tomo_ids_limit}")

    output_dir_path = setup_output_directory(args)
    save_run_parameters(output_dir_path, args, args.training_data_root, args.final_eval_data_root)
    intermediate_results_path, per_tomo_results_path = initialize_results_files(output_dir_path, args.force_rerun)

    if args.debug_mode:
        print("\nMODE DÉBOGAGE ACTIVÉ. Fin du script.")
        sys.exit(0)

    all_tomo_ids = get_tomo_id_mapping(args.training_data_root)

    for exp_name in args.experiment_names:
        exp_path = Path(args.experiments_dir) / exp_name
        if not exp_path.is_dir():
            print(f"AVERTISSEMENT: Dossier d'expérience '{exp_path}' non trouvé. Ignoré.")
            continue

        print("\n" + "="*80)
        print(f"Traitement de l'expérience : {exp_name}")
        print(f"Stratégie: '{args.evaluation_strategy}'")


        # Déterminer la fraction de rayon utilisée pour l'entraînement
        training_radius_fraction = project_config.RADIUS_FRACTION # Valeur par défaut
        config_path = exp_path / 'effective_training_config.json'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    train_config = json.load(f)
                
                cmd_args = train_config.get('command_line_args', {})
                config_params = train_config.get('effective_config_params', {})

                if cmd_args.get('radius_fraction') is not None:
                    training_radius_fraction = cmd_args['radius_fraction']
                    print(f"INFO: Fraction de rayon de l'entraînement ({training_radius_fraction}) trouvée dans les arguments de la ligne de commande du training.")
                elif cmd_args.get('single_pixel_mask'):
                    training_radius_fraction = 0.0
                    print("INFO: Masque d'un seul pixel détecté pour l'entraînement. Utilisation de radius_fraction=0.0 pour l'évaluation.")
                elif 'RADIUS_FRACTION' in config_params:
                    training_radius_fraction = config_params['RADIUS_FRACTION']
                    print(f"INFO: Fraction de rayon de l'entraînement ({training_radius_fraction}) trouvée dans la configuration effective du training.")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"AVERTISSEMENT: Erreur lors de la lecture de '{config_path}'. Utilisation de la valeur par défaut. Erreur: {e}")
        else:
            print(f"AVERTISSEMENT: Fichier 'effective_training_config.json' non trouvé. Utilisation de la valeur par défaut de config.py: {training_radius_fraction}")
        
        # Ajouter la valeur déterminée à l'objet args pour la passer aux fonctions suivantes
        args.training_radius_fraction = training_radius_fraction

        # Copie du fichier de configuration de l'entraînement
        config_src_path = exp_path / 'effective_training_config.json'
        if config_src_path.exists():
            shutil.copy(config_src_path, output_dir_path / f"config_{exp_name}.json")

        models_to_evaluate = get_models_for_evaluation(exp_path, args.evaluation_strategy)
        if not models_to_evaluate:
            print(f"AVERTISSEMENT: Aucun modèle trouvé pour l'expérience {exp_name}. Passe à la suivante.")
            continue

        intermediate_df = pd.read_csv(intermediate_results_path)

        for fold_id, model_path in sorted(models_to_evaluate.items()):
            print(f"\n--- Traitement du Fold '{fold_id}' ---")
            
            # Vérifier si déjà traité
            already_done = not intermediate_df[
                (intermediate_df['experiment'] == exp_name) & 
                (intermediate_df['fold_idx'] == (0 if fold_id == 'ensemble' else fold_id))
            ].empty
            if already_done and not args.force_rerun:
                print(f"  -> Résultat pour le fold {fold_id} déjà trouvé. Ignoré.")
                continue

            best_params = {}
            if is_grid_search_needed(args):
                # Pour l'ensemble, on utilise le premier tomo de validation pour la recherche d'HP
                val_tomo_id = all_tomo_ids[0] if fold_id == 'ensemble' else all_tomo_ids[int(fold_id)]
                best_params = run_hp_search_for_model(model_path, val_tomo_id, output_dir_path, args)
            else:
                print("INFO: Tous les HP sont fixés, la recherche par grille est sautée.")
                # Remplir best_params avec les valeurs fixes pour la cohérence
                best_params = {
                    'vol_frac': args.fixed_vol_frac_threshold,
                    'nms_frac': args.fixed_nms_radius_fraction,
                    'ws_cluster_rad_frac': args.fixed_ws_cluster_rad_frac,
                    'cluster_radius_frac': args.fixed_cluster_radius_fraction,
                    'min_cluster_vol_frac': args.fixed_min_cluster_vol_frac, # Non utilisé par watershed
                    'plm_gpu_peak_thresh': args.fixed_plm_gpu_peak_thresh
                }

            result_final = run_final_evaluation_for_model(
                model_path, best_params, output_dir_path, exp_name,
                (0 if fold_id == 'ensemble' else fold_id), per_tomo_results_path, args
            )

            # Sauvegarder le résultat intermédiaire
            new_row = {
                'experiment': exp_name,
                'fold_idx': (0 if fold_id == 'ensemble' else fold_id),
                'score': result_final['score'],
                'time_per_tomo': result_final['time_per_tomo'],
                'num_tomos': result_final['num_tomos'],
                'params_json': json.dumps(result_final.get('best_params', {}))
            }
            pd.DataFrame([new_row]).to_csv(intermediate_results_path, mode='a', header=False, index=False)
            print(f"  -> Résultat pour le fold {fold_id} sauvegardé.")
        
        # Nettoyage des dossiers temporaires
        if 'ensemble' in models_to_evaluate and models_to_evaluate['ensemble'].exists():
            shutil.rmtree(models_to_evaluate['ensemble'])
        if args.evaluation_strategy == 'ensemble_specialists':
            for fold_id in models_to_evaluate:
                if models_to_evaluate[fold_id].exists():
                    shutil.rmtree(models_to_evaluate[fold_id])

    # --- Agrégation finale ---
    aggregate_and_save_final_results(intermediate_results_path, output_dir_path, args.evaluation_strategy)

if __name__ == "__main__":
    main()