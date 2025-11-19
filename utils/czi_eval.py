import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path

try:
    from scipy.spatial import KDTree
except ImportError:
    print("AVERTISSEMENT: 'scipy' n'est pas installé. L'évaluation CZI ne fonctionnera pas. Installez avec 'pip install scipy'.")
    KDTree = None


def _czi_compute_metrics(
    ref_points: np.ndarray, match_threshold: float, cand_points: np.ndarray
) -> Tuple[int, int, int]:
    """
    Calcule les Vrais Positifs (TP), Faux Positifs (FP) et Faux Négatifs (FN) en utilisant
    la méthode query_ball_tree, fidèle à l'évaluation CZI originale.
    """
    if KDTree is None:
        raise ImportError("La classe 'KDTree' de scipy est requise pour l'évaluation CZI.")
    num_reference_particles = len(ref_points)
    num_candidate_particles = len(cand_points)

    if num_reference_particles == 0:
        # Aucun GT, tous les candidats sont des FP
        return 0, num_candidate_particles, 0
    if num_candidate_particles == 0:
        # Aucun candidat, tous les GT sont des FN
        return 0, 0, num_reference_particles

    ref_tree = KDTree(ref_points)
    cand_tree = KDTree(cand_points)

    # Pour chaque candidat, trouve tous les points de référence dans le rayon.
    raw_matches_cand_to_ref = cand_tree.query_ball_tree(ref_tree, r=match_threshold)

    # Un TP est un point de référence qui a au moins un candidat dans son rayon.
    matched_ref_indices = set()


    for matches in raw_matches_cand_to_ref:
        for ref_idx in matches:
            matched_ref_indices.add(ref_idx)
    tp = len(matched_ref_indices)

    # --- MODIFICATION POUR ALIGNEMENT AVEC LE CHALLENGE CZI ---
    # Le calcul des FP est aligné sur la logique officielle du challenge,
    # où fp = nombre_total_de_prédictions - nombre_de_vrais_positifs.
    # Note: Cette méthode peut compter des FP de manière incorrecte si plusieurs
    # prédictions correspondent à la même particule de vérité terrain, mais
    # elle est utilisée ici pour une stricte conformité avec le code du challenge.
    fp = num_candidate_particles - tp

    # Un FN est un point de référence qui n'a aucun candidat dans son rayon.
    fn = num_reference_particles - tp

    return int(tp), int(fp), int(fn)


def run_czi_evaluation(
    submission_df: pd.DataFrame,
    solution_df: pd.DataFrame,
    particle_config: Dict,
    beta: int = 4,
    exp_name: str = None,
    fold_idx: int = None,
    per_tomo_scores_csv_path: str = None
) -> Tuple[float, pd.DataFrame, Dict[str, float]]:
    """
    Exécute l'évaluation CZI.

    Retourne:
        - Le score F-beta global pondéré.
        - Un DataFrame avec les métriques détaillées par classe.
        - Un dictionnaire avec le score F-beta pour chaque tomogramme.
    """
    
    particle_radii = {p_name: p_info['radius'] for p_name, p_info in particle_config.items()}
    weights = {p_name: p_info['weight'] for p_name, p_info in particle_config.items()}

    # Le seuil de correspondance est la moitié du rayon de la particule (0.5 * rayon).
    match_thresholds = {k: v * 0.5 for k, v in particle_radii.items()}
    
    known_particle_types = set(weights.keys())
    
    # Pour le score global
    global_results: Dict[str, Dict[str, int]] = {p_type: {'tp': 0, 'fp': 0, 'fn': 0} for p_type in known_particle_types}
    
    can_save_incrementally = all([exp_name is not None, fold_idx is not None, per_tomo_scores_csv_path is not None])
    
    # Pour les scores par tomogramme
    per_tomo_scores: Dict[str, float] = {}

    # Itérer sur tous les tomogrammes présents dans la vérité terrain pour s'assurer
    # que les tomogrammes sans aucune prédiction sont également comptabilisés (score de 0).
    experiments_to_evaluate = set(solution_df['experiment'].unique())
    
    for experiment in sorted(list(experiments_to_evaluate)): # Trier pour un ordre déterministe
        # Métriques pour ce tomogramme spécifique
        tomo_results: Dict[str, Dict[str, int]] = {p_type: {'tp': 0, 'fp': 0, 'fn': 0} for p_type in known_particle_types}

        for p_type in known_particle_types:
            ref_points = solution_df.loc[(solution_df['experiment'] == experiment) & (solution_df['particle_type'] == p_type), ['x', 'y', 'z']].values
            cand_df_slice = submission_df.loc[(submission_df['experiment'] == experiment) & (submission_df['particle_type'] == p_type)]
            cand_points = cand_df_slice[['x', 'y', 'z']].values
            
            if p_type not in match_thresholds: continue

            tp, fp, fn = _czi_compute_metrics(ref_points, match_thresholds[p_type], cand_points)
            
            # Agrégation pour le score global
            global_results[p_type]['tp'] += tp
            global_results[p_type]['fp'] += fp
            global_results[p_type]['fn'] += fn
            
            # Stockage pour le score par tomogramme
            tomo_results[p_type]['tp'] = tp
            tomo_results[p_type]['fp'] = fp
            tomo_results[p_type]['fn'] = fn

        # Calculer le score pour ce tomogramme
        tomo_results_df = pd.DataFrame.from_dict(tomo_results, orient='index')
        tomo_results_df['precision'] = tomo_results_df['tp'] / (tomo_results_df['tp'] + tomo_results_df['fp'])
        tomo_results_df['recall'] = tomo_results_df['tp'] / (tomo_results_df['tp'] + tomo_results_df['fn'])
        tomo_results_df.fillna(0.0, inplace=True)
        
        beta_sq_tomo = beta ** 2
        fbeta_num_tomo = (1 + beta_sq_tomo) * tomo_results_df['precision'] * tomo_results_df['recall']
        fbeta_den_tomo = (beta_sq_tomo * tomo_results_df['precision']) + tomo_results_df['recall']
        tomo_results_df['f_beta'] = (fbeta_num_tomo / fbeta_den_tomo).fillna(0.0)
        tomo_results_df['weight'] = tomo_results_df.index.map(weights)
        total_weight_tomo = tomo_results_df['weight'].sum()
        tomo_score = (tomo_results_df['f_beta'] * tomo_results_df['weight']).sum() / total_weight_tomo if total_weight_tomo > 0 else 0.0
        per_tomo_scores[experiment] = tomo_score

        if can_save_incrementally:
            row = {
                'experiment': exp_name,
                'fold_idx': fold_idx,
                'tomo_id': experiment,
                'f4_score': tomo_score
            }
            pd.DataFrame([row]).to_csv(per_tomo_scores_csv_path, mode='a', header=False, index=False)

    # Calcul du score global
    results_df = pd.DataFrame.from_dict(global_results, orient='index')
    results_df['precision'] = results_df['tp'] / (results_df['tp'] + results_df['fp'])
    results_df['recall'] = results_df['tp'] / (results_df['tp'] + results_df['fn'])
    results_df.fillna(0.0, inplace=True)
    
    beta_sq = beta ** 2
    fbeta_num = (1 + beta_sq) * results_df['precision'] * results_df['recall']
    fbeta_den = (beta_sq * results_df['precision']) + results_df['recall']
    results_df['f_beta'] = (fbeta_num / fbeta_den).fillna(0.0)
    results_df['weight'] = results_df.index.map(weights)
    total_weight = results_df['weight'].sum()
    final_score = (results_df['f_beta'] * results_df['weight']).sum() / total_weight if total_weight > 0 else 0.0
    if can_save_incrementally and experiments_to_evaluate:
        print(f"  -> {len(experiments_to_evaluate)} scores par tomogramme sauvegardés de manière incrémentale dans {Path(per_tomo_scores_csv_path).name}")

    results_df_out = results_df.reset_index().rename(columns={'index': 'particle_type'})

    return final_score, results_df_out, per_tomo_scores