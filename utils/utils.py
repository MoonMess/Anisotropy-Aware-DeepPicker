from datetime import datetime
import os
import numpy as np
import re
from pathlib import Path
from typing import List, Tuple, Optional
import config

def get_outputbasename(args):
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.encoder_name}_{args.mode}_{date}"
    if args.comment:
        output_dir += f"_{args.comment}"

    return output_dir

def normalize_globally_percentile(tomo_data: np.ndarray, lower_p: float = 5.0, upper_p: float = 99.0) -> np.ndarray:
    """
    Applique une normalisation globale par percentiles à un tomogramme.
    Les valeurs sont clippées puis mises à l'échelle dans l'intervalle [0, 1].
    """
    p_lower = np.percentile(tomo_data, lower_p)
    p_upper = np.percentile(tomo_data, upper_p)
    
    # Éviter la division par zéro si les percentiles sont identiques (image plate)
    if p_upper - p_lower > 1e-8:
        normalized_tomo = np.clip(tomo_data, p_lower, p_upper)
        normalized_tomo = (normalized_tomo - p_lower) / (p_upper - p_lower)
    else:
        # Retourner une copie en float32 pour la cohérence de type
        normalized_tomo = tomo_data.astype(np.float32)
        
    return normalized_tomo

def find_best_generalist_checkpoint(checkpoint_paths: List[Path]) -> Optional[Tuple[Path, str, float]]:
    """
    Trouve le meilleur checkpoint "généraliste" à partir d'une liste de chemins.
    Priorise les checkpoints basés sur le F-score (plus haut = mieux), et se rabat sur la perte (plus bas = mieux).

    Args:
        checkpoint_paths (List[Path]): Une liste d'objets Path pour les checkpoints.

    Returns:
        Optional[Tuple[Path, str, float]]: Un tuple contenant le chemin vers le meilleur checkpoint,
                                           le type de métrique ('f-score' ou 'loss'), et la valeur de la métrique.
                                           Retourne None si aucun checkpoint valide n'est trouvé.
    """
    if not checkpoint_paths:
        return None

    # Filtrer pour ne garder que les checkpoints généralistes
    specialist_fragments = [f'-best-{cname}-' for cname in config.CLASS_NAMES]
    generalist_ckpts = [p for p in checkpoint_paths if not any(frag in p.name for frag in specialist_fragments)]

    if not generalist_ckpts:
        return None

    # --- Priorité 1: Maximiser le F-score ---
    best_score = -1.0
    best_fscore_checkpoint = None
    fscore_pattern = re.compile(r"f\d+=(\d+(?:\.\d+)?)")
    
    for path in generalist_ckpts:
        if 'best-fscore' in path.name:
            match = fscore_pattern.search(path.name)
            if match:
                try:
                    score = float(match.group(1))
                    if score > best_score:
                        best_score = score
                        best_fscore_checkpoint = path
                except ValueError:
                    continue # Ignorer si le parsing échoue

    if best_fscore_checkpoint:
        return best_fscore_checkpoint, 'f-score', best_score

    # --- Priorité 2: Minimiser la perte (fallback) ---
    print("AVERTISSEMENT: Aucun checkpoint 'best-fscore' trouvé. Recherche du meilleur checkpoint basé sur la perte.")
    best_loss = float('inf')
    best_loss_checkpoint = None
    loss_pattern = re.compile(r"(?:val_total_loss|val_loss|loss)=(\d+(?:\.\d+)?)")

    for path in generalist_ckpts:
        if 'best-loss' in path.name:
            match = loss_pattern.search(path.name)
            if match:
                try:
                    loss = float(match.group(1))
                    if loss < best_loss:
                        best_loss = loss
                        best_loss_checkpoint = path
                except ValueError:
                    continue

    if best_loss_checkpoint:
        return best_loss_checkpoint, 'loss', best_loss

    return None

def generate_sliding_window_coords(tomo_shape: Tuple[int, int, int], patch_size: Tuple[int, int, int], overlap_fraction: float) -> List[Tuple[int, int, int]]:
    """
    Génère une liste de coordonnées (z, y, x) pour une inférence par fenêtre glissante.

    Args:
        tomo_shape (tuple): La forme du tomogramme (D, H, W).
        patch_size (tuple): La taille des patchs (p_d, p_h, p_w).
        overlap_fraction (float): La fraction de chevauchement entre les patchs (ex: 0.5 pour 50%).

    Returns:
        list: Une liste de tuples de coordonnées (z, y, x) pour le coin supérieur gauche de chaque patch.
    """
    d, h, w = tomo_shape
    p_d, p_h, p_w = patch_size
    
    stride_d = int(p_d * (1 - overlap_fraction))
    stride_h = int(p_h * (1 - overlap_fraction))
    stride_w = int(p_w * (1 - overlap_fraction))

    # S'assurer qu'on avance d'au moins 1 pixel pour éviter les boucles infinies
    stride_d = max(1, stride_d)
    stride_h = max(1, stride_h)
    stride_w = max(1, stride_w)

    z_coords = list(range(0, d - p_d, stride_d)) + [d - p_d] if d > p_d else [0]
    y_coords = list(range(0, h - p_h, stride_h)) + [h - p_h] if h > p_h else [0]
    x_coords = list(range(0, w - p_w, stride_w)) + [w - p_w] if w > p_w else [0]
    
    coords_set = set()
    for z in z_coords:
        for y in y_coords:
            for x in x_coords:
                coords_set.add((z, y, x))
    
    return sorted(list(coords_set))