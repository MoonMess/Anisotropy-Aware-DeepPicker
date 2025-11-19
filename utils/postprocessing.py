import torch
import pandas as pd
import config
import numpy as np
from particle_config import CZI_PARTICLE_CONFIG
import time
import zarr
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

try:
    from scipy.spatial import KDTree
    from scipy import stats as sp_stats
except ImportError:
    print("AVERTISSEMENT: 'scipy' n'est pas installé. La NMS et le vote de classe ne fonctionneront pas. Installez avec 'pip install scipy'.")
    KDTree = None
    sp_stats = None

try:
    from scipy.ndimage import maximum
except ImportError:
    print("AVERTISSEMENT: 'scipy.ndimage.maximum' non trouvé. Le calcul du score pour cc3d ne fonctionnera pas.")
    maximum = None

import cc3d

try:
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max as skimage_peak_local_max
    from scipy.ndimage import distance_transform_edt
except ImportError:
    print("AVERTISSEMENT: 'scikit-image' ou 'scipy.ndimage' n'est pas installé. La méthode watershed ne fonctionnera pas.")
    watershed = None
    distance_transform_edt = None
    skimage_peak_local_max = None

try:
    from sklearn.cluster import MeanShift
except ImportError:
    print("AVERTISSESEMENT: 'scikit-learn' n'est pas installé. La méthode de clustering dforiginal ne fonctionnera pas. Installez avec 'pip install scikit-learn'.")
    MeanShift = None

def extract_particles_peak_local_max_gpu(
    seg_map_probs: torch.Tensor,
    particle_config: dict,
    per_class_hps: dict,
    voxel_spacing: float,
    downsampling_factor: int,
    internal_downsample_factor: int = 1,
    training_radius_fraction: float = config.RADIUS_FRACTION
) -> list:
    """
    Extracts particles using a GPU-based local maxima detection approach.
    This method is inspired by the provided snippet and is designed for speed.
    It includes a Non-Maximum Suppression (NMS) step on CPU after peak detection.
    """
    if KDTree is None:
        print("AVERTISSEMENT: Scipy non trouvé, la NMS pour 'peak_local_max_gpu' sera ignorée.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("AVERTISSEMENT: Aucun GPU détecté. La méthode 'peak_local_max_gpu' sera très lente sur CPU.")

    # --- Sous-échantillonnage interne ---
    if internal_downsample_factor > 1:
        downsampled_shape = tuple(s // internal_downsample_factor for s in seg_map_probs.shape[1:])
        seg_map_probs_gpu = F.interpolate(
            seg_map_probs.unsqueeze(0), size=downsampled_shape, mode='trilinear', align_corners=False
        ).squeeze(0).to(device)
    else:
        seg_map_probs_gpu = seg_map_probs.to(device)
    
    detected_particles = []

    for class_id in range(1, config.NUM_CLASSES + 1):
        class_name = None
        for name, props in CZI_PARTICLE_CONFIG.items():
            if props['id'] == class_id:
                class_name = name
                break
        if not class_name: continue

        prob_map_class = seg_map_probs_gpu[class_id]

        hps_for_class = per_class_hps.get(class_name)
        if not hps_for_class:
            continue
        
        # Get hyperparameters for this method
        kernel_size = 3  # Fixed kernel size
        mp_num = 5       # Fixed number of mean pooling iterations
        peak_thresh = hps_for_class.get('plm_gpu_peak_thresh', config.VALIDATION_CONF_THRESHOLD)
        nms_frac = hps_for_class.get('nms_frac', config.VALIDATION_NMS_RADIUS_FRACTION) # Reuse nms_frac from peak_local_max

        # 1. Binarize probability map using argmax
        pred_bin = (torch.argmax(seg_map_probs_gpu, dim=0) == class_id).float()

        # 2. Create pooling layers
        meanPool = nn.AvgPool3d(kernel_size, 1, kernel_size // 2).to(device)
        maxPool = nn.MaxPool3d(kernel_size, 1, kernel_size // 2).to(device)

        # 3. Create heatmap via repeated average pooling
        hmax = pred_bin.unsqueeze(0).unsqueeze(0)
        for _ in range(mp_num):
            hmax = meanPool(hmax)
        
        # 4. Detect peaks by comparing with max-pooled version
        pred_heatmap = hmax.clone()
        hmax_pooled = maxPool(pred_heatmap)
        
        # Détecter les maxima locaux (les points qui n'ont pas été modifiés par le max pooling)
        is_peak = (hmax_pooled == pred_heatmap)
        # Appliquer le seuil de confiance sur la heatmap
        above_threshold = (pred_heatmap > peak_thresh)
        # Un pic est conservé s'il est un maximum local ET au-dessus du seuil
        keep_mask = (is_peak & above_threshold).squeeze()
        
        coords_gpu = keep_mask.nonzero(as_tuple=False)

        if coords_gpu.shape[0] == 0:
            continue

        scores_gpu = prob_map_class[coords_gpu[:, 0], coords_gpu[:, 1], coords_gpu[:, 2]]

        coords_cpu = coords_gpu.cpu().numpy()
        scores_cpu = scores_gpu.cpu().numpy()

        # 5. Apply Non-Maximum Suppression (NMS) on CPU
        if KDTree is not None and nms_frac > 0:
            temp_df = pd.DataFrame({'z': coords_cpu[:, 0], 'y': coords_cpu[:, 1], 'x': coords_cpu[:, 2], 'score': scores_cpu}).sort_values('score', ascending=False).reset_index(drop=True)
            coords_for_nms = temp_df[['z', 'y', 'x']].values
            
            # Le rayon de base pour la NMS doit être cohérent avec le rayon utilisé pour générer les masques d'entraînement.
            radius_in_angstroms = particle_config[class_name]['radius']
            training_mask_radius_A = radius_in_angstroms * training_radius_fraction

            total_downsampling_factor = downsampling_factor * internal_downsample_factor
            effective_voxel_spacing = voxel_spacing * total_downsampling_factor
            # Le rayon de suppression est maintenant basé sur le rayon du masque d'entraînement.
            suppression_radius_px = (training_mask_radius_A / effective_voxel_spacing) * nms_frac
            suppression_radius_px = max(1.0, suppression_radius_px)

            tree = KDTree(coords_for_nms)
            keep_indices = []
            discarded_indices = set()
            
            for i in range(len(temp_df)):
                if i in discarded_indices: continue
                keep_indices.append(i)
                neighbors = tree.query_ball_point(coords_for_nms[i], r=suppression_radius_px)
                for neighbor_idx in neighbors:
                    if neighbor_idx != i: discarded_indices.add(neighbor_idx)

            final_df = temp_df.iloc[keep_indices]
            for _, row in final_df.iterrows():
                detected_particles.append({
                    'particle_type': class_name, 
                    'x': row['x'] * internal_downsample_factor, 
                    'y': row['y'] * internal_downsample_factor, 
                    'z': row['z'] * internal_downsample_factor, 
                    'score': row['score']
                })
        else: # No NMS
            for i in range(len(coords_cpu)):
                detected_particles.append({
                    'particle_type': class_name, 
                    'x': coords_cpu[i, 2] * internal_downsample_factor, 
                    'y': coords_cpu[i, 1] * internal_downsample_factor, 
                    'z': coords_cpu[i, 0] * internal_downsample_factor, 
                    'score': scores_cpu[i]
                })

    return detected_particles

def extract_particles_cc3d(
    seg_map_probs: torch.Tensor,
    particle_config: dict,
    per_class_hps: dict,
    voxel_spacing: float,
    downsampling_factor: int,
    internal_downsample_factor: int = 1,
    training_radius_fraction: float = config.RADIUS_FRACTION
) -> list:
    """
    Extrait les particules en utilisant une approche combinant argmax et composantes connexes (cc3d).
    Pour chaque classe, les régions où cette classe est la plus probable (résultat de l'argmax)
    sont identifiées. Seules les régions (composantes) dépassant un seuil de volume sont conservées,
    et leurs centroïdes sont extraits.
    Le score est la probabilité maximale de la classe au sein de chaque composante.
    
    Args:
        seg_map_probs (torch.Tensor): Tensor de probabilités de segmentation de forme (C, D, H, W).
        particle_config (dict): Dictionnaire de configuration des particules (contient les ID de classe).
        per_class_hps (dict): Dictionnaire d'HP contenant 'vol_frac' pour chaque classe.
        voxel_spacing (float): Taille du voxel en Angstroms. (Non utilisé directement ici)
        downsampling_factor (int): Facteur de sous-échantillonnage. (Non utilisé directement ici)

    Returns:
        list: Une liste de dictionnaires, chaque dictionnaire représentant une particule détectée.
    """
    if cc3d is None or maximum is None:
        print("AVERTISSEMENT: 'cc3d' ou 'scipy' n'est pas installé. L'extraction de particules par composantes connexes est ignorée.")
        return []

    # --- Sous-échantillonnage interne ---
    if internal_downsample_factor > 1:
        downsampled_shape = tuple(s // internal_downsample_factor for s in seg_map_probs.shape[1:])
        seg_map_probs = F.interpolate(
            seg_map_probs.unsqueeze(0), size=downsampled_shape, mode='trilinear', align_corners=False
        ).squeeze(0)

    detected_particles = []
    class_map_probs_np = seg_map_probs.cpu().numpy()
    
    # Appliquer argmax pour obtenir une carte de segmentation où chaque voxel a une seule classe.
    # Cela garantit qu'il n'y a pas de chevauchement entre les classes pour la détection.
    argmax_map = np.argmax(class_map_probs_np, axis=0)
    
    target_class_name = per_class_hps.get('target_class_for_hp_tuning')

    for class_id in range(1, config.NUM_CLASSES + 1):
        class_name = None
        for name, props in particle_config.items():
            if props['id'] == class_id:
                class_name = name
                break
        if not class_name: continue

        # Si une classe cible est spécifiée pour la recherche d'HP, on ignore toutes les autres.
        if target_class_name and class_name != target_class_name:
            continue

        # Vérifier si la classe est à traiter, même si le seuil n'est plus utilisé pour le masquage.
        hps_for_class = per_class_hps.get(class_name)
        if not hps_for_class:
            continue
        
        # Priorité pour le seuil de volume :
        # 1. 'vol_frac' dans les HPs fournis pour la classe (résultat de la recherche de grille).
        # 2. 'vol_frac_override' dans la configuration de la particule (pour les cas spéciaux comme beta-amylase).
        # 3. La valeur par défaut globale de config.py.
        vol_frac_thresh = hps_for_class.get(
            'vol_frac', 
            particle_config[class_name].get('vol_frac_override', config.CC3D_VOL_FRAC_THRESHOLD)
        )
        # Le seuil de confiance 'conf' n'est plus utilisé pour créer le masque binaire.
        # La segmentation est maintenant déterminée par l'argmax.
        # 1. Créer un masque binaire pour la classe actuelle à partir de la carte argmax.
        binary_mask = (argmax_map == class_id)

        if not np.any(binary_mask):
            continue

        # 2. Appliquer les composantes connexes
        labels_out, N = cc3d.connected_components(binary_mask, return_N=True, connectivity=config.CC3D_CONNECTIVITY)

        if N == 0:
            continue

        # 3. Extraire les statistiques (centroïdes) et les scores
        stats = cc3d.statistics(labels_out)
        centroids = stats['centroids'][1:] # Ignorer le fond (label 0)
        voxel_counts = stats['voxel_counts'][1:]

        # Calculer la probabilité maximale pour chaque composante connexe.
        # On utilise toujours la carte de probabilités originale pour obtenir un score de confiance.
        max_probs = maximum(class_map_probs_np[class_id], labels=labels_out, index=np.arange(1, N + 1))

        # --- 4. Appliquer le filtre de taille basé sur la fraction de volume ---
        # Le calcul du seuil de taille doit prendre en compte le sous-échantillonnage
        # pour que la comparaison soit cohérente.
        radius_in_angstroms = particle_config[class_name]['radius']
        
        # Calculer la taille effective du voxel dans la carte sous-échantillonnée.
        # C'est l'étape clé qui prend en compte le `downsampling_factor`.
        total_downsampling_factor = downsampling_factor * internal_downsample_factor
        effective_voxel_spacing = voxel_spacing * total_downsampling_factor
        # Convertir le rayon de la particule (en Angstroms) en un rayon en pixels
        # dans l'espace sous-échantillonné. On utilise RADIUS_FRACTION pour être
        # cohérent avec la taille des masques utilisés à l'entraînement.
        radius_px = (radius_in_angstroms * training_radius_fraction) / effective_voxel_spacing
        
        # Volume d'une sphère parfaite en pixels (dans l'espace sous-échantillonné).
        volume_sphere_px = (4/3) * np.pi * (radius_px ** 3)
        min_size_px = volume_sphere_px * vol_frac_thresh

        for i in range(N):
            # Appliquer le filtre de taille. `voxel_counts` est le nombre de voxels
            # dans la composante, mesuré sur la carte sous-échantillonnée.
            if voxel_counts[i] >= min_size_px:
                z, y, x = centroids[i]
                score = max_probs[i]
                detected_particles.append({
                    'particle_type': class_name, 
                    'x': x * internal_downsample_factor, 
                    'y': y * internal_downsample_factor, 
                    'z': z * internal_downsample_factor, 
                    'score': float(score),
                })

    return detected_particles

def extract_particles_watershed(
    seg_map_probs: torch.Tensor,
    particle_config: dict,
    per_class_hps: dict,
    voxel_spacing: float,
    downsampling_factor: int,
    internal_downsample_factor: int = 1,
    training_radius_fraction: float = config.RADIUS_FRACTION
) -> list:
    """
    Extrait les particules en utilisant une approche basée sur l'algorithme watershed pour séparer
    les instances qui se touchent.
    """
    if watershed is None or distance_transform_edt is None or skimage_peak_local_max is None or cc3d is None or maximum is None:
        print("AVERTISSEMENT: Dépendances manquantes pour watershed (skimage, scipy, cc3d). Méthode ignorée.")
        return []

    # --- Sous-échantillonnage interne ---
    if internal_downsample_factor > 1:
        downsampled_shape = tuple(s // internal_downsample_factor for s in seg_map_probs.shape[1:])
        seg_map_probs = F.interpolate(
            seg_map_probs.unsqueeze(0), size=downsampled_shape, mode='trilinear', align_corners=False
        ).squeeze(0)

    detected_particles = []
    class_map_probs_np = seg_map_probs.cpu().numpy()
    argmax_map = np.argmax(class_map_probs_np, axis=0)
    
    target_class_name = per_class_hps.get('target_class_for_hp_tuning')

    for class_id in range(1, config.NUM_CLASSES + 1):
        class_name = None
        for name, props in particle_config.items():
            if props['id'] == class_id:
                class_name = name
                break
        if not class_name: continue

        if target_class_name and class_name != target_class_name:
            continue

        hps_for_class = per_class_hps.get(class_name)
        if not hps_for_class:
            continue
        
        # Récupérer les HPs spécifiques à la méthode watershed
        vol_frac_thresh = hps_for_class.get(
            'vol_frac', 
            particle_config[class_name].get('vol_frac_override', config.CC3D_VOL_FRAC_THRESHOLD)
        )
        # NOUVEAU: Utiliser la clé spécifique 'ws_cluster_rad_frac' avec un fallback sur l'ancienne clé pour la compatibilité.
        cluster_radius_frac = hps_for_class.get('ws_cluster_rad_frac', hps_for_class.get('cluster_radius_frac', 1.0))


        binary_mask = (argmax_map == class_id)
        if not np.any(binary_mask):
            continue

        # 1. Calculer la transformée de distance
        distance = distance_transform_edt(binary_mask)

        # 2. Trouver les pics pour les marqueurs du watershed
        radius_in_angstroms = particle_config[class_name]['radius']
        total_downsampling_factor = downsampling_factor * internal_downsample_factor
        effective_voxel_spacing = voxel_spacing * total_downsampling_factor
        radius_px = radius_in_angstroms / effective_voxel_spacing # Rayon physique en pixels

        # NOUVELLE LOGIQUE: La distance minimale est basée sur le diamètre physique (2 * rayon).
        # L'HP `cluster_radius_frac` agit comme un multiplicateur sur ce diamètre.
        min_distance_pk = int(max(1, (2 * radius_px) * cluster_radius_frac))

        coords = skimage_peak_local_max(distance, min_distance=min_distance_pk, labels=binary_mask)
        
        markers_mask = np.zeros(distance.shape, dtype=bool)
        markers_mask[tuple(coords.T)] = True
        
        # 3. Étiqueter les marqueurs
        markers, num_features = cc3d.connected_components(markers_mask, return_N=True)
        if num_features == 0:
            continue

        # 4. Appliquer l'algorithme watershed
        labels_out = watershed(-distance, markers, mask=binary_mask)

        # 5. Extraire les statistiques et filtrer
        stats = cc3d.statistics(labels_out)
        centroids = stats['centroids'][1:] # Ignorer le fond
        voxel_counts = stats['voxel_counts'][1:]

        max_probs = maximum(class_map_probs_np[class_id], labels=labels_out, index=np.arange(1, num_features + 1))

        # --- Filtre de taille ---
        # Le radius_px ici doit aussi utiliser RADIUS_FRACTION pour la cohérence.
        radius_px_for_volume = (radius_in_angstroms * training_radius_fraction) / effective_voxel_spacing
        volume_sphere_px = (4/3) * np.pi * (radius_px_for_volume ** 3)
        min_size_px = volume_sphere_px * vol_frac_thresh

        for i in range(num_features):
            if voxel_counts[i] >= min_size_px:
                z, y, x = centroids[i]
                score = max_probs[i]
                detected_particles.append({
                    'particle_type': class_name, 
                    'x': x * internal_downsample_factor, 
                    'y': y * internal_downsample_factor, 
                    'z': z * internal_downsample_factor, 
                    'score': float(score),
                })

    return detected_particles

def extract_particles_meanshift(
    seg_map_probs: torch.Tensor,
    particle_config: dict,
    per_class_hps: dict,
    voxel_spacing: float,
    downsampling_factor: int,
    internal_downsample_factor: int = 1,
    training_radius_fraction: float = config.RADIUS_FRACTION
) -> list:
    """
    Extrait les particules en utilisant une approche de clustering MeanShift. Pour améliorer
    la performance, le clustering est effectué sur une carte de probabilités
    sous-échantillonnée d'un facteur 2 supplémentaire.

    La bande passante (bandwidth) est calculée dynamiquement :
    - En mode de recherche d'HP par classe, elle se base sur le rayon de la classe cible.
    - En mode d'évaluation finale ou de recherche globale, elle se base sur le rayon moyen de toutes les particules.
    """
    if MeanShift is None:
        print("AVERTISSEMENT: 'scikit-learn' n'est pas installé. L'extraction de particules par clustering MeanShift est ignorée.")
        return []
    if sp_stats is None:
        print("AVERTISSEMENT: 'scipy.stats' n'est pas installé. Le vote de classe pour le clustering est ignoré.")
        return []

    detected_particles = []
    
    # --- Sous-échantillonnage interne ---
    if internal_downsample_factor > 1:
        # Downsampler la carte de probabilités pour le calcul du score et de l'argmax
        downsampled_shape = tuple(s // internal_downsample_factor for s in seg_map_probs.shape[1:])
        map_for_clustering = F.interpolate(
            seg_map_probs.unsqueeze(0), size=downsampled_shape, mode='trilinear', align_corners=False
        ).squeeze(0)
    else:
        map_for_clustering = seg_map_probs

    # Obtenir la carte d'argmax et les coordonnées des voxels sur la carte (potentiellement) sous-échantillonnée
    argmax_map_ds = torch.argmax(map_for_clustering, dim=0)
    
    # 3. Déterminer les HP et le rayon de base pour le clustering
    target_class_for_hp_tuning = per_class_hps.get('target_class_for_hp_tuning')

    if target_class_for_hp_tuning:
        # Mode: Recherche d'HP pour une classe spécifique
        hps_for_run = per_class_hps.get(target_class_for_hp_tuning)
        if not hps_for_run:
            print(f"AVERTISSEMENT: HP non trouvés pour la classe cible '{target_class_for_hp_tuning}'. Clustering ignoré.")
            return []
        
        base_radius_A = particle_config[target_class_for_hp_tuning]['radius']
        print(f"  INFO (meanshift): HP search for '{target_class_for_hp_tuning}'. Using its radius ({base_radius_A} A) as base for bandwidth.")
        
        # Filtrer les voxels pour ne garder que ceux de la classe cible.
        target_class_id = particle_config[target_class_for_hp_tuning]['id']
        objvoxels_coords = torch.argwhere(argmax_map_ds == target_class_id).cpu().numpy()
        print(f"  INFO (meanshift): HP search mode. Clustering only voxels for class '{target_class_for_hp_tuning}' (ID: {target_class_id}).")

        if objvoxels_coords.shape[0] == 0:
            return []
        
        # En mode de recherche par classe, on utilise les HPs de la grille de recherche
        cluster_radius_frac = hps_for_run.get('cluster_radius_frac')
        min_cluster_vol_frac = hps_for_run.get('min_cluster_vol_frac', 0.05)

        if cluster_radius_frac is None:
            print("AVERTISSEMENT: 'cluster_radius_frac' non trouvé dans les HP. Clustering ignoré.")
            return []

        # --- Exécution du clustering pour la classe cible ---
        detected_particles.extend(
            _run_meanshift_on_voxels(
                objvoxels_coords, map_for_clustering, argmax_map_ds, base_radius_A, 
                cluster_radius_frac, min_cluster_vol_frac, particle_config, 
                voxel_spacing, downsampling_factor, internal_downsample_factor,
                training_radius_fraction
            )
        )
    else:
        # NOUVEAU: Mode global - Itérer sur chaque classe et exécuter MeanShift séparément.
        print("  INFO (meanshift): Mode global. Traitement de chaque classe de particule séparément.")
        for class_name, props in particle_config.items():
            class_id = props['id']
            if class_id == 0: continue

            # Isoler les voxels pour la classe actuelle
            objvoxels_coords = torch.argwhere(argmax_map_ds == class_id).cpu().numpy()
            if objvoxels_coords.shape[0] == 0:
                continue
            
            print(f"\n  --- Classe: {class_name} ({objvoxels_coords.shape[0]} voxels) ---")

            # NOUVEAU: Utiliser le rayon spécifique de la classe
            base_radius_A = props['radius']

            # Récupérer les HPs pour cette classe depuis la grille de recherche globale.
            hps_for_class = per_class_hps.get(class_name, {})
            cluster_radius_frac = hps_for_class.get('cluster_radius_frac', 1.0) # Utiliser la valeur de la grille
            min_cluster_vol_frac = hps_for_class.get('min_cluster_vol_frac', 0.05)
            
            # --- Exécution du clustering pour cette classe ---
            detected_particles.extend(
                _run_meanshift_on_voxels(
                    objvoxels_coords, map_for_clustering, argmax_map_ds, base_radius_A, 
                    cluster_radius_frac, min_cluster_vol_frac, particle_config, 
                voxel_spacing, downsampling_factor, internal_downsample_factor,
                training_radius_fraction
                )
            )
        
    return detected_particles

def _run_meanshift_on_voxels(
    objvoxels_coords: np.ndarray,
    map_for_clustering: torch.Tensor,
    argmax_map_ds: torch.Tensor,
    base_radius_A: float,
    cluster_radius_frac: float,
    min_cluster_vol_frac: float,
    particle_config: dict,
    voxel_spacing: float,
    downsampling_factor: int,
    internal_downsample_factor: int,
    training_radius_fraction: float
) -> list:
    """Fonction helper pour exécuter le clustering MeanShift sur un ensemble de voxels."""
    detected_particles = []
    # Convertir les tenseurs en numpy pour le reste du traitement
    class_map_probs_np_ds = map_for_clustering.cpu().numpy()
    argmax_map_np_ds = argmax_map_ds.cpu().numpy()
    
    # 4. Exécuter MeanShift
    # Le `downsampling_factor` est celui du pipeline global. On ajoute notre facteur local.
    total_downsampling_factor = downsampling_factor * internal_downsample_factor
    effective_voxel_spacing_ds = voxel_spacing * total_downsampling_factor
    
    # Aligner le rayon de base avec celui utilisé pour l'entraînement (comme pour watershed) en utilisant RADIUS_FRACTION.
    bandwidth_px = (base_radius_A * training_radius_fraction / effective_voxel_spacing_ds) * cluster_radius_frac
    bandwidth_px = max(1.0, bandwidth_px)  # Doit être au moins 1.

    print(f"  INFO (meanshift): Lancement du clustering MeanShift sur une carte sous-échantillonnée ({objvoxels_coords.shape[0]} points) avec une bande passante de {bandwidth_px:.2f} pixels...")
    start_time = time.time()
    ms = MeanShift(bandwidth=bandwidth_px, bin_seeding=True, n_jobs=-1)
    ms.fit(objvoxels_coords)
    end_time = time.time()
    print(f"  INFO (meanshift): Clustering terminé en {end_time - start_time:.2f}s. {len(ms.cluster_centers_)} clusters trouvés.")
    
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    unique_labels = np.unique(labels)
    
    id_to_class_name = {props['id']: name for name, props in particle_config.items()}

    for k in unique_labels:
        if k == -1: continue  # Points de bruit

        my_members_mask = (labels == k)
        member_coords = objvoxels_coords[my_members_mask]

        # --- Vote de classe ---
        member_class_ids = argmax_map_np_ds[member_coords[:, 0], member_coords[:, 1], member_coords[:, 2]]
        if len(member_class_ids) == 0: continue
        
        winning_class_id = int(sp_stats.mode(member_class_ids, keepdims=False)[0])
        if winning_class_id == 0: continue
            
        winning_class_name = id_to_class_name.get(winning_class_id)
        if not winning_class_name: continue

        # --- NOUVEAU: Filtrage par fraction de volume ---
        radius_in_angstroms = particle_config[winning_class_name]['radius']
        # effective_voxel_spacing_ds est déjà calculé
        # Rendre ce calcul cohérent avec celui de la bande passante
        radius_px = (radius_in_angstroms * training_radius_fraction) / effective_voxel_spacing_ds
        volume_sphere_px = (4/3) * np.pi * (radius_px ** 3)
        min_size_px = volume_sphere_px * min_cluster_vol_frac

        if len(member_coords) < min_size_px:
            continue
        # --- FIN NOUVEAU ---

        # --- Score ---
        # Utiliser le 90ème percentile pour un score plus robuste que le max.
        member_probs = class_map_probs_np_ds[winning_class_id, member_coords[:, 0], member_coords[:, 1], member_coords[:, 2]]
        score = np.percentile(member_probs, 90)
        
        # Les centroïdes sont dans l'espace sous-échantillonné. Il faut les remonter.
        cluster_center_z_ds, cluster_center_y_ds, cluster_center_x_ds = cluster_centers[k]
        detected_particles.append({
            'particle_type': winning_class_name,
            'x': cluster_center_x_ds * internal_downsample_factor,
            'y': cluster_center_y_ds * internal_downsample_factor,
            'z': cluster_center_z_ds * internal_downsample_factor,
            'score': float(score),
        })
        
    return detected_particles    

def process_predictions_to_df(
    prediction_map: torch.Tensor,
    tomo_id: str,
    particle_config: dict,
    voxel_spacing: float,
    downsampling_factor: int = config.POSTPROC_DOWNSAMPLING_FACTOR,
    conf_threshold: float = None,
    nms_radius_fraction: float = None,
    vol_fraction_threshold: float = None,
    ws_cluster_rad_frac: float = None,
    cluster_radius_fraction: float = None,
    min_cluster_vol_frac: float = None,    
    plm_gpu_peak_thresh: float = None,
    per_class_hps: dict = None,
    postprocessing_method: str = config.DEFAULT_POSTPROC_METHOD,
    internal_downsample_factor: int = 1,
    training_radius_fraction: float = config.RADIUS_FRACTION
) -> pd.DataFrame:
    """
    Pipeline de post-traitement unifié qui convertit une carte de probabilités en un DataFrame de détections.
    
    Args:
        prediction_map (torch.Tensor): Carte de probabilités (C, D, H, W) sur CPU.
        tomo_id (str): ID du tomogramme.
        particle_config (dict): Configuration des particules (rayons, etc.).
        voxel_spacing (float): Taille du voxel en Angstroms.
        downsampling_factor (int): Facteur de sous-échantillonnage global pour le lissage, appliqué avant la détection.
        conf_threshold (float, optional): Seuil de confiance global (utilisé par certaines méthodes si per_class_hps n'est pas fourni).
        nms_radius_fraction (float, optional): Fraction NMS globale (utilisé par 'peak_local_max_gpu').
        vol_fraction_threshold (float, optional): Seuil de fraction de volume global (utilisé par 'cc3d', 'watershed').
        cluster_radius_fraction (float, optional): Fraction de rayon de clustering globale (utilisé par 'meanshift').
        min_cluster_vol_frac (float, optional): Fraction de volume de cluster minimale globale (utilisé par 'meanshift').
        plm_gpu_peak_thresh (float, optional): Seuil de pic global (utilisé par 'peak_local_max_gpu').
        per_class_hps (dict, optional): Dictionnaire d'HP par classe. Prend le pas sur les paramètres globaux.
        postprocessing_method (str): Méthode de détection à utiliser ('cc3d', 'meanshift', 'peak_local_max_gpu', 'watershed').
        internal_downsample_factor (int): Facteur de sous-échantillonnage supplémentaire appliqué à l'intérieur de la
            méthode de détection, principalement pour le benchmarking de performance.
        training_radius_fraction (float): Fraction de rayon utilisée lors de l'entraînement, pour assurer la cohérence
            des calculs de taille/volume.

    Returns:
        pd.DataFrame: DataFrame des particules détectées avec les colonnes standard.
    """
    # 1. Lissage par sous-échantillonnage
    original_shape = prediction_map.shape[1:]
    downsampled_shape = (
        original_shape[0] // downsampling_factor,
        original_shape[1] // downsampling_factor,
        original_shape[2] // downsampling_factor,
    )
    
    map_downsampled = F.interpolate(
        prediction_map.unsqueeze(0), size=downsampled_shape, mode='trilinear', align_corners=False
    ).squeeze(0)

    # Déterminer les HP à utiliser. Priorité à per_class_hps.
    hps_to_use = per_class_hps
    if hps_to_use is None:
        # Construire un dictionnaire par défaut si des valeurs uniques sont fournies
        default_vol_frac = vol_fraction_threshold if vol_fraction_threshold is not None else config.CC3D_VOL_FRAC_THRESHOLD
        # Pour cc3d (simplifié), nms_radius_fraction peut être None. On met une valeur par défaut qui ne sera pas utilisée.
        default_nms_frac = nms_radius_fraction if nms_radius_fraction is not None else 0.0
        default_cluster_radius_frac = cluster_radius_fraction if cluster_radius_fraction is not None else 1.0
        default_min_cluster_vol_frac = min_cluster_vol_frac if min_cluster_vol_frac is not None else 0.05
        # HPs for peak_local_max_gpu
        default_plm_gpu_peak_thresh = plm_gpu_peak_thresh if plm_gpu_peak_thresh is not None else 0.1
        # Gérer le paramètre spécifique à watershed, avec fallback sur le générique pour la compatibilité
        default_ws_cluster_rad_frac = ws_cluster_rad_frac if ws_cluster_rad_frac is not None else default_cluster_radius_frac

        hps_to_use = {
            class_name: {
                'conf': conf_threshold, 
                'nms_frac': default_nms_frac, 
                'vol_frac': default_vol_frac,
                'ws_cluster_rad_frac': default_ws_cluster_rad_frac,
                'cluster_radius_frac': default_cluster_radius_frac,
                'min_cluster_vol_frac': default_min_cluster_vol_frac,
                'plm_gpu_peak_thresh': default_plm_gpu_peak_thresh,
            }
            for class_name in particle_config.keys()
        }

    # NOUVEAU: Utiliser un dispatcher pour appeler la bonne méthode
    method_dispatcher = {
        'cc3d': extract_particles_cc3d,
        'watershed': extract_particles_watershed,
        'meanshift': extract_particles_meanshift,
        'peak_local_max_gpu': extract_particles_peak_local_max_gpu,
    }

    extraction_func = method_dispatcher.get(postprocessing_method)
    if not extraction_func:
        raise ValueError(f"Méthode de post-traitement non reconnue: '{postprocessing_method}'")

    # 2. Détection de particules (appel dynamique)
    detected_particles_px = extraction_func(
        map_downsampled,
        particle_config=particle_config,
        per_class_hps=hps_to_use,
        voxel_spacing=voxel_spacing,
        downsampling_factor=downsampling_factor,
        internal_downsample_factor=internal_downsample_factor,
        training_radius_fraction=training_radius_fraction,
    )

    # 3. Formater la sortie (logique commune)
    if not detected_particles_px:
        return pd.DataFrame(columns=['experiment', 'particle_type', 'x', 'y', 'z', 'score'])
    
    df = pd.DataFrame(detected_particles_px)
    df['experiment'] = tomo_id
    for axis in ['x', 'y', 'z']:
        df[axis] = df[axis] * downsampling_factor * voxel_spacing
    return df[['experiment', 'particle_type', 'x', 'y', 'z', 'score']]

    # L'ancien bloc 'peak_local_max_gpu' est maintenant géré par le dispatcher
    # elif postprocessing_method == 'peak_local_max_gpu':
    #     detected_particles_px = extract_particles_peak_local_max_gpu(
    #         ...
    #     )
    #     ...
    # else:
    #     raise ValueError(...)


def _nms_single_tomo(df: pd.DataFrame, particle_config: dict, per_class_hps: dict, training_radius_fraction: float) -> pd.DataFrame:
    """
    Applique la NMS sur un DataFrame correspondant à un seul tomogramme.
    Les coordonnées sont supposées être en Angstroms.
    """
    if df.empty:
        return df

    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    
    coords = df[['x', 'y', 'z']].values
    tree = KDTree(coords)
    
    keep_indices = set(range(len(df)))
    
    for i in range(len(df)):
        if i not in keep_indices:
            continue
        
        p_type = df.loc[i, 'particle_type']
        # Récupérer nms_frac avec une valeur par défaut pour la robustesse
        nms_frac = per_class_hps.get(p_type, {}).get('nms_frac', 0.8)
        # Utiliser le rayon du masque d'entraînement pour la cohérence avec les autres méthodes
        physical_radius = particle_config[p_type]['radius']
        suppression_radius = (physical_radius * training_radius_fraction) * nms_frac
        
        neighbors = tree.query_ball_point(coords[i], r=suppression_radius)
        
        neighbors.remove(i)
        keep_indices.difference_update(neighbors)

    return df.iloc[list(keep_indices)].copy()




def nms_on_coords(df: pd.DataFrame, particle_config: dict, per_class_hps: dict, training_radius_fraction: float = config.RADIUS_FRACTION) -> pd.DataFrame:
    """
    Applique une Non-Maximum Suppression basée sur la distance euclidienne, par tomogramme.
    Args:
        df (pd.DataFrame): DataFrame avec les colonnes ['experiment', 'particle_type', 'x', 'y', 'z', 'score'].
        particle_config (dict): Dictionnaire de configuration des particules (ex: CZI_PARTICLE_CONFIG).
        per_class_hps (dict): Dictionnaire d'HP par classe, contenant 'nms_frac'.
    Returns:
        pd.DataFrame: DataFrame filtré après NMS.
    """
    if df.empty:
        return df
    if KDTree is None:
        print("AVERTISSEMENT: Scipy non trouvé, NMS sur les coordonnées est ignorée.")
        return df.drop_duplicates(subset=['experiment', 'particle_type', 'x', 'y', 'z'])

    if 'experiment' not in df.columns:
        print("AVERTISSEMENT: Colonne 'experiment' non trouvée dans le DataFrame pour la NMS. Traitement de toutes les détections comme un seul tomogramme.")
        return _nms_single_tomo(df, particle_config, per_class_hps, training_radius_fraction)

    # Appliquer la NMS pour chaque tomogramme séparément
    all_kept_dfs = [
        _nms_single_tomo(group, particle_config, per_class_hps, training_radius_fraction)
        for _, group in df.groupby('experiment', sort=False)
    ]
    
    return pd.concat(all_kept_dfs, ignore_index=True) if all_kept_dfs else pd.DataFrame(columns=df.columns)