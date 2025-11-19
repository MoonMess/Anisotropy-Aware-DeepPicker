#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour générer les vérités terrains (ground truth) pour un entraînement
de segmentation sémantique.

Ce script lit les tomogrammes au format .zarr et les coordonnées des particules
au format .json depuis la structure de données CZI originale. Il produit deux
types de cartes pour chaque tomogramme : une carte de segmentation où chaque
voxel a une étiquette de classe.
"""

import argparse
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

try:
    import zarr
except ImportError:
    print("Erreur : La bibliothèque 'zarr' n'est pas installée.")
    print("Veuillez l'installer, par exemple via 'pip install zarr'")
    exit(1)
try:
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours
    PLOTTING_LIBS_AVAILABLE = True
except ImportError:
    PLOTTING_LIBS_AVAILABLE = False

import os
# Constante pour la conversion des coordonnées de picks (Angstroms) en indices de voxels.
from particle_config import CZI_PARTICLE_CONFIG
from config import VOXEL_SPACING, CZI_DATA_ROOT

def generate_ground_truth(tomo_shape, particles_data, particle_config, voxel_spacing, radius_fraction, single_pixel=False):
    """
    Génère la carte de segmentation sémantique de manière efficace.

    Args:
        tomo_shape (tuple): La forme du tomogramme (Z, Y, X).
        particles_data (list): Liste de dictionnaires, chacun décrivant une particule (coordonnées en Angstroms).
        particle_config (dict): Dictionnaire de configuration des particules (ID, rayon).
        voxel_spacing (float): La taille d'un voxel en Angstroms.
        radius_fraction (float): Fraction du rayon à utiliser (ignoré si single_pixel=True).
        single_pixel (bool): Si True, génère un masque d'un seul pixel au centroïde.

    Returns:
        np.ndarray: La carte de segmentation (seg_map).
    """
    seg_map = np.zeros(tomo_shape, dtype=np.uint8)

    for particle in tqdm(particles_data, leave=False, desc="  Particules"):
        class_name = particle['class_name']
        config = particle_config.get(class_name)
        if not config:
            continue

        # Propriétés de la particule
        class_id = config['id']

        # Conversion des coordonnées d'Angstroms en pixels
        center_x_px = particle['x_angstrom'] / voxel_spacing
        center_y_px = particle['y_angstrom'] / voxel_spacing
        center_z_px = particle['z_angstrom'] / voxel_spacing

        if single_pixel:
            # --- NOUVELLE LOGIQUE POUR LE MASQUE D'UN SEUL PIXEL ---
            # Arrondir aux coordonnées entières les plus proches
            iz = int(round(center_z_px))
            iy = int(round(center_y_px))
            ix = int(round(center_x_px))

            # Vérifier que les coordonnées sont dans les limites du tomogramme
            if 0 <= iz < tomo_shape[0] and 0 <= iy < tomo_shape[1] and 0 <= ix < tomo_shape[2]:
                seg_map[iz, iy, ix] = class_id
        else:
            # --- LOGIQUE ORIGINALE POUR LES SPHÈRES ---
            radius_px = (config['radius'] / voxel_spacing) * radius_fraction
            bbox_radius_px = radius_px

            # Définition d'une boîte englobante pour optimiser les calculs
            z_min = int(max(0, np.floor(center_z_px - bbox_radius_px)))
            z_max = int(min(tomo_shape[0], np.ceil(center_z_px + bbox_radius_px) + 1))
            y_min = int(max(0, np.floor(center_y_px - bbox_radius_px)))
            y_max = int(min(tomo_shape[1], np.ceil(center_y_px + bbox_radius_px) + 1))
            x_min = int(max(0, np.floor(center_x_px - bbox_radius_px)))
            x_max = int(min(tomo_shape[2], np.ceil(center_x_px + bbox_radius_px) + 1))

            if z_min >= z_max or y_min >= y_max or x_min >= x_max:
                continue

            # Grille de coordonnées pour la boîte englobante
            bb_z, bb_y, bb_x = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]

            # Calcul de la distance au carré par rapport au centre dans la boîte
            dist_sq_in_bb = (bb_x - center_x_px)**2 + (bb_y - center_y_px)**2 + (bb_z - center_z_px)**2

            # Mise à jour de la carte de segmentation
            sphere_mask_in_bb = dist_sq_in_bb <= radius_px**2
            seg_map_slice = seg_map[z_min:z_max, y_min:y_max, x_min:x_max]
            seg_map_slice[sphere_mask_in_bb] = class_id

    return seg_map

def _save_debug_plot(
    tomogram_slice: np.ndarray,
    seg_mask_slice: np.ndarray,
    particles_on_slice: list,    output_path: Path,
    particle_config: dict,
    run_id: str,
    tomo_type: str,
    z_slice_idx: int
):
    """
    Sauvegarde un graphique de débogage montrant les contours de segmentation
    et les annotations de vérité terrain.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    fig.suptitle(f"Visualisation pour {run_id} ({tomo_type}) - Coupe Z={z_slice_idx}", fontsize=16)

    # Normalisation du tomogramme pour l'affichage
    vmin, vmax = np.percentile(tomogram_slice, [1, 99])


    colors = ['red', 'cyan', 'lime', 'yellow', 'magenta', 'orange', 'dodgerblue']
    id_to_color = {p_info['id']: colors[i % len(colors)] for i, p_info in enumerate(particle_config.values())}
    id_to_name = {p_info['id']: p_name for p_name, p_info in particle_config.items()}

    # --- Axe 1: Masque de Segmentation ---
    ax.imshow(tomogram_slice, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title("Tomogramme + Contours de Segmentation + Annotations GT")

    labels_in_slice = np.unique(seg_mask_slice)
    legend_handles_seg, legend_labels_seg = [], []


    for label_id in sorted(labels_in_slice):
        if label_id == 0:
            continue
        current_label_mask = (seg_mask_slice == label_id)
        contours = find_contours(current_label_mask, level=0.5)
        color = id_to_color.get(label_id, 'white')
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1.2, color=color)

    for p in particles_on_slice:
        x_px = p['x_angstrom'] / VOXEL_SPACING
        y_px = p['y_angstrom'] / VOXEL_SPACING
        class_id = particle_config[p['class_name']]['id']
        color = id_to_color.get(class_id, 'white')
        ax.scatter(x_px, y_px, s=80, facecolors='none', edgecolors=color, linewidths=1.5, marker='o', label=p['class_name'])

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), title="Annotations GT (Cercle)")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close(fig)



def _parse_json_picks(file_path: str) -> list:
    """Lit un fichier de 'picks' JSON et retourne une liste de coordonnées."""
    coordinates = []
    if not Path(file_path).exists():
        return coordinates
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            for point in data.get('points', []):
                location = point.get('location')
                if location and 'x' in location and 'y' in location and 'z' in location:
                    coordinates.append(location)
    except Exception as e:
        print(f"Avertissement: Impossible de traiter le fichier de picks {file_path}: {e}")
    return coordinates

def main():
    parser = argparse.ArgumentParser(
        description="Génère les cartes de segmentation pour l'entraînement sur les données CZI."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=CZI_DATA_ROOT,
        help="Chemin vers le dossier racine du dataset CZI (contenant 'train', etc.)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Chemin vers le dossier de sortie où les sous-dossiers 'segmentations' et 'centroids' seront créés."
    )
    parser.add_argument(
        "--tomo_type",
        type=str,
        default="denoised",
        help="Type de tomogramme à traiter (ex: 'denoised', 'isonetcorrected')."
    )
    parser.add_argument(
        "--radius_fraction",
        type=float,
        default=0.5,
        help="Fraction du rayon configuré à utiliser pour les masques sphériques. Défaut: 1.0"
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Générer et sauvegarder une image de prévisualisation pour chaque masque."
    )

    parser.add_argument(
        "--single_pixel_mask",
        action="store_true",
        help="Générer un masque d'un seul pixel au centroïde au lieu d'une sphère."
    )
    args = parser.parse_args()

    data_root_path = Path(args.data_root)
    tomo_runs_dir = data_root_path / "train/static/ExperimentRuns"
    picks_base_dir = data_root_path / "train/overlay/ExperimentRuns"
    output_path = Path(args.output_dir) if args.output_dir else Path(os.path.join(data_root_path, "mask"))

    # Création des dossiers de sortie
    # Le nom du dossier reflète le mode utilisé
    if args.single_pixel_mask:
        # Utiliser radius_fraction 0.0 pour la compatibilité avec le dataloader
        seg_output_path = output_path / "segmentations_0.0"
    else:
        seg_output_path = output_path / f"segmentations_{args.radius_fraction}"
    seg_output_path.mkdir(parents=True, exist_ok=True)

    run_ids = sorted([d.name for d in tomo_runs_dir.iterdir() if d.is_dir() and d.name.startswith("TS_")])
    if not run_ids:
        print(f"Erreur : Aucun dossier de run ('TS_*') trouvé dans {tomo_runs_dir}")
        return
    
    # Désactiver la visualisation si les bibliothèques ne sont pas disponibles
    if args.save_plots and not PLOTTING_LIBS_AVAILABLE:
        print("\nAvertissement: Bibliothèques de visualisation (matplotlib, scikit-image) non trouvées. Visualisation désactivée.")
        args.save_plots = False

    for run_id in tqdm(run_ids, desc="Traitement des Tomogrammes"):
        tomo_path = tomo_runs_dir / run_id / "VoxelSpacing10.000" / f"{args.tomo_type}.zarr"
        if not tomo_path.exists():
            tqdm.write(f"Avertissement: Tomogramme non trouvé pour {run_id} à {tomo_path}. Ignoré.")
            continue

        try:
            zarr_obj = zarr.open(tomo_path, mode='r')
            tomo_shape = zarr_obj['0'].shape if isinstance(zarr_obj, zarr.hierarchy.Group) else zarr_obj.shape
        except Exception as e:
            tqdm.write(f"Erreur lors de la lecture de {tomo_path}: {e}")
            continue

        tqdm.write(f"\nTraitement de {run_id} (forme: {tomo_shape})")

        # Collecter toutes les particules pour ce tomogramme
        all_particles = []
        for p_name in CZI_PARTICLE_CONFIG:
            json_path = picks_base_dir / run_id / "Picks" / f"{p_name}.json"
            coords_list = _parse_json_picks(str(json_path))
            for loc in coords_list:
                all_particles.append({
                    "class_name": p_name,
                    "x_angstrom": loc['x'],
                    "y_angstrom": loc['y'],
                    "z_angstrom": loc['z']
                })

        if not all_particles:
            tqdm.write(f"  Aucune particule trouvée pour {run_id}. Passage au suivant.")
            continue

        # Génération et sauvegarde des cartes
        seg_map = generate_ground_truth(
            tomo_shape, all_particles, CZI_PARTICLE_CONFIG, VOXEL_SPACING, args.radius_fraction, single_pixel=args.single_pixel_mask
        )

        seg_map_file = seg_output_path / f"{run_id}.npy"
        np.save(seg_map_file, seg_map)

        tqdm.write(f"  Cartes sauvegardées pour {run_id}.")

        # Sauvegarde du plot si demandé
        if args.save_plots:
            try:
                # Charger les données du tomogramme pour la visualisation
                tomogram_data = zarr_obj['0'][:] if isinstance(zarr_obj, zarr.hierarchy.Group) else zarr_obj[:]
                
                # Choisir une coupe centrale
                z_slice_idx = tomo_shape[0] // 2
                tomo_slice = tomogram_data[z_slice_idx, :, :]
                seg_mask_slice = seg_map[z_slice_idx, :, :]

                # Filtrer les particules pour la visualisation
                particles_on_slice = []
                for p in all_particles:
                    config = CZI_PARTICLE_CONFIG.get(p['class_name'])
                    if not config: continue
                    radius_px = (config['radius'] / VOXEL_SPACING) * args.radius_fraction
                    z_px = p['z_angstrom'] / VOXEL_SPACING
                    if abs(z_px - z_slice_idx) <= radius_px:
                        particles_on_slice.append(p)                
                plot_output_path = output_path / f"{run_id}_{args.tomo_type}_mask_preview.png"
                _save_debug_plot(
                    tomo_slice, seg_mask_slice, particles_on_slice,
                    plot_output_path, CZI_PARTICLE_CONFIG, run_id, args.tomo_type,
                    z_slice_idx
                )
                tqdm.write(f"  Image de visualisation sauvegardée : {plot_output_path}")
            except Exception as e:
                tqdm.write(f"Erreur lors de la génération de la visualisation pour {run_id}: {e}")


if __name__ == "__main__":
    main()
