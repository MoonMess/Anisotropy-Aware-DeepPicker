import torch
import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import json
import shutil
from particle_config import CZI_PARTICLE_CONFIG
import random
import pytorch_lightning as pl
import config
from utils.utils import normalize_globally_percentile, generate_sliding_window_coords

try:
    import zarr
except ImportError:
    print("Erreur: La bibliothèque 'zarr' n'est pas installée. Veuillez l'installer avec 'pip install zarr'.")
    exit(1)

from monai.transforms import (
    Compose,
    RandRotate90d,
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
    RandScaleIntensityd,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandAdjustContrastd,
    Rand3DElastic
)

class RandMeanStdShiftd:
    """
    Transformation MONAI pour appliquer une variation aléatoire de la moyenne et de l'écart-type.
    L'amplitude de la variation est également choisie aléatoirement jusqu'à une limite maximale.
    """
    def __init__(self, keys, prob=0.5, max_shift=0.5):
        self.keys = keys
        self.prob = prob
        self.max_shift = max_shift

    def __call__(self, data):
        d = dict(data)
        if torch.rand(1).item() < self.prob:
            for key in self.keys:
                image = d[key]
                std, mean = torch.std_mean(image)
                # Choisir une amplitude de variation aléatoire pour cette transformation,
                # jusqu'à la limite max_shift.
                current_shift = torch.rand(1).item() * self.max_shift

                # Si le shift est quasi nul, on ne fait rien pour éviter les divisions par zéro.
                if current_shift < 1e-6:
                    continue

                # Calculer la variation dans l'intervalle [-current_shift, +current_shift]
                factor = 1.0 / (current_shift * 2.0)
                shift_mean_factor = (torch.rand(1).item() / factor) - current_shift
                shift_std_factor = (torch.rand(1).item() / factor) - current_shift

                # L'ancienne implémentation (new_mean = mean + mean * shift_mean_factor) était incorrecte.
                # Si la moyenne du patch est proche de zéro (ce qui est le cas après la normalisation Z-score),
                # le décalage de la moyenne était quasi nul. Le décalage doit être additif.
                new_mean = mean + shift_mean_factor
                new_std = std + std * shift_std_factor

                # Appliquer la transformation
                d[key] = (image - mean) / (std) * new_std + new_mean
        return d


def collate_fn(batch):
    tomos = []
    seg_targets = []
    run_ids = []
    patch_offsets = []
    for sample in batch:
        tomos.append(sample['tomo'])
        seg_targets.append(sample['seg_target'])
        run_ids.append(sample['run_id'])
        patch_offsets.append(sample['patch_offset'])
    
    tomos = torch.stack(tomos, 0)
    seg_targets = torch.stack(seg_targets, 0)
    patch_offsets = torch.stack(patch_offsets, 0)
    return {'tomo': tomos, 'seg_target': seg_targets, 'run_id': run_ids, 'patch_offset': patch_offsets}


class Dataset(Dataset):
    def __init__(self, file_list, patch_size, mode='train', steps_per_epoch=None, augmentation_level='advanced', augmentation_mode='anisotropic', czi_data_root=None, validation_overlap_fraction=0.5, use_global_norm=False, **kwargs):
        """
        Args:
            file_list (list): Liste de dictionnaires, chacun contenant 'tomo_path', 'seg_path'.
            patch_size (list): Taille des patchs à extraire [D, H, W].
            mode (str): 'train' pour l'échantillonnage aléatoire, 'val' pour la fenêtre glissante.
            steps_per_epoch (int, optional): Nombre de patchs à générer par époque pour l'entraînement.
            augmentation_level (str): 'none', 'base' ou 'advanced' pour contrôler le niveau d'augmentation.
            augmentation_mode (str): 'anisotropic' ou 'isotropic' pour le comportement des augmentations.
            czi_data_root (str, optional): Chemin racine des données CZI, requis pour le mode 'train'.
        """
        self.patch_size = patch_size
        self.mode = mode
        self.steps_per_epoch = steps_per_epoch
        self.augmentation_level = augmentation_level
        self.augmentation_mode = augmentation_mode
        self.val_overlap_fraction = validation_overlap_fraction
        self.file_list = file_list
        self.use_global_norm = use_global_norm
        if not self.file_list:
            raise ValueError("La liste de fichiers fournie au Dataset est vide.")
        
        # Transformations de pré-traitement (normalisation, ajout de canal)
        self.pre_transforms = Compose(
            [
                EnsureChannelFirstd(keys=["tomo", "seg_target"], channel_dim="no_channel"),
                NormalizeIntensityd(keys="tomo", nonzero=False, channel_wise=False), # Applique Z-score sur tout le patch
            ]
        )
        
        # Transformations d'augmentation aléatoires pour l'entraînement
        self.random_transforms = None
        if self.mode == 'train' and self.augmentation_level != 'none':
            if self.augmentation_mode == 'isotropic':
                print("INFO: Utilisation des augmentations ISOTROPES (tous les axes sont traités de la même manière).")
                base_transforms = [

                ]
                advanced_transforms = [
                    RandAdjustContrastd(keys=["tomo"], gamma=(0.7, 1.5), prob=0.3),
                    RandMeanStdShiftd(keys=["tomo"], prob=config.AUG_MEAN_STD_SHIFT_PROB, max_shift=config.AUG_MEAN_STD_SHIFT_MAX_AMOUNT)
                ]
            else: # 'anisotropic' (défaut)
                print("INFO: Utilisation des augmentations ANISOTROPES (l'axe Z est traité différemment).")
                base_transforms = [
                    RandRotate90d(keys=["tomo", "seg_target"], prob=0.5, spatial_axes=[1, 2]), # Rotation uniquement autour de Z
                    RandFlipd(keys=["tomo", "seg_target"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["tomo", "seg_target"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["tomo", "seg_target"], prob=0.5, spatial_axis=2),
                    RandAffined(
                        keys=["tomo", "seg_target"],
                        prob=0.3,
                        rotate_range=(0.0, 0.0, 0.1),  # Rotation uniquement sur le plan XY (autour de Z)
                        scale_range=(0.05, 0.05, 0.05),   # Mise à l'échelle isotrope
                        mode=('bilinear', 'nearest'),
                    ),
                ]
                advanced_transforms = [
                    RandAdjustContrastd(keys=["tomo"], gamma=(0.7, 1.5), prob=0.3),
                    RandMeanStdShiftd(keys=["tomo"], prob=config.AUG_MEAN_STD_SHIFT_PROB, max_shift=config.AUG_MEAN_STD_SHIFT_MAX_AMOUNT)
                ]

            if self.augmentation_level == 'base':
                print("INFO: Niveau d'augmentation 'base' sélectionné (rotations, flips).")
                self.random_transforms = Compose(base_transforms)
            elif self.augmentation_level == 'advanced':
                print("INFO: Niveau d'augmentation 'advanced' sélectionné (base + affine, intensité).")
                self.random_transforms = Compose(base_transforms + advanced_transforms)
            else:
                 print("INFO: Niveau d'augmentation non reconnu ou 'none'. Seules les transformations de pré-traitement seront appliquées.")

        
        if self.mode == 'val':
            self._prepare_validation_patches()
        elif self.mode == 'train':
            if czi_data_root is None:
                raise ValueError("czi_data_root doit être fourni pour l'échantillonnage en mode entraînement.")
            self._prepare_training_particles(czi_data_root)
    def _prepare_training_particles(self, czi_data_root):
        """Pré-charge les coordonnées de toutes les particules pour un échantillonnage équilibré."""
        print("INFO: Préparation des particules pour l'échantillonnage d'entraînement équilibré...")
        self.particles_by_class = {p_name: [] for p_name in CZI_PARTICLE_CONFIG}
        
        picks_base_dir = Path(czi_data_root) / "train/overlay/ExperimentRuns"

        for tomo_idx, file_info in enumerate(self.file_list):
            run_id = file_info['run_id']
            picks_dir = picks_base_dir / run_id / "Picks"
            if not picks_dir.is_dir():
                continue

            for p_name in self.particles_by_class.keys():
                json_path = picks_dir / f"{p_name}.json"
                if not json_path.exists(): continue
                
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    for point in data.get('points', []):
                        loc = point.get('location')
                        if loc:
                            self.particles_by_class[p_name].append({
                                "tomo_idx": tomo_idx,
                                "center_coords_px": (loc['z'] / config.VOXEL_SPACING, loc['y'] / config.VOXEL_SPACING, loc['x'] / config.VOXEL_SPACING)
                            })
        
        self.available_classes = [p_name for p_name, p_list in self.particles_by_class.items() if p_list]
        if not self.available_classes:
            raise ValueError("Aucune particule trouvée pour l'entraînement. Vérifiez les chemins et les fichiers JSON.")
        
        total_particles = sum(len(p) for p in self.particles_by_class.values())
        print(f"INFO: {total_particles} particules trouvées dans {len(self.available_classes)} classes pour l'entraînement.")

    def _prepare_validation_patches(self):
        """Pré-calcule les coordonnées de tous les patchs pour la validation par fenêtre glissante."""
        self.patch_infos = []

        for tomo_idx, file_info in enumerate(self.file_list):
            try:
                val_tomo_path = file_info["tomo_paths"].get('denoised')
                if not val_tomo_path:
                    print(f"AVERTISSEMENT: Tomogramme 'denoised' non trouvé pour la validation de {file_info['run_id']}. Ignoré.")
                    continue
                store = zarr.open(val_tomo_path, mode='r')
                zarr_tomo = store['0'] if isinstance(store, zarr.hierarchy.Group) else store
                tomo_shape = zarr_tomo.shape
            except Exception as e:
                print(f"AVERTISSEMENT: Impossible de lire le tomogramme de validation pour {file_info['run_id']}. Erreur: {e}")
                continue

            patch_coords = generate_sliding_window_coords(tomo_shape, self.patch_size, self.val_overlap_fraction)
            for coords in patch_coords:
                self.patch_infos.append({
                    "tomo_idx": tomo_idx,
                    "coords": coords
                })

            print(f"INFO: {len(patch_coords)} patchs de validation uniques générés pour {file_info['run_id']}.")

    def __len__(self):
        if self.mode == 'val':
            return len(self.patch_infos)
        if self.mode == 'train' and self.steps_per_epoch:
            return self.steps_per_epoch
        
        return len(self.file_list)

    def __getitem__(self, index):
        if self.mode == 'val':
            patch_info = self.patch_infos[index]
            tomo_index = patch_info["tomo_idx"]
            coords = patch_info["coords"]
            return self.load_item(tomo_index, top_left_coords=coords)
        else: # mode 'train'
            # Échantillonnage équilibré par classe :
            # 1. Choisir une classe au hasard parmi celles qui ont des particules.
            chosen_class = random.choice(self.available_classes)
            # 2. Choisir une particule au hasard dans cette classe.
            particle_info = random.choice(self.particles_by_class[chosen_class])
            
            tomo_index_to_load = particle_info["tomo_idx"]
            center_coords_px = particle_info["center_coords_px"]
            return self.load_item(tomo_index_to_load, center_coords_px=center_coords_px)
        
    def load_item(self, index, top_left_coords=None, center_coords_px=None):
        """Méthode pour charger un élément de données."""
        file_info = self.file_list[index]
        tomo_paths = file_info["tomo_paths"]
        seg_path = file_info["seg_path"]

        if self.mode == 'train':
            available_for_training = {
                t_type: t_path for t_type, t_path in tomo_paths.items() if t_type in config.TRAINING_TOMO_TYPES
            }
            if not available_for_training:
                raise FileNotFoundError(
                    f"Aucun des tomogrammes d'entraînement spécifiés dans config.TRAINING_TOMO_TYPES "
                    f"({config.TRAINING_TOMO_TYPES}) n'a été trouvé pour le run_id {file_info['run_id']}."
                )
            chosen_tomo_type = random.choice(list(available_for_training.keys()))
            tomo_path_to_load = available_for_training[chosen_tomo_type]

            if self.use_global_norm:
                # --- Logique de mise en cache pour les tomogrammes normalisés ---
                original_path = Path(tomo_path_to_load)
                normalized_path = original_path.parent / f"{original_path.stem}_norm599.zarr"
                
                if not normalized_path.exists():
                    lock_path = normalized_path.with_suffix('.zarr.lock')
                    import time

                    try:
                        # Tenter d'acquérir un verrou en créant un fichier de manière atomique.
                        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL)
                        os.close(lock_fd)
                        
                        # Verrou acquis. Ce processus est responsable de la création du fichier.
                        try:
                            # Double-vérification au cas où un autre processus l'aurait créé juste avant.
                            if not normalized_path.exists():
                                print(f"INFO: Verrou acquis. Création de {normalized_path.name}...")
                                
                                original_store = zarr.open(tomo_path_to_load, mode='r')
                                original_tomo_data = original_store['0'] if isinstance(original_store, zarr.hierarchy.Group) else original_store
                                normalized_tomo_data = normalize_globally_percentile(original_tomo_data[:])
                                
                                try:
                                    chunks = original_tomo_data.chunks
                                    compressor = original_tomo_data.compressor
                                except AttributeError:
                                    chunks = True
                                    compressor = zarr.storage.default_compressor

                                temp_path = normalized_path.with_suffix('.zarr.tmp')
                                if temp_path.exists(): shutil.rmtree(str(temp_path))
                                
                                zarr.save_array(str(temp_path), normalized_tomo_data, chunks=chunks, compressor=compressor)
                                os.rename(temp_path, normalized_path)
                                print(f"INFO: Tomogramme normalisé sauvegardé.")
                        finally:
                            os.remove(lock_path)

                    except FileExistsError:
                        print(f"INFO: En attente du cache {normalized_path.name}...")
                        timeout = 120
                        start_time = time.time()
                        while not normalized_path.exists():
                            if not lock_path.exists() or time.time() - start_time > timeout:
                                raise TimeoutError(f"Échec de la création du cache pour {normalized_path.name}. Verrou: {lock_path}")
                            time.sleep(2)
                        print(f"INFO: Cache {normalized_path.name} trouvé.")
                
                zarr_tomo = zarr.open(str(normalized_path), mode='r')
            else:
                # Si la normalisation globale est désactivée, charger le tomogramme original.
                original_store = zarr.open(tomo_path_to_load, mode='r')
                zarr_tomo = original_store['0'] if isinstance(original_store, zarr.hierarchy.Group) else original_store
        else: # mode == 'val'
            tomo_path_to_load = tomo_paths.get('denoised')
            if not tomo_path_to_load:
                raise FileNotFoundError(f"Tomogramme 'denoised' requis pour la validation mais non trouvé pour {file_info['run_id']}")
            # Pour la validation, on ne met pas en cache, l'inférence gère la normalisation.
            val_store = zarr.open(tomo_path_to_load, mode='r')
            zarr_tomo = val_store['0'] if isinstance(val_store, zarr.hierarchy.Group) else val_store

        seg_map = np.load(seg_path)
        
        d, h, w = zarr_tomo.shape
        p_d, p_h, p_w = self.patch_size

        if top_left_coords: # Mode validation (fenêtre glissante)
            z, y, x = top_left_coords
        elif center_coords_px: # Mode entraînement (centré sur particule)
            center_z, center_y, center_x = center_coords_px
            
            # Ajouter une petite variation aléatoire pour que le centre ne soit pas toujours exact
            offset_d = np.random.randint(-p_d // 8, p_d // 8 + 1)
            offset_h = np.random.randint(-p_h // 8, p_h // 8 + 1)
            offset_w = np.random.randint(-p_w // 8, p_w // 8 + 1)

            # Calculer les coordonnées du coin supérieur gauche
            z = int(round(center_z - p_d / 2)) + offset_d
            y = int(round(center_y - p_h / 2)) + offset_h
            x = int(round(center_x - p_w / 2)) + offset_w

            # S'assurer que le patch est dans les limites du tomogramme
            z = np.clip(z, 0, d - p_d if d > p_d else 0)
            y = np.clip(y, 0, h - p_h if h > p_h else 0)
            x = np.clip(x, 0, w - p_w if w > p_w else 0)
        else:
            raise ValueError("Pour charger un item, 'top_left_coords' (validation) ou 'center_coords_px' (entraînement) doit être fourni.")
        patch = zarr_tomo[z:z+p_d, y:y+p_h, x:x+p_w]
        patch_seg = seg_map[z:z+p_d, y:y+p_h, x:x+p_w]

        sample_dict = {
            'tomo': patch.copy(),
            'seg_target': patch_seg.copy()
        }

        # Appliquer les transformations de pré-traitement (ajout de canal, normalisation)
        sample_dict = self.pre_transforms(sample_dict)

        # Appliquer les augmentations aléatoires si en mode entraînement
        if self.mode == 'train' and self.augmentation_level != 'none' and self.random_transforms:
            sample_dict = self.random_transforms(sample_dict)

        # Création de l'échantillon final pour le collate_fn
        final_sample = {
            'tomo': sample_dict['tomo'], 
            'seg_target': sample_dict['seg_target'].squeeze(0).long(), # La loss attend (D,H,W) en Long
            'run_id': file_info['run_id'],
            'patch_offset': torch.tensor([x, y, z], dtype=torch.float)
        }

        return final_sample

class CryoETDataModule(pl.LightningDataModule):
    """
    Encapsule la logique de chargement des données.
    """
    def __init__(self, czi_data_root, label_dir, patch_size, batch_size, num_workers, val_tomo_id, tomo_type, train_steps_per_epoch, radius_fraction, augmentation_level='advanced', augmentation_mode='anisotropic', validation_sw_batch_size=None, validation_overlap_fraction=0.5, use_global_norm=False):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = None
        self.val_dataset = None
        if self.hparams.validation_sw_batch_size is None:
            self.hparams.validation_sw_batch_size = self.hparams.batch_size

    def _count_training_particles(self, train_files, czi_data_root):
        """Compte le nombre total de particules dans les fichiers d'entraînement."""
        total_particles = 0
        picks_base_dir = Path(czi_data_root) / "train/overlay/ExperimentRuns"

        for file_info in train_files:
            run_id = file_info['run_id']
            picks_dir = picks_base_dir / run_id / "Picks"
            if not picks_dir.is_dir():
                continue

            for p_name in CZI_PARTICLE_CONFIG:
                json_path = picks_dir / f"{p_name}.json"
                if not json_path.exists(): continue
                
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        points = data.get('points', [])
                        if isinstance(points, list):
                            total_particles += len(points)
                except (json.JSONDecodeError, TypeError):
                    print(f"Avertissement: Impossible de parser le fichier JSON {json_path}")
        return total_particles

    def _create_file_list(self):
        """Crée la liste complète des paires de tomogrammes/labels."""
        # Le `label_dir` pointe maintenant vers le dossier de sortie de `prepare_mask_czi_multitask.py`
        # ex: output/
        #      |- segmentations_1.0/
        # NOTE: Le chemin vers les masques doit être ajusté dans config.py (LABEL_DIR)
        tomo_runs_dir = Path(self.hparams.czi_data_root) / "train/static/ExperimentRuns"
        mask_root_dir = Path(self.hparams.label_dir)
        
        seg_dir = mask_root_dir / f"segmentations_{self.hparams.radius_fraction}"
        file_list = []
        
        run_ids = sorted([d.name for d in tomo_runs_dir.iterdir() if d.is_dir() and d.name.startswith("TS_")])
        for run_id in run_ids:
            tomo_base_path = tomo_runs_dir / run_id / "VoxelSpacing10.000"
            seg_path = seg_dir / f"{run_id}.npy"

            if tomo_base_path.exists() and seg_path.exists() :
                tomo_paths = {}
                # Scanner tous les tomogrammes .zarr disponibles (denoised, isonetcorrected, etc.)
                for zarr_file in tomo_base_path.glob("*.zarr"):
                    tomo_type_name = zarr_file.stem
                    tomo_paths[tomo_type_name] = str(zarr_file)
                
                if tomo_paths:
                    file_list.append({
                        "tomo_paths": tomo_paths, 
                        "seg_path": str(seg_path), 
                        "run_id": run_id
                    })
        return file_list

    def setup(self, stage: str | None = None):
        if stage == 'fit' or stage is None:
            all_files = self._create_file_list()
            if not all_files:
                raise FileNotFoundError("Aucun fichier de données trouvé. Vérifiez les chemins dans config.py.")
            
            train_files = []
            val_files = []
            
            for file_info in all_files:
                if file_info["run_id"] == self.hparams.val_tomo_id:
                    val_files.append(file_info)
                else:
                    train_files.append(file_info)

            if not val_files:
                raise ValueError(f"L'ID du tomogramme de validation '{self.hparams.val_tomo_id}' n'a pas été trouvé dans les données.")

            # --- Calcul dynamique de steps_per_epoch ---
            train_steps = self.hparams.train_steps_per_epoch
            if train_steps is None:
                print("INFO: `train_steps_per_epoch` n'est pas défini. Calcul dynamique en fonction du nombre de particules d'entraînement.")
                num_train_particles = self._count_training_particles(train_files, self.hparams.czi_data_root)
                if num_train_particles == 0:
                    raise ValueError("Aucune particule d'entraînement trouvée. Impossible de définir `steps_per_epoch`.")
                train_steps = num_train_particles
                print(f"INFO: `train_steps_per_epoch` défini à {train_steps}.")
            # --- Fin du calcul dynamique ---

            self.train_dataset = Dataset(
                train_files, patch_size=self.hparams.patch_size, mode='train',
                steps_per_epoch=train_steps,
                augmentation_level=self.hparams.augmentation_level,
                augmentation_mode=self.hparams.augmentation_mode,
                czi_data_root=self.hparams.czi_data_root,
                use_global_norm=self.hparams.use_global_norm,
            )
            self.val_dataset = Dataset(
                val_files, 
                patch_size=self.hparams.patch_size, 
                mode='val',
                validation_overlap_fraction=self.hparams.validation_overlap_fraction
            )
            print(f"INFO: Données d'entraînement: {len(train_files)} tomogrammes, configuré pour {train_steps} patchs/époque.")
            print(f"INFO: Données de validation: {len(val_files)} tomogramme(s) divisé(s) en {len(self.val_dataset)} patchs.")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=True, # Le shuffle est implicite avec l'échantillonnage aléatoire de patchs
            collate_fn=collate_fn
        )


    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.validation_sw_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate_fn
        )