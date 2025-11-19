import torch
import zarr
import numpy as np
from torch.amp import autocast
from monai.inferers import sliding_window_inference
from monai.utils import BlendMode
from monai.transforms import Compose, NormalizeIntensityd
from typing import Union, List, Dict, Optional
from utils.utils import normalize_globally_percentile

# Importation pour la logique d'ensemble spécialiste
from particle_config import CZI_PARTICLE_CONFIG

try:
    import mrcfile
except ImportError:
    print("AVERTISSEMENT: 'mrcfile' n'est pas installé. La lecture des tomogrammes .mrc ne fonctionnera pas. Installez avec 'pip install mrcfile'.")
    mrcfile = None



# Importation conditionnelle pour éviter les dépendances circulaires ou les erreurs si le module n'est pas utilisé
try:
    from models.system import UnetSystem
except ImportError:
    UnetSystem = None


# --- Nouvelle fonction pour l'inférence d'ensemble K-Fold ---
def run_inference_on_tomogram(
    models: Union['UnetSystem', List['UnetSystem'], Dict[str, 'UnetSystem']], 
    tomo_path: str, 
    device: torch.device, 
    patch_size: tuple,
    use_tta: bool = False, 
    precision: str = 'bf16',
    overlap_fraction: float = 0.5,
    sw_batch_size: int = 4,
    blend_mode: BlendMode = BlendMode.GAUSSIAN,
    progress: bool = True, # Affiche la barre de progression pour sliding_window_inference
    progress_desc: str = "Inférence", # Description pour la barre de progression
    use_global_norm: bool = False,
    norm_percentiles: List[float] = None
) -> torch.Tensor:
    """
    Exécute l'inférence sur un tomogramme complet en utilisant une approche par fenêtre glissante.
    Cette fonction peut gérer un modèle unique ou un ensemble de modèles.

    Args:
        models (Union[UnetSystem, List[UnetSystem]]): Le ou les modèles à utiliser pour l'inférence.
        tomo_path (str): Chemin vers le fichier tomogramme .zarr.
        device (torch.device): Le périphérique sur lequel exécuter l'inférence (ex: 'cuda').
        patch_size (tuple): La taille des patchs (D, H, W).
        use_tta (bool): Si True, active l'augmentation au moment du test (TTA).
        precision (str): Précision pour l'inférence ('bf16', 'fp16', 'fp32').
        overlap_fraction (float): Fraction de chevauchement entre les patchs.
        sw_batch_size (int): Taille du batch pour l'inférence par fenêtre glissante de MONAI.
        blend_mode (BlendMode): Mode de mélange pour les prédictions qui se chevauchent (ex: GAUSSIAN).
        progress (bool): Si True, affiche une barre de progression.
        progress_desc (str): Description pour la barre de progression.
        use_global_norm (bool): Si True, applique la normalisation globale par percentiles.
        norm_percentiles (List[float], optional): Une liste de deux floats [lower, upper] pour la normalisation. Si None, utilise les valeurs par défaut (5, 99).

    Returns:
        torch.Tensor: Une carte de probabilités de segmentation pour le tomogramme complet, sur CPU.
    """
    print(f"  INFO: Lancement de l'inférence sur le tomogramme : {tomo_path}")
    
    if tomo_path.endswith('.zarr'):
        tomo_data_np = zarr.open(tomo_path, mode='r')['0'][:]
    elif tomo_path.endswith('.mrc'):
        if mrcfile is None:
            raise ImportError("Le paquet 'mrcfile' est requis pour lire les fichiers .mrc mais n'est pas installé.")
        with mrcfile.open(tomo_path) as mrc:
            # Assurer que les données sont en float32 pour la cohérence
            tomo_data_np = mrc.data.astype(np.float32)
    else:
        raise ValueError(f"Format de tomogramme non supporté : {tomo_path}. Seuls .zarr et .mrc sont gérés.")

    # --- NOUVELLE PIPELINE DE NORMALISATION COHÉRENTE ---
    # Étape 1 (Globale): Normalisation par percentiles 5-99 sur le tomogramme entier.
    if use_global_norm:
        lower_p = norm_percentiles[0] if norm_percentiles else 5.0
        upper_p = norm_percentiles[1] if norm_percentiles else 99.0
        print(f"  INFO: Application de la normalisation globale avec les percentiles {lower_p}-{upper_p}.")
        tomo_data_np = normalize_globally_percentile(tomo_data_np, lower_p=lower_p, upper_p=upper_p)

    if precision == 'bf16':
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    elif precision == 'fp16':
        dtype = torch.float16
    else:  # fp32
        dtype = torch.float32
    
    if dtype != torch.bfloat16 and device.type == 'cuda':
        print(f"  AVERTISSEMENT: Utilisation de la précision {dtype}. L'inférence peut être plus lente.")

    tomo_gpu = torch.from_numpy(tomo_data_np).unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)
    
    if not isinstance(models, (list, dict)):
        models = [models]

    # Étape 2 (Locale): Définir la transformation Z-score qui sera appliquée à chaque patch.
    inference_transforms = Compose([
        NormalizeIntensityd(keys="tomo", nonzero=False, channel_wise=False)
    ])

    def predictor(patches: torch.Tensor) -> torch.Tensor:
        """Fonction de prédiction pour sliding_window_inference, gérant l'ensemble et la TTA."""
        transformed_patches = inference_transforms({'tomo': patches})['tomo']
        
        # --- Logique pour l'ensemble de spécialistes (comportement voulu) ---
        if isinstance(models, dict):
            if not models: return torch.zeros(0) # type: ignore
            
            num_total_classes = next(iter(models.values())).hparams.num_classes + 1
            final_seg_probs = torch.zeros((patches.shape[0], num_total_classes, *patches.shape[2:]), device=device, dtype=dtype)
            
            background_probs = torch.zeros_like(final_seg_probs[:, 0, ...])
            
            for class_name, model in models.items():
                model.eval()
                model_seg_probs = model._get_tta_predictions(transformed_patches) if use_tta else torch.softmax(model(transformed_patches)['seg'], dim=1)
                
                background_probs += model_seg_probs[:, 0, ...].to(dtype)
                
                # Récupérer l'ID de la classe spécialiste
                class_id = CZI_PARTICLE_CONFIG.get(class_name, {}).get('id')
                if class_id is not None:
                    final_seg_probs[:, class_id, ...] = model_seg_probs[:, class_id, ...].to(dtype)

            final_seg_probs[:, 0, ...] = background_probs / len(models)
            return final_seg_probs

        # --- Logique pour un ensemble de modèles généralistes (y compris K-Fold ensemble) ---
        elif isinstance(models, list):
            if not models: return torch.zeros(0) # type: ignore
            
            ensembled_seg_probs = torch.zeros((patches.shape[0], models[0].hparams.num_classes + 1, *patches.shape[2:]), device=device, dtype=dtype)
            for model in models:
                model.eval()
                model_seg_probs = model._get_tta_predictions(transformed_patches) if use_tta else torch.softmax(model(transformed_patches)['seg'], dim=1)
                ensembled_seg_probs += model_seg_probs.to(dtype)
            return ensembled_seg_probs / len(models)
        
        return torch.zeros(0) # Should not be reached

    with torch.no_grad(), autocast(device_type=device.type, dtype=dtype):
        full_seg_map = sliding_window_inference(
            inputs=tomo_gpu, 
            roi_size=patch_size, 
            sw_batch_size=sw_batch_size,
            predictor=predictor, 
            overlap=overlap_fraction, 
            mode=blend_mode, 
            progress=progress,
            #progress_kwargs={"desc": progress_desc} # This argument is for sliding_window_inference, not the predictor
        )

    return full_seg_map.squeeze(0).float().cpu()