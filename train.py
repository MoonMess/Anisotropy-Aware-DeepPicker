import argparse
import torch
import os
# Forcer l'utilisation d'algorithmes déterministes pour la reproductibilité
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, StochasticWeightAveraging
import wandb
import shutil, json

import config
from utils.dataloader import CryoETDataModule
from utils.utils import get_outputbasename
from models.system import UnetSystem

class EMACallback(Callback):
    """
    Applique l'Exponential Moving Average (EMA) aux poids du modèle.
    Cette technique peut améliorer la généralisation en lissant les poids.
    """
    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay
        self.ema_state_dict = None
        self.original_state_dict = None
        print(f"INFO: Callback EMA initialisé avec un decay de {self.decay}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Met à jour les poids EMA après chaque pas d'entraînement."""
        with torch.no_grad():
            # Initialiser avec les poids du modèle si c'est le premier pas
            if self.ema_state_dict is None:
                self.ema_state_dict = {name: p.clone().detach() for name, p in pl_module.named_parameters() if p.requires_grad}

            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    ema_param = self.ema_state_dict[name]
                    ema_param.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def on_validation_epoch_start(self, trainer, pl_module):
        """Sauvegarde les poids originaux et charge les poids EMA pour la validation."""
        if self.ema_state_dict is not None:
            self.original_state_dict = {name: p.clone().detach() for name, p in pl_module.named_parameters()}
            # Charger les poids EMA dans le modèle pour la validation
            pl_module.load_state_dict(self.ema_state_dict, strict=False)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Restaure les poids originaux après la validation."""
        if self.original_state_dict is not None:
            # Restaurer les poids d'entraînement originaux
            pl_module.load_state_dict(self.original_state_dict, strict=False)
            self.original_state_dict = None

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Sauvegarde les poids EMA dans le checkpoint pour une utilisation en inférence."""
        if self.ema_state_dict is not None:
            # Créer un state_dict complet pour la sauvegarde.
            # Il contiendra les paramètres moyennés par EMA et les buffers du modèle original.
            full_ema_state_dict = pl_module.state_dict()
            # Remplacer les poids dans la copie par les poids EMA
            for key, param in self.ema_state_dict.items():
                if key in full_ema_state_dict and full_ema_state_dict[key].shape == param.shape:
                    full_ema_state_dict[key] = param
            
            checkpoint['ema_state_dict'] = full_ema_state_dict

    def state_dict(self):
        return self.ema_state_dict.copy() if self.ema_state_dict is not None else {}

    def load_state_dict(self, state_dict):
        self.ema_state_dict = state_dict

def main():
    parser = argparse.ArgumentParser(description="Script d'entraînement pour ResUNet3D avec PyTorch Lightning et K-Fold Cross-Validation.")
    parser.add_argument(
        "--encoder_name", type=str, default="resnet50",
        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'deepfinder2'],
        help="Choisir l'encodeur pour le modèle U-Net. 'custom' est l'implémentation originale."
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="Chemin vers un checkpoint (.ckpt) pour reprendre l'entraînement."
    )
    parser.add_argument(
        "--comment", type=str, default="",
        help="Commentaire supplémentaire pour identifier cette exécution dans les logs W&B."
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Exécute un seul fold spécifique par son index. Raccourci pour --run_specific_folds."
    )
    parser.add_argument(
        "--run_specific_folds", type=int, nargs='+', default=None,
        help="Liste d'index de folds à exécuter (ex: 0 2 5). Si non spécifié, tous les folds sont exécutés."
    )
    parser.add_argument(
        "--use_swa", action="store_true",
        help="Activer Stochastic Weight Averaging (SWA)."
    )
    parser.add_argument(
        "--use_ema", action="store_true",
        help="Activer Exponential Moving Average (EMA) des poids."
    )
    parser.add_argument(
        "--augmentation_level", type=str, default=config.AUGMENTATION_LEVEL,
        choices=['none', 'base', 'advanced'],
        help="Niveau d'augmentation à appliquer: 'none', 'base' (flips/rotations), 'advanced' (base + affine/bruit)."
    )
    parser.add_argument(
        "--augmentation_mode", type=str, default='anisotropic',
        choices=['anisotropic', 'isotropic'],
        help="Mode d'augmentation: 'anisotropic' (ne modifie pas l'axe Z de manière anisotrope) ou 'isotropic' (traite tous les axes de la même manière)."
    )
    parser.add_argument(
        "--class_loss_weights", type=float, nargs='+', default=None,
        help="Liste des poids pour la perte, un par classe, en commençant par le fond. Ex: 1.0 256.0 256.0 ..."
    )
    parser.add_argument(
        "--use_mixup", action="store_true",
        help="Activer l'augmentation Mixup."
    )
    parser.add_argument(
        "--mixup_prob", type=float, default=None,
        help="Probabilité d'appliquer Mixup si activé. Remplace la valeur de config.py."
    )
    parser.add_argument(
        "--loss_type", type=str, default='ce',
        choices=['ce', 'focal_tversky', 'ce_tversky', 'tversky', 'focal', 'focal_tversky_pp'],
        help="Type de perte à utiliser. Remplace la valeur de config.py."
    )
    parser.add_argument(
        "--use_scheduler", action="store_true",
        help="Activer le scheduler de taux d'apprentissage (le type est défini dans config.py)."
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Taux d'apprentissage à utiliser. Remplace la valeur de config.py."
    )
    parser.add_argument(
        "--patch_size", type=int, nargs=3, default=None,
        help="Taille du patch (D H W). Remplace la valeur de config.py."
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Taille du batch d'entraînement. Remplace la valeur de config.py."
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Nombre d'époques pour l'entraînement. Remplace la valeur de config.py."
    )
    parser.add_argument(
        "--norm_type", type=str, default='batch',
        choices=['batch', 'instance', 'mix'],
        help="Type de normalisation à utiliser ('batch', 'instance', 'mix'). Remplace la valeur de config.py."
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Chemin vers la racine des données CZI. Remplace la valeur de config.py."
    )
    parser.add_argument(
        "--norm_percentiles", type=float, nargs=2, default=None,
        help="Percentiles (lower upper) pour la normalisation globale. Ex: 0 95. Remplace les valeurs de config.py."
    )
    parser.add_argument(
        "--disable_global_norm", action="store_true",
        help="Désactiver la normalisation globale par percentiles (5-99) pendant l'entraînement."
    )
    parser.add_argument(
        "--use_gradnorm", action="store_true",
        help="Activer GradNorm pour équilibrer dynamiquement les pertes hybrides (ex: ce_tversky)."
    )
    parser.add_argument(
        "--gradnorm_lr", type=float, default=0.001,
        help="Taux d'apprentissage pour l'optimiseur des poids de GradNorm. Défaut: 0.025."
    )
    parser.add_argument(
        "--training_tomo_types", type=str, nargs='+', default=None,
        help="Liste des types de tomogrammes à utiliser pour l'entraînement (ex: denoised isonetcorrected). Remplace la valeur de config.py."
    )
    parser.add_argument(
        "--single_pixel_mask", action="store_true",
        help="Utiliser un masque d'un seul pixel au centroïde pour l'entraînement."
    )
    parser.add_argument(
        "--radius_fraction", type=float, default=None,
        help="Fraction du rayon à utiliser pour les masques. Remplace la valeur de config.py et --single_pixel_mask."
    )

    args = parser.parse_args()

    # Pour utiliser les Tensor Cores sur les GPU compatibles
    torch.set_float32_matmul_precision('medium')

    # Remplacer la configuration par défaut si l'argument est fourni
    if args.training_tomo_types:
        config.TRAINING_TOMO_TYPES = args.training_tomo_types
        print(f"INFO: Utilisation des types de tomogrammes d'entraînement personnalisés : {config.TRAINING_TOMO_TYPES}")

    # Désactiver le benchmark cuDNN pour la reproductibilité.
    # `deterministic=True` dans le Trainer le fait aussi, mais on l'ajoute pour être explicite.
    torch.backends.cudnn.benchmark = False


    if args.class_loss_weights:
        if len(args.class_loss_weights) != (config.NUM_CLASSES + 1):
            raise ValueError(f"L'argument --class_loss_weights doit contenir {config.NUM_CLASSES + 1} valeurs (1 pour le fond + {config.NUM_CLASSES} classes), mais {len(args.class_loss_weights)} ont été fournies.")
        config.CLASS_LOSS_WEIGHTS = args.class_loss_weights
        print(f"INFO: Utilisation des poids de perte personnalisés fournis en argument : {config.CLASS_LOSS_WEIGHTS}")

    if args.norm_percentiles:
        config.NORM_LOWER_PERCENTILE, config.NORM_UPPER_PERCENTILE = args.norm_percentiles
        print(f"INFO: Utilisation des percentiles de normalisation personnalisés : {config.NORM_LOWER_PERCENTILE}-{config.NORM_UPPER_PERCENTILE}")

    if args.disable_global_norm:
        print("INFO: La normalisation globale par percentiles (5-99) est DÉSACTIVÉE pour l'entraînement.")

    # Utiliser le data_root de l'argument s'il est fourni, sinon celui de config.py
    czi_data_root = args.data_root if args.data_root is not None else config.CZI_DATA_ROOT
    label_dir = os.path.join(czi_data_root, "mask/")

    # --- Configuration pour K-Fold Cross-Validation ---
    # 1. Obtenir la liste de tous les tomogrammes disponibles
    print("INFO: Recherche des tomogrammes disponibles pour la cross-validation...")
    temp_dm_for_ids = CryoETDataModule(
        czi_data_root=czi_data_root, label_dir=label_dir,
        patch_size=config.PATCH_SIZE, batch_size=1, validation_sw_batch_size=1, num_workers=0,
        val_tomo_id="dummy", tomo_type="dummy", train_steps_per_epoch=1,
        radius_fraction=config.RADIUS_FRACTION, augmentation_level='none'
    )
    all_files = temp_dm_for_ids._create_file_list()
    all_tomo_ids = sorted([f['run_id'] for f in all_files])
    num_folds = len(all_tomo_ids)
    
    if num_folds == 0:
        raise ValueError("Aucun tomogramme trouvé. Impossible de lancer l'entraînement K-Fold.")
    
    folds_to_run = range(num_folds)
    # Priorité à l'argument --fold s'il est fourni
    if args.fold is not None:
        if args.fold < 0 or args.fold >= num_folds:
            raise ValueError(f"Le fold spécifié {args.fold} est hors des limites (0-{num_folds-1}).")
        folds_to_run = [args.fold]
        print(f"INFO: Exécution du fold unique demandé : {args.fold}")
    elif args.run_specific_folds:
        # Filtrer pour ne garder que les folds valides
        valid_folds = [f for f in args.run_specific_folds if f < num_folds]
        if not valid_folds:
            raise ValueError(f"Les folds spécifiés {args.run_specific_folds} sont hors des limites (0-{num_folds-1}).")
        folds_to_run = valid_folds
        print(f"INFO: Exécution des folds spécifiques demandés : {folds_to_run}")
    else:
        print(f"INFO: {num_folds} tomogrammes trouvés. Démarrage de l'entraînement en {num_folds}-Fold Cross-Validation.")
    
    print(f"INFO: Tomogrammes disponibles pour la validation (index -> id): {dict(enumerate(all_tomo_ids))}")

    base_output_name = get_outputbasename(args)

    # --- NOUVEAU: Sauvegarde de la configuration effective de l'entraînement ---
    checkpoint_dir_for_run = os.path.join(config.OUTPUT_ROOT, "checkpoints", base_output_name)
    os.makedirs(checkpoint_dir_for_run, exist_ok=True)

    effective_config = {}
    # 1. Ajouter les arguments de la ligne de commande
    # Convertir l'objet Namespace en dictionnaire
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    effective_config['command_line_args'] = args_dict
    
    # 2. Ajouter les paramètres pertinents du module config (ceux qui ont été utilisés)
    config_params = {}
    for key in dir(config):
        # Sauvegarder les constantes (variables en majuscules) et quelques autres
        if key.isupper() or key in ['norm_type', 'dropout_rate', 'act']:
            value = getattr(config, key)
            # S'assurer que la valeur est sérialisable en JSON
            if isinstance(value, (str, int, float, bool, list, tuple, dict, type(None))):
                config_params[key] = value
    effective_config['effective_config_params'] = config_params
    
    config_save_path = os.path.join(checkpoint_dir_for_run, 'effective_training_config.json')
    with open(config_save_path, 'w') as f:
        json.dump(effective_config, f, indent=4)
    print(f"INFO: Configuration effective de l'entraînement sauvegardée dans : {config_save_path}")

    # Définir un dossier de logs W&B unique pour ce run, pour éviter de supprimer les logs d'autres runs.
    wandb_run_dir = os.path.join(config.OUTPUT_ROOT, "wandb_logs", base_output_name)
    os.makedirs(wandb_run_dir, exist_ok=True)
    print(f"INFO: Les logs W&B pour ce run seront temporairement stockés dans : {wandb_run_dir}")

    # --- Boucle Principale K-Fold ---
    for fold_idx in folds_to_run:
        val_tomo_id_for_fold = all_tomo_ids[fold_idx]
        
        print("\n" + "="*80)
        print(f"DÉMARRAGE DU FOLD {fold_idx + 1}/{num_folds} | Tomogramme de validation: {val_tomo_id_for_fold}")
        print("="*80 + "\n")

        pl.seed_everything(42 + fold_idx, workers=True)

        # Utiliser la taille de patch de l'argument si fournie, sinon celle de config.py
        patch_size = args.patch_size if args.patch_size is not None else config.PATCH_SIZE
        if args.patch_size is not None:
            print(f"INFO: Utilisation de la taille de patch personnalisée : {patch_size}")

        # Utiliser la taille de batch de l'argument si fournie, sinon celle de config.py
        batch_size = args.batch_size if args.batch_size is not None else config.BATCH_SIZE
        if args.batch_size is not None:
            print(f"INFO: Utilisation de la taille de batch personnalisée : {batch_size}")

        # Utiliser le nombre d'époques de l'argument s'il est fourni, sinon celui de config.py
        epochs = args.epochs if args.epochs is not None else config.EPOCHS
        if args.epochs is not None:
            print(f"INFO: Utilisation du nombre d'époques personnalisé : {epochs}")

        # Déterminer la fraction de rayon à utiliser, avec priorité à l'argument de la ligne de commande
        if args.radius_fraction is not None:
            radius_fraction = args.radius_fraction
            print(f"INFO: Utilisation de la fraction de rayon personnalisée : {radius_fraction}")
        else:
            radius_fraction = config.RADIUS_FRACTION

        datamodule = CryoETDataModule(
            czi_data_root=czi_data_root,
            label_dir=label_dir,
            patch_size=patch_size,
            batch_size=batch_size,
            validation_sw_batch_size=config.VALIDATION_SW_BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            val_tomo_id=val_tomo_id_for_fold,
            tomo_type=config.TOMO_TYPE,
            train_steps_per_epoch=config.TRAIN_STEPS_PER_EPOCH,
            augmentation_level=args.augmentation_level,
            augmentation_mode=args.augmentation_mode,
            radius_fraction=radius_fraction,
            validation_overlap_fraction=config.VALIDATION_OVERLAP_FRACTION,
            use_global_norm=not args.disable_global_norm,
        )

        # Le scheduler est activé si le flag est présent, sinon il utilise la valeur de config.py (False par défaut)
        use_scheduler = args.use_scheduler or getattr(config, 'USE_SCHEDULER', False)

        # Utiliser la probabilité de mixup de l'argument si fournie, sinon celle de config.py
        mixup_prob = args.mixup_prob if args.mixup_prob is not None else getattr(config, 'MIXUP_PROB', 1.0)
        mixup_beta = getattr(config, 'MIXUP_BETA', 1.0) # Paramètre Beta pour Mixup
        if args.mixup_prob is not None:
            print(f"INFO: Utilisation de la probabilité de Mixup personnalisée : {mixup_prob}")
            
        # Utiliser le type de perte de l'argument s'il est fourni, sinon celui de config.py
        loss_type = args.loss_type if args.loss_type is not None else getattr(config, 'LOSS_TYPE', 'ce_tversky')
        if args.loss_type is not None:
            print(f"INFO: Utilisation du type de perte personnalisé : {loss_type}")

        # Utiliser le learning rate de l'argument s'il est fourni, sinon celui de config.py
        learning_rate = args.lr if args.lr is not None else config.LEARNING_RATE
        if args.lr is not None:
            print(f"INFO: Utilisation du taux d'apprentissage personnalisé : {learning_rate}")

        # Utiliser le type de normalisation de l'argument s'il est fourni, sinon celui de config.py
        norm_type = args.norm_type if args.norm_type is not None else config.norm_type
        if args.norm_type is not None:
            print(f"INFO: Utilisation du type de normalisation personnalisé : {norm_type}")

        model = UnetSystem(
            encoder_name=args.encoder_name,
            learning_rate=learning_rate,
            num_classes=config.NUM_CLASSES,
            patch_size=patch_size,
            use_scheduler=use_scheduler,
            norm_type=norm_type,
            use_mixup=args.use_mixup,
            mixup_prob=mixup_prob, # type: ignore
            use_ema=args.use_ema,
            mixup_beta=mixup_beta,
            loss_type=loss_type,
            use_global_norm=not args.disable_global_norm,
            use_gradnorm=args.use_gradnorm,
            gradnorm_lr=args.gradnorm_lr
        )
        

        fold_output_name = f"{base_output_name}_fold_{fold_idx}"
        wandb_logger = WandbLogger(
            project="deepfinder2.1-czi-kfold",
            entity=None,
            # Désactivé pour éviter les erreurs de quota de disque.
            # Les checkpoints sont déjà sauvegardés localement par ModelCheckpoint.
            log_model=False,
            save_dir=wandb_run_dir,
            name=fold_output_name,
            group=base_output_name, # Regroupe les folds dans W&B
            job_type=f'train_fold_{fold_idx}',
            config=vars(args)
        )
        wandb_logger.experiment.config.update({"val_tomo_id": val_tomo_id_for_fold})

        # Tous les checkpoints sont sauvegardés dans un seul dossier pour l'expérience.
        # Le dossier a déjà été créé pour sauvegarder la configuration.
        checkpoint_dir = checkpoint_dir_for_run
        print(f"INFO: Les checkpoints pour ce run seront sauvegardés dans : {checkpoint_dir}")

        # Callback principal basé sur le score F-beta global pour le "meilleur" modèle généraliste
        main_metric_name = f"val/czi_f{config.CZI_EVAL_BETA}_score"
        fscore_checkpoint_callback = ModelCheckpoint(
            monitor=main_metric_name,
            dirpath=checkpoint_dir,
            # Le nom du fichier inclut l'index du fold pour les différencier.
            filename=f'fold_{fold_idx}-best-fscore-{{epoch}}-f{config.CZI_EVAL_BETA}={{{main_metric_name}:.4f}}',
            save_top_k=1,
            mode='max',
            auto_insert_metric_name=False,
        )
        callbacks = [fscore_checkpoint_callback]

        # Ajouter un callback pour chaque classe pour sauvegarder les "spécialistes" F-beta
        for class_name in config.CLASS_NAMES:
            metric_name = f'val_f{config.CZI_EVAL_BETA}_{class_name}'
            class_checkpoint_callback = ModelCheckpoint(
                monitor=metric_name,
                dirpath=checkpoint_dir,
                filename=f'fold_{fold_idx}-best-{class_name}-{{epoch}}-f{config.CZI_EVAL_BETA}={{{metric_name}:.4f}}',
                save_top_k=1, # On ne garde que le meilleur modèle pour chaque spécialité
                mode='max',
                auto_insert_metric_name=False,
            )
            callbacks.append(class_checkpoint_callback)

        precision = 'bf16-mixed' if config.DEVICE == 'cuda' and torch.cuda.is_bf16_supported() else '32-true'

        # --- Ajout des callbacks SWA ou EMA ---
        if args.use_swa and args.use_ema:
            raise ValueError("SWA et EMA ne peuvent pas être activés simultanément.")

        if args.use_swa:
            print(f"INFO: SWA est activé. Démarrage à {config.SWA_EPOCH_START*100}% des époques avec un LR de {config.SWA_LRS}.")
            swa_callback = StochasticWeightAveraging(
                swa_lrs=config.SWA_LRS,
                swa_epoch_start=config.SWA_EPOCH_START
            )
            callbacks.append(swa_callback)

        if args.use_ema:
            callbacks.append(EMACallback(decay=config.EMA_DECAY))

        trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",
            max_epochs=epochs,
            logger=wandb_logger,
            callbacks=callbacks,
            precision=precision,
            gradient_clip_val=1.0 if not args.use_gradnorm else None,
            check_val_every_n_epoch=config.VALIDATE_EVERY_N_EPOCHS,
            accumulate_grad_batches=config.GRADIENT_ACCUMULATION_STEPS,
            log_every_n_steps=getattr(config, 'LOG_EVERY_N_STEPS', 50), # Utilise la valeur de config ou 50 par défaut
            # 'warn' permet d'utiliser des algorithmes potentiellement non déterministes (comme label_smoothing > 0)
            # tout en garantissant la reproductibilité pour les autres opérations, en affichant un avertissement.
            # 'True' lèverait une erreur.
            deterministic='warn'
        )

        ckpt_path = None
        if args.resume_from_checkpoint:
            # Note: La reprise depuis un checkpoint est complexe en K-Fold.
            # Cette implémentation simple reprendra le même checkpoint pour chaque fold.
            if os.path.exists(args.resume_from_checkpoint):
                print(f"INFO: Reprise de l'entraînement à partir du checkpoint : {args.resume_from_checkpoint}")
                ckpt_path = args.resume_from_checkpoint
            else:
                print(f"AVERTISSEMENT: Checkpoint non trouvé à {args.resume_from_checkpoint}. Démarrage d'un nouvel entraînement pour ce fold.")

        trainer.fit(model, datamodule, ckpt_path=ckpt_path)
        
        # Terminer la session W&B pour ce fold avant de commencer le suivant
        wandb.finish()

        # Nettoyage pour libérer la mémoire avant le prochain fold
        del datamodule, model, wandb_logger, trainer
        torch.cuda.empty_cache()

    # --- Nettoyage Final ---
     # Supprimer le dossier de logs W&B spécifique à ce run.
    if os.path.exists(wandb_run_dir) and os.path.isdir(wandb_run_dir):
        print(f"\nINFO: Nettoyage du dossier de logs W&B spécifique à ce run : {wandb_run_dir}")
        try:
            shutil.rmtree(wandb_run_dir)
            print("INFO: Dossier de logs W&B du run supprimé avec succès.")
        except OSError as e:
            print(f"ERREUR: Impossible de supprimer le dossier {wandb_run_dir}. Erreur: {e}")

    print("\n" + "="*80)
    print("Entraînement K-Fold terminé avec succès.")
    print(f"Les modèles pour chaque fold sont sauvegardés dans 'checkpoints/{base_output_name}/'.")
    print("="*80)

    # Imprimer le nom de l'expérience à la fin pour pouvoir le récupérer dans un script shell
    print(f"EXPERIMENT_NAME={base_output_name}")



if __name__ == "__main__":
    main()