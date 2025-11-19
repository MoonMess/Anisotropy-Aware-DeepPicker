#!/bin/bash
# ressource reservation

#OAR -q production
#OAR -l gpu=1, walltime=48:00:00
#OAR -p gpu_model='A40'

#OAR -O OAR_output/OAR_mmessaou_%jobid%.stdout
#OAR -E OAR_output/OAR_mmessaou_%jobid%.stderr

#loading tools
module load conda
conda activate ~/storage/envs/nucleofinder

# # Lancer l'entraînement et capturer le nom de l'expérience généré.
# echo "--- Démarrage de l'entraînement ---"
# TRAIN_OUTPUT=$(python train.py --encoder_name resnet50 --comment denoised_only --training_tomo_types denoised)
# echo "$TRAIN_OUTPUT"
# EXPERIMENT_NAME=$(echo "$TRAIN_OUTPUT" | grep "EXPERIMENT_NAME=" | cut -d'=' -f2)

# if [ -z "$EXPERIMENT_NAME" ]; then
#     echo "ERREUR: Impossible de récupérer le nom de l'expérience depuis la sortie de l'entraînement."
#     exit 1
# fi

# echo "--- Entraînement terminé. Nom de l'expérience récupéré : $EXPERIMENT_NAME ---"
# echo "--- Démarrage de l'évaluation dans la foulée ---"

# # Lancer l'évaluation en utilisant le nom de l'expérience qui vient d'être généré.
# # Le post-traitement 'cc3d' est utilisé par défaut.

#Sans TTA et downsample 1
python -u run_experiment_evaluation.py \
    --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
    --experiment_names resnet50_se_20251104_162430_denoised_only \
    --evaluation_strategy best_single \
    --hp_search_strategy per_class \
    --postprocessing_method peak_local_max_gpu \
    --force_rerun

python -u run_experiment_evaluation.py \
    --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
    --experiment_names resnet50_se_20251104_162430_denoised_only \
    --evaluation_strategy best_single \
    --postprocessing_method peak_local_max_gpu \
    --use_tta  \
    --hp_search_strategy per_class \
    --force_rerun