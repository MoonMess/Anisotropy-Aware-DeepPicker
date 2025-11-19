#!/bin/bash
# ressource reservation

#OAR -q production
#OAR -l gpu=1, walltime=24:00:00
#OAR -p gpu_model='RTX A5000'


#OAR -O OAR_output/OAR_mmessaou_%jobid%.stdout
#OAR -E OAR_output/OAR_mmessaou_%jobid%.stderr

#loading tools
module load conda
conda activate ~/storage/envs/nucleofinder

# # Lancer l'entraînement et capturer le nom de l'expérience généré.
# echo "--- Démarrage de l'entraînement ---"
# TRAIN_OUTPUT=$(python train.py --encoder_name df1 --comment "best_ce" --augmentation_level advanced --use_mean_std_shift)
# echo "$TRAIN_OUTPUT"
# EXPERIMENT_NAME=$(echo "$TRAIN_OUTPUT" | grep "EXPERIMENT_NAME=" | cut -d'=' -f2)

# if [ -z "$EXPERIMENT_NAME" ]; then
#     echo "ERREUR: Impossible de récupérer le nom de l'expérience depuis la sortie de l'entraînement."
#     exit 1
# fi

echo "--- Entraînement terminé. Nom de l'expérience récupéré : $EXPERIMENT_NAME ---"

# --- Évaluation avec la stratégie 'best_single' ---
echo "--- Démarrage de l'évaluation avec la stratégie 'ensemble_specialists' ---"
# python -u run_experiment_evaluation.py \
#     --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
#     --experiment_names df1_se_20251025_233202_best_ce \
#     --evaluation_strategy kfold_ensemble \
#     --comment "kfold_ensemble_peak_local_max_gpu " \
#     --postprocessing_method peak_local_max_gpu \
#     --force_rerun

python -u run_experiment_evaluation.py \
    --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
    --experiment_names df1_se_20251025_233202_best_ce \
    --evaluation_strategy kfold_ensemble \
    --postprocessing_method peak_local_max_gpu \
    --comment "kfold_ensemble_tta_peak_local_max_gpu " \
    --use_tta

python -u run_experiment_evaluation.py \
    --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
    --experiment_names df1_se_20251025_233202_best_ce \
    --evaluation_strategy kfold_ensemble \
    --comment "kfold_ensemble_cc3d " \
    --postprocessing_method cc3d \
    --force_rerun

python -u run_experiment_evaluation.py \
    --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
    --experiment_names df1_se_20251025_233202_best_ce \
    --evaluation_strategy kfold_ensemble \
    --postprocessing_method cc3d \
    --comment "kfold_ensemble_tta_cc3d " \
    --force_rerun \
    --use_tta
