#!/bin/bash
# ressource reservation

#OAR -q production
#OAR -l gpu=1, walltime=24:00:00
#OAR -p gpu_model='A40'

#OAR -O OAR_output/OAR_mmessaou_%jobid%.stdout
#OAR -E OAR_output/OAR_mmessaou_%jobid%.stderr

#loading tools
module load conda
conda activate ~/storage/envs/nucleofinder

# # Lancer l'entraînement et capturer le nom de l'expérience généré.
# echo "--- Démarrage de l'entraînement ---"
# TRAIN_OUTPUT=$(python train.py --encoder_name df1 --comment "basic")
# echo "$TRAIN_OUTPUT"
# EXPERIMENT_NAME=$(echo "$TRAIN_OUTPUT" | grep "EXPERIMENT_NAME=" | cut -d'=' -f2)

# if [ -z "$EXPERIMENT_NAME" ]; then
#     echo "ERREUR: Impossible de récupérer le nom de l'expérience depuis la sortie de l'entraînement."
#     exit 1
# fi

# echo "--- Entraînement terminé. Nom de l'expérience récupéré : $EXPERIMENT_NAME ---"
# echo "--- Démarrage de l'évaluation dans la foulée ---"

# Lancer l'évaluation en utilisant le nom de l'expérience qui vient d'être généré.
# Le post-traitement 'cc3d' est utilisé par défaut.

#Sans TTA et downsample 1
# python -u run_experiment_evaluation.py \
#     --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
#     --experiment_names df1_se_20251028_205854_basic \
#     --evaluation_strategy best_single \
#     --postprocessing_method peak_local_max_gpu \
#     --comment "downsample1" \
#     --force_rerun


# python -u run_experiment_evaluation.py \
#     --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
#     --experiment_names df1_se_20251028_205854_basic \
#     --evaluation_strategy best_single \
#     --postprocessing_method cc3d \
#     --comment "downsample1" \
#     --force_rerun

# python -u run_experiment_evaluation.py \
#     --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
#     --experiment_names df1_se_20251028_205854_basic \
#     --evaluation_strategy best_single \
#     --postprocessing_method watershed \
#     --comment "downsample1" \
#     --force_rerun

# python -u run_experiment_evaluation.py \
#     --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
#     --experiment_names df1_se_20251028_205854_basic \
#     --evaluation_strategy best_single \
#     --postprocessing_method meanshift \
#     --comment "downsample1" \
#     --force_rerun

#Sans TTa et downsample 2
# python -u run_experiment_evaluation.py \
#     --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
#     --experiment_names df1_se_20251028_205854_basic \
#     --evaluation_strategy best_single \
#     --postprocessing_method peak_local_max_gpu \
#     --postproc_internal_downsample 2 \
#     --comment "downsample2" \
#     --force_rerun

# python -u run_experiment_evaluation.py \
#     --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
#     --experiment_names df1_se_20251028_205854_basic \
#     --evaluation_strategy best_single \
#     --postprocessing_method cc3d \
#     --postproc_internal_downsample 2 \
#     --comment "downsample2" \
#     --force_rerun

# python -u run_experiment_evaluation.py \
#     --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
#     --experiment_names df1_se_20251028_205854_basic \
#     --evaluation_strategy best_single \
#     --postprocessing_method watershed \
#     --postproc_internal_downsample 2 \
#     --comment "downsample2" \
#     --force_rerun

python -u run_experiment_evaluation.py \
    --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
    --experiment_names df1_se_20251028_205854_basic \
    --evaluation_strategy best_single \
    --postprocessing_method meanshift \
    --postproc_internal_downsample 2 \
    --comment "downsample2" \
    --force_rerun


#TTA
# python -u run_experiment_evaluation.py \
#     --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
#     --experiment_names df1_se_20251028_205854_basic \
#     --evaluation_strategy best_single \
#     --postprocessing_method peak_local_max_gpu \
#     --use_tta  \
#     --force_rerun


#ensemble + TTA
# python -u run_experiment_evaluation.py \
#     --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
#     --experiment_names df1_se_20251028_205854_basic \
#     --evaluation_strategy kfold_ensemble \
#     --postprocessing_method peak_local_max_gpu \
#     --use_tta  \
#     --force_rerun

# #ensemble
# python -u run_experiment_evaluation.py \
#     --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
#     --experiment_names df1_se_20251028_205854_basic \
#     --evaluation_strategy kfold_ensemble \
#     --postprocessing_method peak_local_max_gpu \
#     --force_rerun

# #ensemble specialist
# python -u run_experiment_evaluation.py \
#     --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
#     --experiment_names df1_se_20251028_205854_basic \
#     --evaluation_strategy ensemble_specialists \
#     --postprocessing_method peak_local_max_gpu \
#     --force_rerun

# #ensemble specialist + tta
# python -u run_experiment_evaluation.py \
#     --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
#     --experiment_names df1_se_20251028_205854_basic \
#     --evaluation_strategy ensemble_specialists \
#     --postprocessing_method peak_local_max_gpu \
#     --use_tta  \
#     --force_rerun

# #blend mode constant
# python -u run_experiment_evaluation.py \
#     --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
#     --experiment_names df1_se_20251028_205854_basic \
#     --evaluation_strategy best_single \
#     --postprocessing_method peak_local_max_gpu \
#     --comment "blendmode_constant" \
#     --blend_mode constant \
#     --force_rerun
