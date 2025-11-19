#!/bin/bash
# ressource reservation

# -t besteffort
#OAR -q production
#OAR -l gpu=1, walltime=24:00:00
#OAR -p gpu_mem>14000 AND gpu_model!='Tesla P100-PCIE-16GB' AND gpu_model!='Quadro P6000' AND gpu_model!='Tesla V100-SXM2-32GB'AND gpu_model!='Tesla V100-PCIE-32GB'

#OAR -O OAR_output/OAR_mmessaou_%jobid%.stdout
#OAR -E OAR_output/OAR_mmessaou_%jobid%.stderr

#loading tools
module load conda
conda activate ~/storage/envs/nucleofinder


# Lancer l'entraînement et capturer le nom de l'expérience généré.
echo "--- Démarrage de l'entraînement ---"
TRAIN_OUTPUT=$(python train.py --encoder_name df1 --fold 0 --comment "gradnorm1e-3" --gradnorm_lr 1e-3 --loss_type focal_tversky)
echo "$TRAIN_OUTPUT"
EXPERIMENT_NAME=$(echo "$TRAIN_OUTPUT" | grep "EXPERIMENT_NAME=" | cut -d'=' -f2)

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "ERREUR: Impossible de récupérer le nom de l'expérience depuis la sortie de l'entraînement."
    exit 1
fi

echo "--- Entraînement terminé. Nom de l'expérience récupéré : $EXPERIMENT_NAME ---"
echo "--- Démarrage de l'évaluation dans la foulée ---"

# Lancer l'évaluation en utilisant le nom de l'expérience qui vient d'être généré.
python -u run_experiment_evaluation.py \
    --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
    --experiment_names "$EXPERIMENT_NAME" \
    --evaluation_strategy best_single \
    --force_rerun
