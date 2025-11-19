#!/bin/bash
# ressource reservation

#OAR -t besteffort
#OAR -l gpu=1, walltime=24:00:00
#OAR -p gpu_model='RTX A5000'

#OAR -O OAR_output/OAR_mmessaou_%jobid%.stdout
#OAR -E OAR_output/OAR_mmessaou_%jobid%.stderr
#OAR --notify mail:mounir.messaoudi@inria.fr

#loading tools
module load conda
conda activate ~/storage/envs/nucleofinder


python -u run_experiment_evaluation.py \
    --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
    --experiment_names df1_se_20251024_092016_basic_R05 \
    --evaluation_strategy best_single \
    --comment "best_single_cc3d_downsample2_RTXA5000" \
    --postproc_internal_downsample 2 \
    --force_rerun

python -u run_experiment_evaluation.py \
    --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
    --experiment_names df1_se_20251024_092016_basic_R05 \
    --evaluation_strategy best_single \
    --comment "best_single_meanshift_downsample2_RTXA5000" \
    --postprocessing_method meanshift \
    --postproc_internal_downsample 2 \
    --force_rerun \

python -u run_experiment_evaluation.py \
    --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
    --experiment_names df1_se_20251024_092016_basic_R05 \
    --evaluation_strategy best_single \
    --comment "best_single_MP_NMS_downsample2_RTXA5000" \
    --postprocessing_method peak_local_max_gpu \
    --postproc_internal_downsample 2 \
    --force_rerun

python -u run_experiment_evaluation.py \
    --experiments_dir ../../storage/outputs/deepfinder2.1/checkpoints/ \
    --experiment_names df1_se_20251024_092016_basic_R05 \
    --evaluation_strategy best_single \
    --comment "best_single_watershed_downsample2_RTXA5000" \
    --postprocessing_method watershed \
    --postproc_internal_downsample 2 \
    --force_rerun