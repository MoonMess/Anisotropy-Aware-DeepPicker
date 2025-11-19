#!/bin/bash
# Create output/OAR_output directory if it does not exist
mkdir -p OAR_output

# WARNING, lines starting with #OAR are not comments ! They are interpreted by oarsub -S
set -x
# ressource reservation

#OAR -q production
#OAR -l cpu=1, walltime=96:00:00
#OAR -O OAR_output/OAR_download_%jobid%.stdout
#OAR -E OAR_output/OAR_download_%jobid%.stderr

#loading tools
module load conda
conda activate ~/storage/envs/nucleofinder

python download_dataset/public_dataset.py