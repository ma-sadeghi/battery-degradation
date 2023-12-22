#!/bin/bash

#SBATCH --account=ctb-jgostick
#SBATCH --cpus-per-task=1       # number of CPUs per task
#SBATCH --mem=16G               # memory; default unit is megabytes
#SBATCH --time=00:45:00         # time (HH:MM:SS)
#SBATCH --output=%a-%N-%j.out   # %N: node name, %j: jobID, %a: arrayIdx (when using job array)
#SBATCH --array=1-6

VENV=/home/aminsad/Code/venv
ROOT_DIR=/home/aminsad/Code/battery-degradation

# The next line is CRUCIAL!!!
module load python/3.11 scipy-stack/2023b
source $VENV/bin/activate

cd $ROOT_DIR

condition="charge"
cell_id=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cell_id_list_failed.txt)
path_export="$ROOT_DIR/results/trace"

python inference.py $cell_id $condition $path_export
