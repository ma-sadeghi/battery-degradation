#!/bin/bash

#SBATCH --account=ctb-jgostick
#SBATCH --cpus-per-task=1       # number of CPUs per task
#SBATCH --mem=4G                # memory; default unit is megabytes
#SBATCH --time=00:60:00         # time (HH:MM:SS)
#SBATCH --output=%a-%N-%j.out   # %N: node name, %j: jobID, %a: arrayIdx (when using job array)
#SBATCH --array=0-125:10        # 0, 10, 20, ..., 120

VENV=/home/aminsad/Code/venv
ROOT_DIR=/home/aminsad/Code/battery-degradation

# The next line is CRUCIAL!!!
module load python/3.11 scipy-stack/2023b
source $VENV/bin/activate
module load julia/1.9.1

cd $ROOT_DIR

cell_id="PJ097"
condition="charge"
cycle="$SLURM_ARRAY_TASK_ID"
path_export="$ROOT_DIR/results/ecm"

python gep.py $cell_id $condition $cycle $path_export
