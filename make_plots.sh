#!/bin/bash

# Serial version
# --------------
# path_datasets="datasets/Jones2022/raw-data/fixed-discharge"
# cell_id_list=$(ls $path_datasets)

# for cell_id in $cell_id_list; do
#     fpath="$path_datasets/$cell_id"
#     echo "Processing $fpath"
#     python plots.py --cell_id=$cell_id
# done

# Parallel version
# ----------------
path_datasets="datasets/Jones2022/raw-data/variable-discharge"
cell_id_list=$(ls "$path_datasets")

# Define the number of cores you want to use
n_cores=12

process() {
    cell_id="$1"
    fpath="$path_datasets/$cell_id"
    echo "Processing $fpath"
    python plots.py --cell_id="$cell_id"
}
# Export variables to be available in parallel subshell
export path_datasets
export -f process

# Run in parallel
# NOTE: GNU parallel requires each command on a new line hence the tr command
echo $cell_id_list | tr ' ' '\n' | parallel -j $n_cores process
