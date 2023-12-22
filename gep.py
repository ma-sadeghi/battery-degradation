# %%
# Imports and initialization
import glob
import os

import autoeis as ae
import jax
import numpy as np
import numpyro
import typer

from helpers import *

ae.visualization.set_plot_style()
numpyro.set_platform("cpu")
assert jax.lib.xla_bridge.get_backend().platform == "cpu"


def gep(cell_id: str, condition: str, cycle: int, path_export: str):
    """Performs GEP on the matching Jones2022 dataset."""
    header = f"Cell ID: {cell_id}, condition: {condition}, cycle: {cycle}"
    print(f"{header}\n{'-'*len(header)}\n")

    # %% 
    # Load the dataset and run GEP

    # Load the data (charge/discharge cycles)
    path_dataset = find_dir(cell_id, where="datasets/Jones2022")
    flist = glob.glob(os.path.join(path_dataset, "*.txt"))
    flist = [f for f in flist if is_valid_eis_file(f)]
    flist = [f for f in flist if get_test_condition(f) == condition]
    flist.sort(key=get_cycle_number)

    # Find the nearest available cycle number to the given cycle
    cycles = np.array([get_cycle_number(fpath) for fpath in flist])
    idx = np.argmin(np.abs(cycles - cycle))
    if cycles[idx] != cycle:
        print(f"Cycle {cycle} not found, using cycle {cycles[idx]} instead.")
        cycle = cycles[idx]

    # Load impedance data
    freq, Zreal, Zimag = np.loadtxt(flist[idx], skiprows=1, unpack=True, usecols=(0, 1, 2))
    # -Im(Z) is stored in the file, so we need to flip the sign
    Z = Zreal - 1j * Zimag

    # Run GEP and export to disk
    replicate_id = 0
    circuits = ae.core.generate_equivalent_circuits(Z, freq, iters=10, parallel=False)
    fpath = os.path.join(path_export, f"{cell_id}-{condition}-ecm-C{cycle}R{replicate_id}.csv")
    while os.path.exists(fpath):
        fpath = fpath.replace(f"C{cycle}R{replicate_id}", f"C{cycle}R{replicate_id+1}")
        replicate_id += 1
    circuits.to_csv(fpath, index=None)
    print(f"Exported results to {fpath}")


if __name__ == "__main__":
    typer.run(gep)
