# %%
# Imports and initialization

import argparse
import glob
import os
import pickle

import arviz as az
import autoeis as ae
import jax
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
import seaborn as sns
from impedance.models.circuits import CustomCircuit

from helpers import *

ae.visualization.set_plot_style()
numpyro.set_platform("cpu")
assert jax.lib.xla_bridge.get_backend().platform == "cpu"

# parser = argparse.ArgumentParser()
# parser.add_argument("--cell_id", type=str)
# parser.add_argument("--condition", type=str)
# args = parser.parse_args()

# %%
# Load the posteriors during cycling

path_results = "results"
cell_id = "PJ097" # if args.cell_id is None else args.cell_id
condition = "charge" # if args.condition is None else args.condition
fname = f"{cell_id}-{condition}-posterior.pkl"
fpath = os.path.join(path_results, fname)

with open(fpath, "rb") as f:
    posterior = pickle.load(f)

circuit_str = posterior["circuit"]
variables = ae.utils.get_parameter_labels(circuit_str)
cycles = posterior["cycles"]

# %%
# Plot the posteriors during cycling (violin plots)

for var in variables[:]:
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    data = posterior[var]
    data_long = pd.DataFrame(data).melt(var_name="cycle", value_name=var)
    # Map cycle numbers to the corresponding cycles
    data_long["cycle"] = data_long["cycle"].map(lambda x: cycles[x])
    # Create a series of violin plots, one for each cycle
    # `native_scale=True` tells seaborn x-axis is numerical not categorical
    kwargs_violin = {"inner": "quartile", "density_norm": "count", "native_scale": True, "linewidth": 1}
    sns.violinplot(x="cycle", y=var, data=data_long, ax=ax, **kwargs_violin)
    ae.visualization.show_nticks(ax, n=10)
    title = f"{cell_id} ({condition}) - {circuit_str}"
    ax.set_title(title)
    # Highlight the missing cycles (either due to missing data or bad fits)
    for i in range(1, len(cycles)):
        if cycles[i] - cycles[i-1] > 1:
            ax.axvspan(cycles[i-1], cycles[i], color="gray", alpha=0.2)
    # Save the figure
    fig.tight_layout()
    fname = f"{cell_id}-{condition}-{var}-violin.png"
    fpath = os.path.join(path_results, fname)
    fig.savefig(fpath, dpi=300)

# %%
# Plot divergances during sampling
