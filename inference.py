# %%
# Imports and initialization
import glob
import os
import pickle

import arviz as az
import autoeis as ae
import jax
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from impedance.models.circuits import CustomCircuit
from numpyro.infer import MCMC, NUTS
from tqdm.auto import tqdm
import typer

from helpers import *

ae.visualization.set_plot_style()
numpyro.set_platform("cpu")
assert jax.lib.xla_bridge.get_backend().platform == "cpu"


def inference(cell_id: str, condition: str, path_export="results"):
    """Performs Bayesian inference on the matching Jones2022 dataset"""
    header = f"Cell ID: {cell_id}, condition: {condition}"
    print(f"{header}\n{'-'*len(header)}\n")

    # %% 
    # Set up circuit model and initial guess

    # Load the data (charge/discharge cycles)
    path_dataset = f"datasets/Jones2022/raw-data/variable-discharge/{cell_id}"
    flist = glob.glob(os.path.join(path_dataset, "*.txt"))
    flist = [f for f in flist if is_valid_eis_file(f)]
    flist = [f for f in flist if get_test_condition(f) == condition]
    flist.sort(key=get_cycle_number)

    seed = np.random.randint(2**32 - 1)
    np.random.seed(seed)
    print(f"Random seed: {seed}")

    # Define the circuit model
    # circuit_str = "R1-[P2,R3]-[P4,R5-P6]"  # ! Use seed = 23
    circuit_str = "R1-[P2,R3]-[P4,R5]-[P6,R7]"
    Zfunc = ae.core.circuit_to_function(circuit_str, use_jax=True)

    # Find a better initial guess using impedance.py on the first cycle
    freq, Zreal, Zimag = np.loadtxt(flist[0], skiprows=1, unpack=True, usecols=(0, 1, 2))
    # -Im(Z) is stored in the file, so we need to flip the sign
    Z = Zreal - 1j * Zimag
    values = fit_circuit_parameters(circuit_str, Z, freq)
    variables = ae.utils.get_parameter_labels(circuit_str)
    params_dict = dict(zip(variables, values))
    print(params_dict)

    # Set up the prior distribution for each parameter
    initial_priors = ae.utils.initialize_priors(params_dict, variables=variables)

    # %%
    # Perform Bayesian inference (all cycles)

    cycles = []
    kwargs_mcmc = {"num_warmup": 500, "num_samples": 500}
    nuts_kernel = NUTS(ecm_regression)
    mcmc_list = []
    priors = initial_priors
    rng_key = jax.random.PRNGKey(0)

    for fpath in tqdm(flist[:], desc="Bayesian inference"):
        # Load impedance data
        freq, Zreal, Zimag = np.loadtxt(fpath, skiprows=1, unpack=True, usecols=(0, 1, 2))
        # -Im(Z) is stored in the file, so we need to flip the sign
        Z = Zreal - 1j * Zimag

        # Set up and run MCMC
        rng_key, rng_subkey = jax.random.split(rng_key)
        mcmc = MCMC(nuts_kernel, **kwargs_mcmc, progress_bar=False)
        try:
            mcmc.run(rng_subkey, F=freq, Z_true=Z, priors=priors, circuit_func=Zfunc)
        except Exception as e:
            print(f"Failed to run MCMC for cycle {get_cycle_number(fpath)}. Error: {e}")
            mcmc_list.append(None)
            continue
        cycles.append(get_cycle_number(fpath))
        mcmc_list.append(mcmc)

        # Update priors for the next cycle using the current posteriors
        samples = mcmc.get_samples()
        # Guard against fitting failure
        try:
            priors = ae.utils.initialize_priors_from_posteriors(samples, variables=variables)
        except Exception as e:
            print(f"Failed to update priors for cycle {get_cycle_number(fpath)}. Error: {e}")

    # %%
    # Gather traces and posterior distributions (all cycles)

    # Convert to ArviZ InferenceData
    trace_list = [az.from_numpyro(mcmc) for mcmc in tqdm(mcmc_list, desc="MCMC -> InferenceData")]

    # Gather posterior distributions of parameters from all cycles
    num_samples = trace_list[0].posterior.sizes["draw"]
    posterior = {var: np.empty((num_samples, len(mcmc_list))) for var in variables}

    for i, trace in enumerate(tqdm(trace_list, desc="Gather posteriors")):
        for var in variables:
            posterior[var][:, i] = trace["posterior"][var].to_numpy()

    # Augment the posterior with the cycle numbers and circuit string
    posterior["cycles"] = np.array(cycles)
    posterior["circuit"] = circuit_str

    # Save posterior distributions (although contained in the trace, but for convenience)
    fname = f"{os.path.basename(path_dataset)}-{condition}-posterior.pkl"
    fpath = os.path.join(path_export, fname)
    with open(fpath, "wb") as f:
        pickle.dump(posterior, f)

    # Save MCMC traces (contains much more than the distributions)
    fname = f"{os.path.basename(path_dataset)}-{condition}-trace.pkl"
    fpath = os.path.join(path_export, fname)
    with open(fpath, "wb") as f:
        pickle.dump(trace_list, f)


if __name__ == "__main__":
    typer.run(inference)
