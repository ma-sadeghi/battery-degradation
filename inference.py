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

from helpers import *

ae.visualization.set_plot_style()
numpyro.set_platform("cpu")
assert jax.lib.xla_bridge.get_backend().platform == "cpu"

plot = True
condition = "charge"
path_results = "results"

# %% 
# Set up circuit model and initial guess

# Load the data (charge/discharge cycles)
path_dataset = "datasets/Jones2022/raw-data/fixed-discharge/PJ121"
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
num_params = ae.utils.count_params(circuit_str)
Zfunc = ae.core.circuit_to_function(circuit_str, use_jax=True)

# Find a better initial guess using impedance.py on the first cycle
freq, Zreal, Zimag = np.loadtxt(flist[0], skiprows=1, unpack=True, usecols=(0, 1, 2))
# -Im(Z) is stored in the file, so we need to flip the sign
Z = Zreal - 1j * Zimag
circuit = CustomCircuit(
    circuit=ae.utils.impedancepy_circuit(circuit_str),
    initial_guess=np.random.rand(num_params)
)
circuit.fit(freq, Z)
freq_ = np.logspace(-2.5, 4)
Z_ig = circuit.predict(freq_)
variables = ae.utils.get_parameter_labels(circuit_str)
params_dict = dict(zip(variables, circuit.parameters_))
print(params_dict)

# Plot the Nyquist diagram of the first cycle using the initial guess
if plot:
    # ae.visualization.draw_circuit(circuit_str)
    fig, ax = plt.subplots()
    ae.visualization.plot_nyquist(Z, fmt="bo", label="true", ax=ax, alpha=0.5)
    ae.visualization.plot_nyquist(Z_ig, fmt="-", label="initial guess", ax=ax)

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

for fpath in tqdm(flist[:60], desc="Bayesian inference"):
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

# Save posterior distributions to disk
fname = f"{os.path.basename(path_dataset)}-{condition}-posterior.pkl"
path_export = os.path.join(path_results, fname)
# Augment the posterior with the cycle numbers and circuit string
posterior["cycles"] = np.array(cycles)
posterior["circuit"] = circuit_str

with open(path_export, "wb") as f:
    pickle.dump(posterior, f)
