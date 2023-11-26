# %%
# Imports and initialization
import glob
import os

import arviz as az
import autoeis as ae
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import seaborn as sns
from impedance.models.circuits import CustomCircuit
from numpyro.infer import MCMC, NUTS, log_likelihood
from rich import print
from tqdm.auto import tqdm

from helpers import get_cycle_number

ae.visualization.set_plot_style()
numpyro.set_platform("cpu")

# %%
# Helper functions

def model(F, Z_true, params: dict, circuit_func: callable):
    # Sample each element of X separately
    X = []
    for name, value in params.items():
        if "n" in name:
            # Exponent of CPE elements is bounded between 0 and 1
            X_sample = numpyro.sample(name, dist.Uniform(0, 1))
        else:
            # Search over a log-normal dist spanning [0.01*u0, 100*u0]
            mean, std_dev = jnp.log(value), jnp.log(10)
            X_sample = numpyro.sample(name, dist.LogNormal(mean, std_dev))
        X.append(X_sample)
    X = jnp.array(X)

    # Predict Z using the model
    Z_pred = circuit_func(X, F)

    # Define observation model for real and imaginary parts of Z
    sigma_real = numpyro.sample("sigma_real", dist.Exponential(rate=1.0))
    numpyro.sample("obs_real", dist.Normal(Z_pred.real, sigma_real), obs=Z_true.real)
    sigma_imag = numpyro.sample("sigma_imag", dist.Exponential(rate=1.0))
    numpyro.sample("obs_imag", dist.Normal(Z_pred.imag, sigma_imag), obs=Z_true.imag)

# %%
# Generate synthetic data using a simple circuit model

circuit_str = "R0-P1-[P2,R3]"
circuit_string2 = ae.utils.impedancepy_circuit(circuit_str)
params = np.array([500, 1e-2, 0.5, 1e-1, 1, 250])
circuit = CustomCircuit(circuit_string2, initial_guess=params)

freq = np.logspace(-3, 2, 100)
# Use impedance.py to generate synthetic data
Z_impy = circuit.predict(freq)
# Use AutoEIS to generate synthetic data
func = ae.core.circuit_to_function(circuit_str)
Z_ae = func(params, freq)
assert np.allclose(Z_impy, Z_ae), "Z from impedance.py and AutoEIS do not match"
# Add some noise to the synthetic data proportional to the magnitude of Z
rng_key, rng_subkey = jax.random.split(jax.random.PRNGKey(0))
Z_noisy = Z_ae + 0.01 * Z_ae * jax.random.normal(rng_subkey, shape=Z_impy.shape)

fig, ax = plt.subplots()
ae.visualization.plot_nyquist(Z_ae, fmt="ro", label="simulated", ax=ax)
ae.visualization.plot_nyquist(Z_noisy, fmt="gx", label="noisy", ax=ax)

# %%
# Perform Bayesian optimization using hand-written model

# Populate the parameter dictionary and build circuit function
labels = ae.utils.get_parameter_labels(circuit_str)
params_dict = dict(zip(labels, params))
Zfunc = jax.jit(ae.core.circuit_to_function(circuit_str, jax=True))

# Set up and run the MCMC
num_samples = 500
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=250)
rng_key, rng_subkey = jax.random.split(jax.random.PRNGKey(0))
mcmc.run(rng_subkey, F=freq, Z_true=Z_noisy, params=params_dict, circuit_func=Zfunc)

# Convert to ArviZ InferenceData and plot trace
trace = az.from_numpyro(mcmc)
az.plot_trace(trace)

# Extract the posterior samples
posterior_samples = mcmc.get_samples()
for key, val in params_dict.items():
    std = np.std(posterior_samples[key])
    mean = np.mean(posterior_samples[key])
    err = abs(mean - val) / val
    print(f"{key:4s}: {mean:8.3f} +/- {std:<8.3f} | {err*100:4.1f}% error")

# Plot posterior distribution of R0
fig, ax = plt.subplots(figsize=(6, 4))
az.plot_posterior(trace, var_names="R0", kind="hist", ax=ax)
ax.set_xlabel("R0 (Ohms)")

# %%
# Plot Nyquist plot using the posterior distribution

# Extract the posterior samples
posterior_samples = mcmc.get_samples()

# Generate data using posterior samples
Z_posterior = np.empty((num_samples, len(freq)), dtype=complex)
params_posterior = np.array([posterior_samples[k] for k in labels]).T
assert params_posterior.shape == (num_samples, len(labels))
r2 = {"Zreal": [], "Zimag": [], "|Z|": []}

for i in tqdm(range(num_samples)):
    params_sample = params_posterior[i, :]
    Z_sample = Zfunc(params_sample, freq)
    Z_posterior[i, :] = Z_sample
    r2["Zreal"].append(ae.core.r2_score(Z_noisy.real, Z_sample.real))
    r2["Zimag"].append(ae.core.r2_score(Z_noisy.imag, Z_sample.imag))
    r2["|Z|"].append(ae.core.r2_score(np.abs(Z_noisy), np.abs(Z_sample)))

# Generate Z using posterior mean
params_mean = np.mean(params_posterior, axis=0)
Z_mean = Zfunc(params_mean, freq)

# Plot Nyquist plot with shaded areas
fig, ax = plt.subplots()
for Z_sample in Z_posterior:
    ae.visualization.plot_nyquist(Z_sample, fmt="bx", alpha=0.1, ax=ax)
ae.visualization.plot_nyquist(Z_noisy, fmt="ro", label="true", ax=ax)

# %%
# Perform Bayesian optimization using AutoEIS

eis_data = pd.DataFrame({"freq": freq, "Zreal": Z_noisy.real, "Zimag": Z_noisy.imag})
params_formatted = ae.utils.format_parameters(params, labels)
circuits = pd.DataFrame({"circuitstring": [circuit_str], "Parameters": [params_formatted]})
R = ae.core.find_ohmic_resistance(Z_noisy, freq)
circuits = ae.core.apply_heuristic_rules(circuits, ohmic_resistance=R)
out = ae.perform_bayesian_inference(eis_data, circuits)

# Extract the MCMC object from the output
mcmc_ae = out["BI_models"][0]

# Convert to ArviZ InferenceData and plot trace
trace_ae = az.from_numpyro(mcmc_ae)
az.plot_trace(trace_ae)

# Extract the posterior samples
posterior_samples_ae = mcmc_ae.get_samples()
for label in labels:
    # Scale back to the original parameter values (AutoEIS uses normalized values)
    mean_true = params[labels.index(label)]
    std = np.std(posterior_samples_ae[label]) * mean_true
    mean = np.mean(posterior_samples_ae[label]) * mean_true
    err = abs(mean - mean_true) / mean_true
    print(f"{label:4s}: {mean:8.3f} +/- {std:<8.3f} | {err*100:4.1f}% error")

# %%
# Visualize log-normal distribution to understand the prior

# Sample from a LogNormal distribution
lognormal = dist.LogNormal(loc=0.0, scale=0.2)
samples = lognormal.sample(jax.random.PRNGKey(0), (10000,))

# Plot the histogram of the samples
fig, ax = plt.subplots()
sns.histplot(samples, bins=50, ax=ax, log_scale=True)

# %%
# Plot fake violin plots over fake cycles

trace.to_netcdf("assets/trace.nc")
trace = az.from_netcdf("assets/trace.nc")

data = trace["posterior"]["R0"].to_numpy().T
data = data.repeat(10, axis=1)

fig, ax = plt.subplots()
sns.violinplot(data, inner="quartile", density_norm="count", color="b", ax=ax)

# %% 
# Real dataset: PJ122; Set up circuit model and initial guess

# Load the data (charge/discharge cycles)
path_dataset = "datasets/PJ122/eis-sorted/charge"
flist = glob.glob(os.path.join(path_dataset, "*.csv"))
flist.sort(key=get_cycle_number)
cycles = np.array([get_cycle_number(fpath) for fpath in flist])

# Define the circuit model
# path_circuit = "datasets/PJ122/ecm/PJ122_002_01_GEIS_CA2.csv"
# circuit_df = pd.read_csv(path_circuit)
# circuit_str, params_dict = ae.utils.parse_circuit_dataframe(circuit_df)
circuit_str = "R1-[P2,R3]-[P4,R5]-[P6,R7]"
num_params = ae.utils.count_params(circuit_str)
Zfunc = jax.jit(ae.core.circuit_to_function(circuit_str, jax=True))

# Find a better initial guess using impedance.py on the first cycle
freq, Zreal, Zimag = np.genfromtxt(flist[0], delimiter=",").T
Z = Zreal + 1j * Zimag
circuit = CustomCircuit(
    circuit=ae.utils.impedancepy_circuit(circuit_str),
    initial_guess=np.random.rand(num_params)
)
circuit.fit(freq, Z)
Z_ig = circuit.predict(freq)
params = circuit.parameters_
labels = ae.utils.get_parameter_labels(circuit_str)
params_dict = dict(zip(labels, params))
print(params_dict)

# Plot the Nyquist diagram of the first cycle using the initial guess
fig, ax = plt.subplots()
ae.visualization.plot_nyquist(Z, fmt="ro", label="true", ax=ax)
ae.visualization.plot_nyquist(Z_ig, fmt="bx", label="initial guess", ax=ax)

# %%
# Real datasset: PJ122; Perform Bayesian inference

# Fetch impedance data from the desired cycle
fpath = flist[0]
freq, Zreal, Zimag = np.genfromtxt(fpath, delimiter=",").T
Z = Zreal + 1j * Zimag

# Set up and run the MCMC
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=500, num_warmup=250)
rng_key, rng_subkey = jax.random.split(jax.random.PRNGKey(0))
mcmc.run(rng_subkey, F=freq, Z_true=Z, params=params_dict, circuit_func=Zfunc)

# %% 
# Real dataset: PJ122; Postprocessing, plotting, and analysis

# Convert to ArviZ InferenceData
trace = az.from_numpyro(mcmc)

# Plot trace (posterior distribution + trace) for all parameters
ax = az.plot_trace(trace)
fig = ax[0,0].figure
fig.tight_layout(h_pad=1, w_pad=2)

# Plot posterior distribution of individual parameters
param = "R1"
fig, ax = plt.subplots(figsize=(6, 4))
az.plot_posterior(trace, var_names=param, kind="hist", ax=ax, bins=10)
ax.set_title(f"{param} @ cycle {get_cycle_number(fpath)}")

# %%
# Real dataset: PJ122 (all cycles); Perform Bayesian inference

num_warmup = 1000
num_samples = 10000

nuts_kernel = NUTS(model)
mcmc_list = []

for fpath in tqdm(flist, desc="Bayesian inference"):
    # Load impedance data
    freq, Zreal, Zimag = np.genfromtxt(fpath, delimiter=",").T
    Z = Zreal + 1j * Zimag
    # Set up and run MCMC
    rng_key, rng_subkey = jax.random.split(jax.random.PRNGKey(0))
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup, progress_bar=False)
    mcmc.run(rng_subkey, F=freq, Z_true=Z, params=params_dict, circuit_func=Zfunc)
    mcmc_list.append(mcmc)

# %%
# Real dataset: PJ122 (all cycles); Gather traces and posterior distributions

# Convert to ArviZ InferenceData
trace_list = [az.from_numpyro(mcmc) for mcmc in tqdm(mcmc_list, desc="Convert MCMC to InferenceData")]

# Gather posterior distributions of parameters from all cycles
posterior = {label: np.empty((num_samples, len(mcmc_list))) for label in labels}

for i, trace in enumerate(tqdm(trace_list, desc="Gather posterior distributions")):
    for label in labels:
        posterior[label][:, i] = trace["posterior"][label].to_numpy()

# %%
# Plot posterior distribution of individual parameters (all cycles)

param = "R8"
data = posterior[param]

data_long = pd.DataFrame(data).melt(var_name="cycle", value_name=param)
# Map cycle numbers to the corresponding cycles
# Assuming cycles is in the same order as the columns of the data
data_long["cycle"] = data_long["cycle"].map(lambda x: cycles[x])
# Create the violin plot
fig, ax = plt.subplots()
sns.violinplot(x="cycle", y=param, data=data_long, inner="quartile", density_norm="count", ax=ax)
sns.despine()
xticks = ax.get_xticks()
ae.visualization.show_nticks(ax, n=10)
ax.set_yscale("log")

ae.visualization.draw_circuit(circuit_str)

# %%

# Plot posterior distribution of individual parameters (single cycle)
param = "R1"
fig, ax = plt.subplots(figsize=(6, 4))
az.plot_posterior(trace_list[2], var_names=param, kind="hist", ax=ax, bins=10)
ax.set_title(f"{param} @ cycle {get_cycle_number(flist[2])}")

# %%

# Scratchpad

# Extract the posterior samples
posterior_samples = mcmc.get_samples()
params_dict_posterior = {}
for key in params_dict.keys():
    params_dict_posterior[key] = np.mean(posterior_samples[key])

# Plot prior vs. posterior check (initial guess vs. posterior mean)
params_posterior = np.array(list(params_dict_posterior.values()))
Z_pred_posterior = Zfunc(params_posterior, freq)
Z_pred_prior = Zfunc(params, freq)

fig, ax = plt.subplots()
ae.visualization.plot_nyquist(Z, fmt="bo", label="experiment", ax=ax)
ae.visualization.plot_nyquist(Z_pred_prior, fmt="r-", label="prior mean", ax=ax)
ae.visualization.plot_nyquist(Z_pred_posterior, fmt="g-", label="posterior mean", ax=ax)

# # Compute log likelihood for each sample for both observation nodes
# kwargs = {"F": freq, "Z_true": Z, "params": params_dict_posterior, "circuit_func": Zfunc}
# log_prob_real = log_likelihood(model, posterior_samples, **kwargs)['obs_real']
# log_prob_imag = log_likelihood(model, posterior_samples, **kwargs)['obs_imag']

# # Combine and sum over all data points for both real and imaginary parts
# total_log_prob = np.sum(log_prob_real, axis=0) + np.sum(log_prob_imag, axis=0)

# # Find the sample with the highest log probability
# map_index = np.argmax(total_log_prob)
# map_estimate = {k: v[map_index] for k, v in posterior_samples.items()}
