# %%
# Imports and initialization
import glob
import os
import time

import arviz as az
import autoeis as ae
import jax
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import seaborn as sns
from impedance.models.circuits import CustomCircuit
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS
from scipy.stats import lognorm, norm, truncnorm
from tqdm.auto import tqdm

from helpers import *

ae.visualization.set_plot_style()
numpyro.set_platform("cpu")
assert jax.lib.xla_bridge.get_backend().platform == "cpu"

# %%
# Generate synthetic data using a simple circuit model

circuit_str = "R0-P1-[P2,R3]"
params = np.array([500, 1e-2, 0.5, 1e-1, 1, 250])
circuit = CustomCircuit(
    ae.utils.impedancepy_circuit(circuit_str),
    initial_guess=params
)

freq = np.logspace(-3, 2, 100)
# Use impedance.py to generate synthetic data
Z_impy = circuit.predict(freq)
# Use AutoEIS to generate synthetic data
func = ae.core.circuit_to_function(circuit_str)
Z_ae = func(params, freq)
assert np.allclose(Z_impy, Z_ae), "Z from impedance.py and AutoEIS do not match"
# Add some noise to the synthetic data proportional to the magnitude of Z
rng_key, rng_subkey = jax.random.split(jax.random.PRNGKey(0))
# Add noise inversely proportional to the frequency
noise_base = jax.random.uniform(rng_subkey, shape=Z_impy.shape, minval=-1, maxval=1)
noise_scale = 1e-3 * (1 + 1*np.log10(freq / freq.max())**2)
Z_true = Z_ae * (1 + noise_scale * noise_base)

fig, ax = plt.subplots()
ae.visualization.plot_nyquist(Z_ae, fmt="ro", label="simulated", ax=ax)
ae.visualization.plot_nyquist(Z_true, fmt="gx", label="noisy", ax=ax)

# %%
# Perform Bayesian optimization using hand-written model

# Populate the parameter dictionary and build circuit function
labels = ae.utils.get_parameter_labels(circuit_str)
params_dict = dict(zip(labels, params))
Zfunc = ae.core.circuit_to_function(circuit_str, use_jax=True)

# Set up and run the MCMC
nuts_kernel = NUTS(ecm_regression)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=500)
rng_key, rng_subkey = jax.random.split(jax.random.PRNGKey(0))
priors = ae.utils.initialize_priors(params_dict, variables=labels)
mcmc.run(rng_subkey, F=freq, Z_true=Z_true, priors=priors, circuit_func=Zfunc)

# Convert to ArviZ InferenceData and plot trace
trace = az.from_numpyro(mcmc)
ax = az.plot_trace(trace)
fig = ax[0,0].figure
fig.tight_layout(h_pad=1, w_pad=2)

# Extract the posterior samples
samples = mcmc.get_samples()
for key, val in params_dict.items():
    std = np.std(samples[key])
    mean = np.mean(samples[key])
    err = abs(mean - val) / val
    print(f"{key:4s}: {mean:8.3f} +/- {std:<8.3f} | {err*100:4.1f}% error")

# Plot posterior distribution of R0
fig, ax = plt.subplots(figsize=(6, 4))
az.plot_posterior(trace, var_names="R0", kind="hist", ax=ax, bins=15)
ax.set_xlabel("R0 (Ohms)")

# %%
# Plot Nyquist plot using the posterior distribution

# Extract the posterior samples
samples = mcmc.get_samples()
num_samples = samples[next(iter(samples))].size

# Generate data using posterior samples
Z_posterior = np.empty((num_samples, len(freq)), dtype=complex)
samples_arr = np.array([samples[k] for k in labels]).T
assert samples_arr.shape == (num_samples, len(labels))
r2 = {"Zreal": [], "Zimag": [], "|Z|": []}

for i in tqdm(range(num_samples)):
    sample = samples_arr[i, :]
    Z_sample = Zfunc(sample, freq)
    Z_posterior[i, :] = Z_sample
    r2["Zreal"].append(ae.utils.r2_score(Z_true.real, Z_sample.real))
    r2["Zimag"].append(ae.utils.r2_score(Z_true.imag, Z_sample.imag))
    r2["|Z|"].append(ae.utils.r2_score(np.abs(Z_true), np.abs(Z_sample)))

# Generate Z using posterior mean
params_median = np.median(samples_arr, axis=0)
Z_mean = Zfunc(params_median, freq)
hpdi_Z = hpdi(Z_posterior, prob=0.95)

# Plot Nyquist plot with shaded areas
fig, ax = plt.subplots()
ae.visualization.plot_nyquist(Z_true, fmt="b^", label="true", ax=ax)
ae.visualization.plot_nyquist(Z_mean, fmt="r-", label="mean", ax=ax)
# FIXME: Not sure if this is the correct way to plot the shaded areas
ax.fill_between(Z_posterior.real.mean(axis=0), -hpdi_Z.imag[0], -hpdi_Z.imag[1], color="r", alpha=0.2)
ax.fill_betweenx(-Z_posterior.imag.mean(axis=0), hpdi_Z.real[0], hpdi_Z.real[1], color="r", alpha=0.2)

# %%
# Perform Bayesian optimization using AutoEIS

# FIXME: API is not finalized yet
R = ae.core.find_ohmic_resistance(Z_true, freq)
circuits = ae.core.apply_heuristic_rules(circuits, ohmic_resistance=R)
out = ae.perform_bayesian_inference(circuit_str, Z_true, freq, params_dict)

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
# Visualize different distributions to understand the prior

# from scipy.optimize import curve_fit

# def fit_lognorm(samples):
#     """Finds the parameters that best fit a numpyro.distributions.LogNormal."""
#     assert len(samples) > 200, "Too few samples"

# LogNormal distribution
# target_mean = 50
# target_std_log = 0.5
# loc, scale = np.log(target_mean), target_std_log
# lognormal = dist.LogNormal(loc=loc, scale=scale)
# samples = lognormal.sample(jax.random.PRNGKey(0), (100000,))
np.random.seed(0)
samples = lognorm.rvs(s=0.02, loc=2, scale=50, size=100000)
s, loc, scale = lognorm.fit(samples, floc=0)
loc_numpyro = np.log(scale)
# samples2 = lognorm.rvs(s=s, loc=loc, scale=scale, size=samples.size)
lognormal_fit = dist.LogNormal(loc=loc_numpyro, scale=s)
samples2 = lognormal_fit.sample(jax.random.PRNGKey(1), (100000,))

fig, ax = plt.subplots()
kwargs = {"stat": "density", "element": "step", "bins": 50, "alpha": 0.2, "log_scale": True}
sns.histplot(samples, ax=ax, **kwargs, label="numpyro.dist")
sns.histplot(samples2, ax=ax, **kwargs, label="numpyro.dist (fit)")
lb, ub = np.log10(samples.min()), np.log10(samples.max())
xs = np.logspace(lb, ub, 50)
ys = lognorm.pdf(xs, s=s, loc=loc, scale=scale)
ax.plot(xs, ys, "b-", label="pdf (fit)")
ax.legend()

# # TruncatedNormal distribution
# truncated_normal = dist.truncated.TruncatedNormal(loc=0, scale=0.2, low=0, high=1)
# samples = truncated_normal.sample(jax.random.PRNGKey(0), (10000,))
# fig, ax = plt.subplots()
# kwargs = {"stat": "density", "element": "step", "bins": 50, "alpha": 0.2}
# sns.histplot(samples, ax=ax, **kwargs, label="numpyro.dist")

# low, high = 0, 1
# a, b, loc, scale = truncnorm.fit2(samples, low, high)
# xs = np.linspace(0, 1, 100)
# ys = truncnorm.pdf(xs, a=a, b=b, loc=loc, scale=scale)
# ax.plot(xs, ys, "b-", label="pdf(fit)")
# ax.legend()

# %%
# Plot fake violin plots over fake cycles

trace.to_netcdf("assets/trace.nc")
trace = az.from_netcdf("assets/trace.nc")

data = trace["posterior"]["R0"].to_numpy().T
data = data.repeat(10, axis=1)

fig, ax = plt.subplots()
sns.violinplot(data, inner="quartile", density_norm="count", color="b", ax=ax)

# %% 
# Real dataset; Set up circuit model and initial guess

# Load the data (charge/discharge cycles)
path_dataset = "datasets/JonesLee2022/raw-data/fixed-discharge/PJ121"
flist = glob.glob(os.path.join(path_dataset, "*.txt"))
flist = [f for f in flist if is_valid_eis_file(f)]
flist = [f for f in flist if get_test_condition(f) == "charge"]
flist.sort(key=get_cycle_number)
cycles = np.array([get_cycle_number(fpath) for fpath in flist])
print(f"Number of cycles: {len(flist)} ({min(cycles)} to {max(cycles)})")

# Define the circuit model
circuit_str = "R1-[P2,R3]-[P4,R5]-[P6,R7]"
ae.visualization.draw_circuit(circuit_str)
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
Z_ig = circuit.predict(freq)
params = circuit.parameters_
labels = ae.utils.get_parameter_labels(circuit_str)
params_dict = dict(zip(labels, params))
print(params_dict)

# Plot the Nyquist diagram of the first cycle using the initial guess
fig, ax = plt.subplots()
ae.visualization.plot_nyquist(Z, fmt="bo", label="true", ax=ax, alpha=0.5)
ae.visualization.plot_nyquist(Z_ig, fmt="-", label="initial guess", ax=ax)

# Set up the prior distribution for each parameter
initial_priors = ae.utils.initialize_priors(params_dict, variables=labels)

# %%
# Real dataset; Perform Bayesian inference (single cycle)

# Fetch impedance data from the desired cycle
fpath = flist[0]
freq, Zreal, Zimag = np.loadtxt(flist[0], skiprows=1, unpack=True, usecols=(0, 1, 2))
# -Im(Z) is stored in the file, so we need to flip the sign
Z = Zreal - 1j * Zimag

# Set up and run the MCMC
nuts_kernel = NUTS(ecm_regression)
mcmc = MCMC(nuts_kernel, num_samples=500, num_warmup=500)
rng_key, rng_subkey = jax.random.split(jax.random.PRNGKey(0))
t0 = time.time()
mcmc.run(rng_subkey, F=freq, Z_true=Z, priors=initial_priors, circuit_func=Zfunc)
print(f"Time elapsed: {time.time() - t0:.2f} s")

# Convert to ArviZ InferenceData
samples = mcmc.get_samples()
trace = az.from_numpyro(mcmc)

# Plot trace (posterior distribution + trace) for all parameters
ax = az.plot_trace(trace)
fig = ax[0,0].figure
fig.tight_layout(h_pad=1, w_pad=2)

# %% 
# Real dataset; Postprocessing, plotting, and analysis (single cycle)

# Plot posterior distribution of individual parameters
param = "R1"
fig, ax = plt.subplots(figsize=(6, 4))
h = az.plot_posterior(trace, var_names=param, kind="kde", color="b", label="posterior", ax=ax)
ax.set_title(f"{param} @ cycle {get_cycle_number(fpath)}")

sample = samples[param]
s, loc, scale = lognorm.fit(sample, floc=0)
mydist = dist.LogNormal(loc=np.log(scale), scale=s)
sns.histplot(sample, stat="density", element="step", bins=10, ax=ax, label="posterior", alpha=0.2)
sample2 = mydist.sample(jax.random.PRNGKey(0), (len(sample),))
sns.histplot(sample2, stat="density", element="step", bins=10, ax=ax, label="fitted dist", alpha=0.2)
ax.legend()

# %%
# Real dataset; Perform Bayesian inference (all cycles)

kwargs_mcmc = {"num_warmup": 500, "num_samples": 500}
nuts_kernel = NUTS(ecm_regression)
mcmc_list = []
priors = initial_priors
rng_key = jax.random.PRNGKey(0)

for fpath in tqdm(flist[:], desc="Bayesian inference"):
    # print(f"\ncycle {get_cycle_number(fpath)}")
    # for k, v in priors.items():
    #     samples = v.sample(rng_key, (10000,))
    #     median, std = np.median(samples), np.std(samples)
    #     print(f"  │ {k:4s}: median = {median:4.2e}, std = {std:4.2e}")

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
    mcmc_list.append(mcmc)

    # Update priors: Extract posterior distributions and set them as new priors
    samples = mcmc.get_samples()
    priors = ae.utils.initialize_priors_from_posteriors(samples, variables=labels)

# %%
# Test we can fit the posterior to a distribution to use as prior for the next cycle

# var = "R7"
# sample = samples[var]

# fig, ax = plt.subplots()
# kwargs = {"stat": "proportion", "element": "step", "bins": 20}
# sns.histplot(sample, ax=ax, **kwargs, label="posterior")
# s, loc, scale = lognorm.fit(sample, floc=0)
# random_vars = lognorm.rvs(s=s, loc=loc, scale=scale, size=sample.size)
# sns.histplot(random_vars, ax=ax, **kwargs, alpha=0.5, label="fitted dist")
# ax.set_xlabel(var)
# ax.legend()

# var = "P4n"
# sample = samples[var]

# fig, ax = plt.subplots()
# kwargs = {"stat": "proportion", "element": "step", "bins": 20}
# sns.histplot(sample, ax=ax, **kwargs, label="posterior")
# loc, scale = norm.fit(sample)
# random_vars = truncnorm2_rvs(loc=loc, scale=scale, low=0, high=1, size=sample.size)
# sns.histplot(random_vars, ax=ax, **kwargs, alpha=0.5, label="fitted dist")
# ax.set_xlabel(var)
# ax.legend()

# %%
# Real dataset; Gather traces and posterior distributions (all cycles)

# Convert to ArviZ InferenceData
trace_list = [az.from_numpyro(mcmc) for mcmc in tqdm(mcmc_list, desc="MCMC -> InferenceData")]

# Gather posterior distributions of parameters from all cycles
num_samples = trace_list[0].posterior.sizes["draw"]
posterior = {label: np.empty((num_samples, len(mcmc_list))) for label in labels}

for i, trace in enumerate(tqdm(trace_list, desc="Gather posterior distributions")):
    for label in labels:
        posterior[label][:, i] = trace["posterior"][label].to_numpy()

# %%
# Plot posterior distribution of individual parameters (all cycles)

ae.visualization.set_plot_style()
params = ae.utils.get_parameter_labels(circuit_str)

for param in params[:]:
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    data = posterior[param]
    data_long = pd.DataFrame(data).melt(var_name="cycle", value_name=param)
    # Map cycle numbers to the corresponding cycles
    data_long["cycle"] = data_long["cycle"].map(lambda x: cycles[x])
    # Create a series of violin plots, one for each cycle
    # `native_scale=True` tells seaborn x-axis is numerical not categorical
    kwargs_violin = {"inner": "quartile", "density_norm": "count", "native_scale": True, "linewidth": 1}
    sns.violinplot(x="cycle", y=param, data=data_long, ax=ax, **kwargs_violin)
    ae.visualization.show_nticks(ax, n=10)
    # Highlight the missing cycles (either due to missing data or bad fits)
    for i in range(1, len(cycles)):
        if cycles[i] - cycles[i-1] > 1:
            ax.axvspan(cycles[i-1], cycles[i], color="gray", alpha=0.2)

# %%
# Plot posterior distribution of individual parameters (single cycle)

param = "R1"
fig, ax = plt.subplots(figsize=(6, 4))
az.plot_posterior(trace_list[2], var_names=param, kind="hist", ax=ax, bins=10)
ax.set_title(f"{param} @ cycle {get_cycle_number(flist[2])}")
