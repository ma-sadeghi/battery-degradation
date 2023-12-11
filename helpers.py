import os
import re
import warnings
from functools import partial

import autoeis as ae
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from impedance.models.circuits import CustomCircuit
from scipy.optimize import curve_fit
from scipy.stats import truncnorm

__all__ = [
    'get_first_identifier',
    'get_second_identifier',
    'get_test_condition',
    'get_cycle_number',
    'is_valid_eis_file',
    'filter_by_condition',
    'override_mpl_colors',
    'truncnorm2_pdf',
    'truncnorm2_rvs',
    'fit_truncnorm',
    'ecm_regression',
    'fit_circuit_parameters'
]


def get_first_identifier(fname):
    """Returns the first identifier of the battery testing filename."""
    fname = os.path.basename(fname)
    match = re.search(r'PJ\d+_(\w+)_\d+', fname)
    if match:
        match = match.group(1)
    return match


def get_second_identifier(fname):
    """Returns the second identifier of the battery testing filename."""
    fname = os.path.basename(fname)
    match = re.search(r'PJ\d+_\w+_(\d+)', fname)
    if match:
        match = match.group(1)
    return match


def get_test_condition(fname):
    """Returns 'charge' or 'discharge' given a battery testing filename."""
    identifier = get_second_identifier(fname)
    if identifier in ['01', '05', '09', '13']:
        return 'discharge'
    elif identifier in ['03', '07', '11', '15']:
        return 'charge'
    return None


def is_valid_eis_file(fname):
    """Returns True if the file is a valid EIS file, False otherwise."""
    is_fname_valid = ('EIS' in fname) and (get_test_condition(fname) is not None)
    is_data_valid = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            freq, Zreal, Zimag = np.loadtxt(fname, skiprows=1, usecols=(0, 1, 2), unpack=True)
        assert len(freq) == len(Zreal) == len(Zimag) > 0
        assert np.all(freq > 0)
    except Exception:
        is_data_valid = False
    return is_fname_valid and is_data_valid


def get_cycle_number(fname):
    """Returns the cycle number given a battery testing filename."""
    discharge_identifiers = ['01', '05', '09', '13']
    charge_identifiers = ['03', '07', '11', '15']

    condition = get_test_condition(fname)

    if condition == 'charge':
        identifiers = charge_identifiers
    elif condition == 'discharge':
        identifiers = discharge_identifiers
    else:
        return None

    first_id = get_first_identifier(fname)
    second_id = get_second_identifier(fname)
    # Only keep the numbers (some identifiers have letters in them)
    first_id = re.sub(r'[^\d]', '', first_id)
    second_id = re.sub(r'[^\d]', '', second_id)
    # NOTE: '4*' is because there are 4 charge/discharge cycles per measurement
    # NOTE: '+1' is because Python is 0-indexed, but cycle numbers start at 1
    cycle_number = 4 * (int(first_id) - 1) + identifiers.index(second_id) + 1

    return cycle_number


def filter_by_condition(fnames, condition):
    """Returns a list of filenames that match the given condition."""
    return [f for f in fnames if get_test_condition(f) == condition]


def ecm_regression(F, Z_true, priors: dict, circuit_func: callable):
    """Defines the model for Bayesian inference of a circuit model."""
    # Sample each element of X separately
    X = jnp.array([numpyro.sample(k, v) for k, v in priors.items()])
    # Predict Z using the model
    Z_pred = circuit_func(X, F)
    # Define observation model for real and imaginary parts of Z
    sigma_real = numpyro.sample("sigma_real", dist.Exponential(rate=1.0))
    numpyro.sample("obs_real", dist.Normal(Z_pred.real, sigma_real), obs=Z_true.real)
    sigma_imag = numpyro.sample("sigma_imag", dist.Exponential(rate=1.0))
    numpyro.sample("obs_imag", dist.Normal(Z_pred.imag, sigma_imag), obs=Z_true.imag)


def truncnorm2_pdf(x, loc, scale, low, high):
    """SciPy's truncnorm with human-readable low and high args."""
    a, b = (low - loc) / scale, (high - loc) / scale
    return truncnorm.pdf(x, a, b, loc=loc, scale=scale)


def truncnorm2_rvs(loc, scale, low, high, size):
    """SciPy's truncnorm with human-readable low and high args."""
    a, b = (low - loc) / scale, (high - loc) / scale
    return truncnorm.rvs(a, b, loc=loc, scale=scale, size=size)


# ! this function is not robust, instead use norm.fit to find loc and scale,
# ! use low, high = 0, 1 to find a, b -> build a truncnorm using loc, scale, a, b
def fit_truncnorm(samples, low, high):
    """Finds the parameters that best fit a truncnorm distribution."""
    assert len(samples) > 200, "Too few samples"
    Nx = len(samples) // 20
    p, edges = np.histogram(samples, bins=Nx, density=True)
    x = (edges[1:] + edges[:-1]) / 2
    func = partial(truncnorm2_pdf, low=low, high=high)
    popt, pcov = curve_fit(func, x, p, bounds=(0, [1, 2]))
    loc, scale = popt
    a = (low - loc) / scale
    b = (high - loc) / scale
    return a, b, loc, scale

truncnorm.fit2 = fit_truncnorm


def override_mpl_colors():
    """Overrides matplotlib's default colors with Flexoki colors."""
    # Define the Flexoki-Light color scheme based on the provided table
    # Original sequence: red, orange, yellow, green, cyan, blue, purple, magenta
    flexoki_light_colors = {
        "red": "#D14D41",
        "blue": "#4385BE",
        "green": "#879A39",
        "orange": "#DA702C",
        "purple": "#8B7EC8",
        "yellow": "#D0A215",
        "cyan": "#3AA99F",
        "magenta": "#CE5D97"
    }

    # Define the Flexoki-Light style
    flexoki_light_style = {
        "axes.prop_cycle": mpl.cycler(color=list(flexoki_light_colors.values())),
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.labelcolor": "black",
        "figure.facecolor": "white",
        "grid.color": "lightgray",
        "text.color": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.color": flexoki_light_colors["blue"],
        "patch.edgecolor": "black",
        "axes.spines.top": False,
        "axes.spines.right": False
    }

    # Apply the Flexoki-Light style
    plt.style.use(flexoki_light_style)


def fit_circuit_parameters(
    circuit_str: str,
    Z: np.ndarray[complex],
    freq: np.ndarray[float],
    maxiter=10,
    return_circuit=False
):
    """Finds parameters that best fit the EIS measurements"""
    def _fit(circuit_str, Z, freq):
        num_params = ae.utils.count_params(circuit_str)
        circuit = CustomCircuit(
            circuit=ae.utils.impedancepy_circuit(circuit_str),
            initial_guess=np.random.rand(num_params)
        )
        circuit.fit(freq, Z)
        return circuit

    for _ in range(maxiter):
        try:
            circuit = _fit(circuit_str, Z, freq)
            break
        except Exception:
            continue
    else:
        raise RuntimeError("Failed to fit the circuit")
    
    if return_circuit:
        return circuit
    return circuit.parameters_
