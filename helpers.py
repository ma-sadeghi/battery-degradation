import os
import re

__all__ = [
    'get_first_identifier',
    'get_second_identifier',
    'get_test_condition',
    'get_cycle_number',
    'is_valid_eis_file',
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
