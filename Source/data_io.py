"""
Saving and loading of the numerical data used for the figures of the manuscript.

Every data file is a .npz archive containing numerical arrays plus a JSON string
with the simulation parameters, so it can be loaded without ``allow_pickle``.
"""

import json
from pathlib import Path

import numpy as np


def _to_builtin(value):
    """Convert numpy scalars/arrays to plain Python objects so they can be stored as JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    return value


def save_data(path, parameters=None, **arrays):
    """
    Save arrays and a dictionary of parameters to ``path`` (.npz).

    Parameters
    ----------
    path : str or Path
        Output file.
    parameters : dict, optional
        Simulation parameters (stored as JSON).
    **arrays
        Numerical data to be stored.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params_json = json.dumps(_to_builtin(parameters or {}))
    np.savez(path, parameters_json=np.array(params_json), **arrays)
    print(f'Saved {path.parent.name}/{path.name}: {", ".join(arrays)}')


def load_data(path, verbose=True):
    """
    Load a file written by :func:`save_data`.

    Returns
    -------
    data : dict
        Numerical arrays.
    parameters : dict
        Simulation parameters.
    """
    with np.load(path) as f:
        data = {k: f[k] for k in f.files if k != 'parameters_json'}
        parameters = json.loads(str(f['parameters_json'])) if 'parameters_json' in f.files else {}

    if verbose:
        print(f'{Path(path).name}')
        print(f'    parameters: {parameters}')
        for k, v in data.items():
            print(f'    {k}: shape {v.shape}')

    return data, parameters
