from typing import Optional, Tuple, List
import qutip as qt
import numpy as np
from tqdm.auto import tqdm


def sort_bands(energies: np.ndarray, eigvecs: qt.Qobj, tolerance: Optional[float] = 0.1,
               progress_bar: Optional[bool] = False, window: Optional[int] = 1):
    """
    Sort bands in the energies and modes arrays. Numpy output the eigensystem so that the energy is sorted in
    ascending order. When a crossing between bands is present, numpy does not guarantee the order of the bands. This
    function sorts the bands in the energies and modes arrays. To do that, we compare at each step the modes, and swap
    the bands if the eigenstate of another band_i is closer to the current one.

    Parameters
    ----------
    energies: np.ndarray
        Energies of the system. Shape (n_x, n_bands)
    modes: np.ndarray
        Modes of the system. Shape (n_x, n_bands, n_bands)
    tolerance: Optional(float, default=0.1)
        Tolerance to consider that two energies are close enough to be checked for swapping. If crossings between
        bands are still present, increase this value.
    progress_bar: Optional(bool, default=False)
        Show progress bar.
    window: Optional(int, default=1)
        Number of steps to check for swapping. By increasing this number, the algorithm is more robust but slower.

    Returns
    -------
    energies: np.ndarray
        Sorted energies of the system. Shape (n_x, n_bands)
    modes: np.ndarray
        Sorted modes of the system. Shape (n_x, n_bands, n_bands)
    """

    original_dim = eigvecs[0, 0].dims
    # Translate quantum objects into numpy arrays
    # The numpy convention is: (x_index, basis_index, band_index), but for qutip is (x_index, band_index, basis_index)
    modes = np.swapaxes(np.array([[temp[:] for temp in temp_2] for temp_2 in eigvecs]), 1, 2)
    
    # Make copy of the energies and modes
    
    energies = np.copy(energies)
    modes = np.copy(modes)

    n_x = len(energies)
    n_bands = np.shape(energies)[-1]

    # Compute the distance between the energies
    distances_energies = np.ones((n_x, n_bands, n_bands)) * np.inf
    for step in range(n_bands):
        distances_energies[:, step, step + 1:] = np.abs(energies[:, step][:, None] - energies[:, step + 1:])

    # Compute the minimum distance between the energies for each step
    distances_energies = np.min(distances_energies, axis=(1, 2))

    # Iterate over the steps
    pbar = tqdm(range(n_x - 1), desc='Sorting bands', disable=not progress_bar)
    for step in pbar:

        # If there is any possible crossing
        if np.any(distances_energies[step: np.min([step + window, n_x])] < tolerance):
            energies, modes, new_order = _swap_bands(energies, modes, step, tolerance)

    # make quantum objects again
    modes_quantumobject = []

    modes_quantumobject.append([[qt.Qobj(modes[i][:, nb], dims = original_dim) for nb in range(n_bands)] for i in range(n_x)])
    final_modes = modes_quantumobject[0]

    return energies, final_modes


def _swap_bands(energies: np.ndarray, modes: np.ndarray, step: int, tolerance: float) -> Tuple[np.ndarray, np.ndarray]:
    n_bands = np.shape(energies)[-1]
    new_order = []

    # Iterate over all the bands
    for band_i in range(n_bands):

        # Compute the distance between eigenstates in this and the next step
        distance_modes = np.zeros(n_bands)
        for band_j in range(n_bands):
            if band_j not in new_order:
                if np.abs(energies[step, band_i] - energies[step, band_j]) < tolerance:
                    distance_modes[band_j] = _compute_distance_modes(modes[step, :, band_i], modes[step + 1, :, band_j])

        new_order.append(int(np.argmax(distance_modes)))

    # Perform the swaps
    if new_order != list(range(n_bands)):
        energies, modes = _swap_2_bands(energies, modes, new_order, step + 1)

    return energies, modes, new_order


def _swap_2_bands(energies: np.ndarray, modes: np.ndarray, new_order: List[int], step: int) -> Tuple[
    np.ndarray, np.ndarray]:
    energies[step:] = energies[step:, new_order]
    modes[step:] = modes[step:, :, new_order]

    return energies, modes


def _compute_distance_modes(mode_1: np.ndarray, mode_2: np.ndarray) -> float:
    return np.abs(np.sum(mode_1.conj() * mode_2))
