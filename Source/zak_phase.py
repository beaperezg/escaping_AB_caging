"""
Zak phases of the driven diamond chain: exact Floquet bands and second-order effective model.

Rewritten from `Codes/Python/topological_invariant.ipynb`. Differences with the original:
- the Floquet modes come from the exact Floquet operator of the square wave (`square_wave_floquet_operator`)
  instead of QuTiP's `FloquetBasis`;
- the effective model uses the corrected Gamma (`effective_model.compute_Gamma`);
- the Wilson loop runs over a periodic k grid without the duplicated endpoint k = pi, closing with the link
  k_{N-1} -> k_0. This is gauge invariant, so no smooth gauge is needed.

Convention: Z_j = i \\oint dk <u_j(k)| d_k |u_j(k)> = -arg prod_k <u_j(k)|u_j(k + dk)>, in (-pi, pi], for the Bloch
vectors in the basis (C, U, D) of `diamond_chain.Hk_static` (periodic in k, no intracell positions).
Bands are labelled by increasing (quasi)energy.
"""

import numpy as np
from scipy.linalg import expm

from diamond_chain import Hk_static, square_wave_floquet_operator, floquet_modes
from effective_model import J1, J2, Hk_effective


def periodic_kgrid(nk):
    """nk points k_j = -pi + 2 pi j / nk covering the Brillouin zone once (k = pi is k = -pi)."""
    return -np.pi + 2 * np.pi * np.arange(nk) / nk


def zak_phases(states):
    """
    Zak phases of all bands from the discrete Wilson loop.

    Parameters
    ----------
    states : ndarray, shape (nk, dim, nbands)
        states[i, :, j] is the Bloch vector of band j at k_i, on a periodic grid (`periodic_kgrid`).

    Returns
    -------
    ndarray, shape (nbands,), in (-pi, pi]
    """
    links = np.einsum('kib,kib->kb', states.conj(), np.roll(states, -1, axis=0))
    return np.angle(np.prod(links, axis=0).conj())


def square_wave_propagator(H_static, H_drive, A, w, t):
    """U(t, 0) of H(t) = H_static + A sign(sin(w t)) H_drive, for 0 <= t <= T."""
    T = 2 * np.pi / w
    if t <= T / 2:
        return expm(-1j * (H_static + A * H_drive) * t)

    return expm(-1j * (H_static - A * H_drive) * (t - T / 2)) @ expm(-1j * (H_static + A * H_drive) * T / 2)


def floquet_bloch_modes(kvals, phi, A, w, zeta=-1, J=1.0, t0=0.0):
    """
    Exact Floquet modes |u_j(k, t0)> of the driven bulk, sorted by quasienergy, at the time t0 within the period.

    t0 = 0 gives the stroboscopic modes (eigenvectors of U(T)); for t0 > 0 they are propagated with U(t0, 0).

    Returns
    -------
    quasienergies : ndarray, shape (nk, 3)
    modes : ndarray, shape (nk, 3, 3)
    """
    H_drive = np.diag([0, 1, zeta]).astype(complex)
    quasienergies, modes = [], []

    for k in kvals:
        H_static = Hk_static(k, phi, J)
        e, v = floquet_modes(square_wave_floquet_operator(H_static, H_drive, A, w), w)
        if t0 > 0:
            v = square_wave_propagator(H_static, H_drive, A, w, t0) @ v
        quasienergies.append(e)
        modes.append(v)

    return np.array(quasienergies), np.array(modes)


def effective_bloch_states(kvals, Atilde, w, J=1.0):
    """
    Eigenvalues and eigenvectors of H_eff(k) (Eq. h_eff_total), sorted by energy. Shapes (nk, 3) and (nk, 3, 3).
    Singular at integer Atilde (poles of Gamma).
    """
    j1, j2 = J1(Atilde, w, J), J2(Atilde, w, J)
    energies, states = np.linalg.eigh(np.array([Hk_effective(k, j1, j2) for k in kvals]))
    return energies, states
