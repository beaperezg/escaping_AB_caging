"""
Single-particle physics of the phi-flux diamond (rhombic) chain, static and driven.

Conventions (as in the manuscript)
----------------------------------
- Bloch basis (C, U, D).
- Bloch Hamiltonian:
      H0(k) = J [[0,               1 + e^{-ik},  e^{-i phi} + e^{-ik}],
                 [1 + e^{ik},      0,            0                   ],
                 [e^{i phi} + e^{ik}, 0,         0                   ]]
- Finite chain with N unit cells. Sites are ordered as
      {C_1, U_1, D_1, C_2, U_2, D_2, ..., C_N, U_N, D_N, (C_{N+1})},
  i.e. the 0-based index of a site is 3 * (n - 1) + {C: 0, U: 1, D: 2} for cell n = 1, 2, ...
  A 'closed' chain has the extra C_{N+1} site (3N + 1 sites), an 'open' chain does not (3N sites).
- Drive: square-wave modulation of the U and D on-site energies,
      H_d(t) = f(t) * sum_n (U_n^dag U_n + zeta * D_n^dag D_n),
      f(t) = +A for 0 <= t < T/2,   -A for T/2 <= t < T,
  with zeta = +1 (symmetric) or zeta = -1 (antisymmetric driving).
"""

import numpy as np
import qutip as qt
from scipy.linalg import expm, schur


# Square wave f(t)/A = sign(sin(w t)), written as a QuTiP string coefficient.
# The 1e-10 avoids the division by zero at t = n T / 2.
SQUARE_WAVE = 'sin(w * t) / abs(sin(w * t) + 1e-10)'

SUBLATTICE = {'C': 0, 'U': 1, 'D': 2}


def site_index(sublattice, cell):
    """0-based site index of site `sublattice` ('C', 'U' or 'D') in cell `cell` (1-based)."""
    return 3 * (cell - 1) + SUBLATTICE[sublattice]


# ---------------------------------------------------------------------------
# Static chain
# ---------------------------------------------------------------------------

def Hk_static(k, phi, J=1.0, epsC=0.0, epsU=0.0, epsD=0.0):
    """3x3 Bloch Hamiltonian of the static diamond chain at momentum k and flux phi."""
    z_cu = J * (1 + np.exp(-1j * k))
    z_cd = J * (np.exp(-1j * phi) + np.exp(-1j * k))

    return np.array([[epsC, z_cu, z_cd],
                     [np.conj(z_cu), epsU, 0],
                     [np.conj(z_cd), 0, epsD]], dtype=complex)


def bulk_bands(kvals, phi, J=1.0, epsC=0.0, epsU=0.0, epsD=0.0):
    """
    Bulk bands of the static chain.

    Returns
    -------
    bands : ndarray, shape (3, len(kvals))
        Energies, sorted in increasing order for every k.
    """
    Hk = np.array([Hk_static(k, phi, J, epsC, epsU, epsD) for k in kvals])
    return np.linalg.eigvalsh(Hk).T


def finite_chain(Ncells, phi, J=1.0, epsC=0.0, epsU=0.0, epsD=0.0, ending='closed'):
    """
    Real-space Hamiltonian of a finite chain with `Ncells` unit cells.

    The Peierls phase is placed on the intracell C_n <-> D_n hopping, <C_n|H|D_n> = J e^{-i phi},
    so that the flux through every plaquette is phi.

    Parameters
    ----------
    ending : {'closed', 'open'}
        'closed' adds the terminal site C_{N+1} (3N + 1 sites); 'open' ends at {U_N, D_N} (3N sites).
    """
    if ending not in ('closed', 'open'):
        raise ValueError("ending must be 'closed' or 'open'")
    if Ncells < 1:
        raise ValueError('Ncells must be >= 1')

    Nsites = 3 * Ncells + (1 if ending == 'closed' else 0)
    H = np.zeros((Nsites, Nsites), dtype=complex)

    for n in range(Ncells):
        iC, iU, iD = 3 * n, 3 * n + 1, 3 * n + 2

        H[iC, iC] = epsC
        H[iU, iU] = epsU
        H[iD, iD] = epsD

        # Intracell hoppings
        H[iC, iU] = H[iU, iC] = J
        H[iC, iD] = J * np.exp(-1j * phi)
        H[iD, iC] = J * np.exp(+1j * phi)

        # Intercell hoppings to the next C site (the terminal C_{N+1} for a closed chain)
        iC_next = 3 * (n + 1)
        if iC_next < Nsites:
            H[iC_next, iC_next] = epsC
            H[iC_next, iU] = H[iU, iC_next] = J
            H[iC_next, iD] = H[iD, iC_next] = J

    return H


def localize_degenerate_pair(vecs):
    """
    Rotate two (degenerate) eigenvectors into the combinations localized on the left and on
    the right half of the chain.

    Inside a degenerate subspace numpy's eigh returns an arbitrary basis; this makes the
    edge states of the static closed chain (each localized at one end) well defined.

    Parameters
    ----------
    vecs : ndarray, shape (Nsites, 2)

    Returns
    -------
    ndarray, shape (Nsites, 2)
        Column 0 is localized on the left half, column 1 on the right half.
    """
    Nsites = vecs.shape[0]
    P_left = np.diag((np.arange(Nsites) < Nsites / 2).astype(float))
    _, rot = np.linalg.eigh(vecs.conj().T @ P_left @ vecs)
    # eigh sorts in increasing order: the right-localized combination comes first
    return (vecs @ rot)[:, ::-1]


# ---------------------------------------------------------------------------
# Driven chain (Floquet)
# ---------------------------------------------------------------------------

def square_wave_floquet_operator(H_static, H_drive, A, w):
    """
    U(T) = exp(-i (H_static - A H_drive) T/2) exp(-i (H_static + A H_drive) T/2),
    the one-period propagator of H(t) = H_static + A sign(sin(w t)) H_drive (same drive as `SQUARE_WAVE`).

    Exact for the piecewise-constant drive: no time integration is involved.
    """
    T = 2 * np.pi / w
    return expm(-1j * (H_static - A * H_drive) * T / 2) @ expm(-1j * (H_static + A * H_drive) * T / 2)


def floquet_modes(U, w):
    """
    Quasienergies in [-w/2, w/2) (sorted) and Floquet modes at t = 0 (columns) of the one-period propagator U.

    The Schur decomposition gives orthonormal modes also inside degenerate subspaces.
    """
    T = 2 * np.pi / w
    triangular, modes = schur(U, output='complex')
    e_quasi = -np.angle(np.diag(triangular)) / T
    order = np.argsort(e_quasi)

    return e_quasi[order], modes[:, order]


def onsite_drive(Ncells, zeta=-1, ending='closed'):
    """Drive operator diag(U_n -> 1, D_n -> zeta) of the finite chain (as in `finite_chain_driven`)."""
    onsite = np.zeros(3 * Ncells + (1 if ending == 'closed' else 0))
    onsite[1:3 * Ncells:3] = 1      # U_n
    onsite[2:3 * Ncells:3] = zeta   # D_n

    return np.diag(onsite).astype(complex)


def stroboscopic_evolution(U, psi0, ncycles):
    """States U^m psi0 for m = 0, ..., ncycles (array of shape (ncycles + 1, dim))."""
    states = [np.asarray(psi0, dtype=complex).ravel()]
    for _ in range(ncycles):
        states.append(U @ states[-1])

    return np.array(states)


def Hk_driven(k, phi, w, A, zeta=-1, J=1.0, epsC=0.0, epsU=0.0, epsD=0.0):
    """Time-dependent 3x3 Bloch Hamiltonian H0(k) + f(t) diag(0, 1, zeta) as a QuTiP QobjEvo."""
    H_static = qt.Qobj(Hk_static(k, phi, J, epsC, epsU, epsD))
    H_onsite = qt.Qobj(np.diag([0, 1, zeta]).astype(complex))

    return H_static + qt.QobjEvo([A * H_onsite, SQUARE_WAVE], args={'w': w})


def driven_bulk_quasienergies(kvals, phi, A, w, zeta=-1, J=1.0, epsC=0.0, epsU=0.0, epsD=0.0):
    """
    Quasienergies (sorted) of the driven bulk, in the first Floquet zone [-w/2, w/2),
    from the exact Floquet operator of the square wave.

    Returns
    -------
    ndarray, shape (len(kvals), 3)
    """
    H_drive = np.diag([0, 1, zeta]).astype(complex)

    return np.array([floquet_modes(square_wave_floquet_operator(Hk_static(k, phi, J, epsC, epsU, epsD), H_drive, A, w),
                                   w)[0] for k in kvals])


def finite_chain_driven(Ncells, phi, A, w, zeta=-1, J=1.0, epsC=0.0, epsU=0.0, epsD=0.0, ending='closed'):
    """Time-dependent Hamiltonian of the driven finite chain as a QuTiP QobjEvo."""
    H_static = finite_chain(Ncells, phi, J, epsC, epsU, epsD, ending)

    onsite = np.zeros(H_static.shape[0])
    onsite[1:3 * Ncells:3] = 1      # U_n
    onsite[2:3 * Ncells:3] = zeta   # D_n
    H_onsite = qt.Qobj(np.diag(onsite).astype(complex))

    return qt.Qobj(H_static) + qt.QobjEvo([A * H_onsite, SQUARE_WAVE], args={'w': w})


def driven_finite_quasienergies(Ncells, phi, A, w, zeta=-1, J=1.0, epsC=0.0, epsU=0.0, epsD=0.0,
                                ending='closed', return_modes=False):
    """
    Quasienergies (sorted, in [-w/2, w/2)) of the driven finite chain, from the exact Floquet operator of the
    square wave.

    If `return_modes` is True, also returns the Floquet modes at t = 0 as an array of shape
    (Nsites, Nsites), where column i is the mode with quasienergy i.
    """
    H_static = finite_chain(Ncells, phi, J, epsC, epsU, epsD, ending)
    e_quasi, modes = floquet_modes(square_wave_floquet_operator(H_static, onsite_drive(Ncells, zeta, ending), A, w), w)

    if return_modes:
        return e_quasi, modes

    return e_quasi


def stroboscopic_populations(H, psi0, w, ncycles, progress_bar='tqdm'):
    """
    Site occupations |<i|psi(mT)>|^2 at the stroboscopic times t = mT, m = 0, ..., ncycles.

    Parameters
    ----------
    H : QobjEvo
        Time-dependent Hamiltonian (e.g. from `finite_chain_driven`).
    psi0 : Qobj or int
        Initial state, or index of the initially occupied site.

    Returns
    -------
    tvals : ndarray, shape (ncycles + 1,)
    populations : ndarray, shape (ncycles + 1, Nsites)
    """
    dim = H.shape[0]
    if isinstance(psi0, (int, np.integer)):
        psi0 = qt.basis(dim, int(psi0))

    T = 2 * np.pi / w
    tvals = T * np.arange(ncycles + 1)
    states = qt.sesolve(H, psi0, tvals, args={'w': w}, options={'progress_bar': progress_bar}).states
    populations = np.abs(np.array([psi.full().ravel() for psi in states])) ** 2

    return tvals, populations


def stroboscopic_populations_exact(Ncells, phi, A, w, psi0, ncycles, zeta=-1, J=1.0, ending='closed'):
    """
    Site occupations |<i|psi(mT)>|^2 at t = mT, m = 0, ..., ncycles, of the driven finite chain, computed with the
    exact Floquet operator of the square wave (same output as `stroboscopic_populations`).

    psi0 is the index of the initially occupied site.

    Returns
    -------
    tvals : ndarray, shape (ncycles + 1,)
    populations : ndarray, shape (ncycles + 1, Nsites)
    """
    H_static = finite_chain(Ncells, phi, J, ending=ending)
    U = square_wave_floquet_operator(H_static, onsite_drive(Ncells, zeta, ending), A, w)

    initial = np.zeros(H_static.shape[0], dtype=complex)
    initial[psi0] = 1
    tvals = 2 * np.pi / w * np.arange(ncycles + 1)

    return tvals, np.abs(stroboscopic_evolution(U, initial, ncycles)) ** 2
