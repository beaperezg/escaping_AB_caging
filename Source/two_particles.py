"""
Interacting fermions on a short driven diamond chain: Hamiltonian, dynamics and entanglement.

Merged from the original modules `multifermions_tools.py`, `multifermions_on_diamond_lattice.py`,
`entropy_calculations.py` and `contact_and_correlations_functions.py` (folder `Python/Two particles`).

Conventions
-----------
- N = 3 Ncells + 1 spatial sites (closed chain), ordered {C_1, U_1, D_1, C_2, ..., C_{N+1}}.
- Basis states are labelled 'UP|DOWN', two bit strings of length N (leftmost bit = site 0).
- The full Fock space (2^(2N) states) is projected onto the sector with fixed (Nup, Ndown).
- Partitions are given as lists of (sublattice, cell) with 1-based cells, e.g.
  A_spec = [('C', 1), ('U', 1), ('D', 1)]; B is the complement of A.
"""

from collections import defaultdict
from itertools import combinations

import numpy as np
import qutip as qt
import scipy.sparse as sp
from scipy.special import xlogy

from diamond_chain import SQUARE_WAVE, floquet_modes, site_index, square_wave_floquet_operator, stroboscopic_evolution
from sort_bands_algorithm import sort_bands


# ---------------------------------------------------------------------------
# Fock space, basis and projector
# ---------------------------------------------------------------------------

def a_op(N, j, sigma):
    """
    Fermionic annihilation operator a_{j, sigma} on the full 2N-mode Fock space.

    j is the 0-based site index, sigma = 0 (up) or 1 (down). The site order is reversed
    internally so that site 0 corresponds to the least significant bits of QuTiP's basis index.
    """
    idx = 2 * (N - j - 1) + sigma
    a = qt.fdestroy(2 * N, idx)
    dim = int(np.prod(a.dims[0]))
    a.dims = [[dim], [dim]]

    return a


def n_op(N, j, sigma):
    """Number operator n_{j, sigma}."""
    a = a_op(N, j, sigma)
    return a.dag() * a


def generate_basis_labels(N, n_up, n_dn):
    """
    Yield all 'UP|DOWN' labels with `n_up` up and `n_dn` down fermions on N sites.

    Example: list(generate_basis_labels(3, 1, 2))
             -> ['100|110', '100|101', '100|011', '010|110', ...]
    """
    if not (0 <= n_up <= N and 0 <= n_dn <= N):
        return

    def bitstrings(n_particles):
        for occupied in combinations(range(N), n_particles):
            yield ''.join('1' if i in occupied else '0' for i in range(N))

    down_bitstrings = list(bitstrings(n_dn))
    for up in bitstrings(n_up):
        for dn in down_bitstrings:
            yield f'{up}|{dn}'


def inverse_basis_label(basis_label, N):
    """Index in QuTiP's full Fock basis of the configuration 'UP|DOWN' (spins interleaved, site 0 = lowest bits)."""
    up, dn = basis_label.split('|')
    return int(''.join(u + d for u, d in zip(up[::-1], dn[::-1])), 2)


def get_subspace_projector(Ncells, Nup, Ndown):
    """
    Isometry P (shape 2^(2N) x n) from the (Nup, Ndown) sector into the full Fock space.

    Its columns follow the order of `generate_basis_labels`; operators are projected as P^dag O P.
    """
    N = 3 * Ncells + 1
    dims = 2 ** (2 * N)
    rows = np.array([inverse_basis_label(lbl, N) for lbl in generate_basis_labels(N, Nup, Ndown)])
    n = len(rows)
    P = sp.csr_matrix((np.ones(n), (rows, np.arange(n))), shape=(dims, n))

    return qt.Qobj(P, dims=[[dims], [n]])


def generate_psi0(psi0_binary, Ncells, Nup, Ndown):
    """
    Basis state of the (Nup, Ndown) sector given by the label `psi0_binary` ('UP|DOWN').

    Example: '1000001|0000000' for Ncells = 2 places one up fermion on C_1 and one on C_3.
    """
    N = 3 * Ncells + 1
    up, dn = psi0_binary.split('|')
    if len(up) != N or len(dn) != N or set(up + dn) - {'0', '1'}:
        raise ValueError(f"psi0_binary must be two bit strings of length N = {N} separated by '|'.")

    labels = list(generate_basis_labels(N, Nup, Ndown))
    if psi0_binary not in labels:
        raise ValueError('psi0_binary is not in the (Nup, Ndown) sector.')

    return qt.basis(len(labels), labels.index(psi0_binary))


# ---------------------------------------------------------------------------
# Hamiltonian
# ---------------------------------------------------------------------------

def projected_densities(Ncells, Nup, Ndown, P=None):
    """List of projected total densities N_i = n_{i,up} + n_{i,down} for every site i."""
    N = 3 * Ncells + 1
    if P is None:
        P = get_subspace_projector(Ncells, Nup, Ndown)

    return [sum((P.dag() * n_op(N, i, s) * P).tidyup() for s in (0, 1)).tidyup() for i in range(N)]


def build_hamiltonian_matrices(Ncells, Nup, Ndown, V, zeta=-1, J=1.0, phi=np.pi):
    """
    Static Hamiltonian and drive operator projected onto the (Nup, Ndown) sector.

    H_static = J sum_{n, sigma} (C_n^dag U_n + e^{i phi} D_n^dag C_n + C_{n+1}^dag D_n + U_n^dag C_{n+1} + h.c.)
             + V sum_n (N_{U,n} N_{C,n} + N_{D,n} N_{C,n} + N_{U,n} N_{C,n+1} + N_{D,n} N_{C,n+1})
    H_drive  = sum_n (N_{U,n} + zeta N_{D,n})          (multiplied by f(t) in `time_dependent_hamiltonian`)
    """
    N = 3 * Ncells + 1
    P = get_subspace_projector(Ncells, Nup, Ndown)
    densities = projected_densities(Ncells, Nup, Ndown, P)

    H = 0 * densities[0]
    H_drive = 0 * densities[0]

    for n in range(Ncells):
        iC, iU, iD, iC_next = 3 * n, 3 * n + 1, 3 * n + 2, 3 * (n + 1)

        # Hoppings
        for sigma in (0, 1):
            C, U, D, C_next = (a_op(N, i, sigma) for i in (iC, iU, iD, iC_next))
            hop = J * (P.dag() * (C.dag() * U + np.exp(1j * phi) * D.dag() * C
                                  + C_next.dag() * D + U.dag() * C_next) * P).tidyup()
            H += (hop + hop.dag()).tidyup()

        # Nearest-neighbor density-density interaction
        H += V * (densities[iC] * densities[iU] + densities[iC] * densities[iD]
                  + densities[iC_next] * densities[iU] + densities[iC_next] * densities[iD])

        # On-site drive pattern
        H_drive += densities[iU] + zeta * densities[iD]

    return H.tidyup(), H_drive.tidyup()


def time_dependent_hamiltonian(Ncells, Nup, Ndown, V, A, w, zeta=-1, J=1.0, phi=np.pi):
    """H(t) = H_static + f(t) H_drive with the square wave f(t) = A sign(sin(w t))."""
    H_static, H_drive = build_hamiltonian_matrices(Ncells, Nup, Ndown, V, zeta, J, phi)
    return H_static + qt.QobjEvo([A * H_drive, SQUARE_WAVE], args={'w': w})


# ---------------------------------------------------------------------------
# Partitions and entropies
# ---------------------------------------------------------------------------

def build_partition_indices(Ncells, A_spec, B_spec=None):
    """0-based site indices of the partitions A and B (B = complement of A if B_spec is None)."""
    N = 3 * Ncells + 1
    A_sites = sorted({site_index(s, c) for (s, c) in A_spec})
    if B_spec is None:
        B_sites = sorted(set(range(N)) - set(A_sites))
    else:
        B_sites = sorted({site_index(s, c) for (s, c) in B_spec})
    if set(A_sites) & set(B_sites):
        raise ValueError('A and B overlap; choose disjoint partitions.')

    return A_sites, B_sites


def _partition_maps(labels, A_sites, B_sites):
    """For every basis label: index of the configuration restricted to A and to B, and particle numbers in A and B."""
    def encode(up, dn, sites):
        return int(''.join(up[i] + dn[i] for i in sites), 2)

    iA, iB, nA, nB = [], [], [], []
    for lbl in labels:
        up, dn = lbl.split('|')
        iA.append(encode(up, dn, A_sites))
        iB.append(encode(up, dn, B_sites))
        nA.append(sum(up[i] == '1' for i in A_sites) + sum(dn[i] == '1' for i in A_sites))
        nB.append(sum(up[i] == '1' for i in B_sites) + sum(dn[i] == '1' for i in B_sites))

    return np.array(iA), np.array(iB), np.array(nA), np.array(nB)


def reduced_density_matrix_A(psi, iA, iB, dA):
    """
    rho_A = Tr_B |psi><psi| for a pure state given by its amplitudes `psi` in the sector basis.

    The partial trace groups amplitudes with the same configuration of B. This is exact for
    fermions as long as A is a contiguous block at one end of the chain (no Jordan-Wigner signs),
    which is the case for the partitions used in the manuscript.
    """
    rhoA = np.zeros((dA, dA), dtype=complex)
    groups = defaultdict(list)
    for k, b in enumerate(iB):
        groups[b].append(k)

    for rows in groups.values():
        v = np.zeros(dA, dtype=complex)
        np.add.at(v, iA[rows], psi[rows])
        rhoA += np.outer(v, v.conj())

    return 0.5 * (rhoA + rhoA.conj().T)


def entropy_vn(rho, base=np.e):
    """Von Neumann entropy -Tr(rho log rho) of a (not necessarily normalized) density matrix."""
    evals = np.linalg.eigvalsh(rho)
    tol = 10 * np.finfo(float).eps * rho.shape[0] * max(np.abs(evals).max(), 1.0)
    evals = np.where(evals > tol, evals, 0.0)
    if evals.sum() > 0:
        evals = evals / evals.sum()

    return float(-np.sum(xlogy(evals, evals)) / np.log(base))


def entanglement_observables(states, Ncells, Nup, Ndown, A_spec, B_spec=None, base=np.e):
    """
    Entanglement measures for a list of pure states.

    Returns a dictionary with arrays (one value per state):
        'S_A'  : von Neumann entropy of rho_A,
        'p11'  : probability p_{1,1} of one particle in A and one in B,
        'S_11' : entropy of the reduced state in the (n_A, n_B) = (1, 1) sector,
        'E_P'  : operational entanglement E_P(1,1) = p_{1,1} S_11.
    """
    N = 3 * Ncells + 1
    labels = list(generate_basis_labels(N, Nup, Ndown))
    A_sites, B_sites = build_partition_indices(Ncells, A_spec, B_spec)
    iA, iB, nA, nB = _partition_maps(labels, A_sites, B_sites)
    dA = 2 ** (2 * len(A_sites))
    in_sector_11 = (nA == 1) & (nB == 1)

    S_A, p11, S_11 = [], [], []
    for state in states:
        psi = state.full().ravel() if isinstance(state, qt.Qobj) else np.asarray(state).ravel()
        S_A.append(entropy_vn(reduced_density_matrix_A(psi, iA, iB, dA), base))

        psi_11 = np.where(in_sector_11, psi, 0)
        p = float(np.vdot(psi_11, psi_11).real)
        p11.append(p)
        S_11.append(entropy_vn(reduced_density_matrix_A(psi_11 / np.sqrt(p), iA, iB, dA), base) if p > 0 else np.nan)

    S_A, p11, S_11 = np.array(S_A), np.array(p11), np.array(S_11)

    return {'S_A': S_A, 'p11': p11, 'S_11': S_11, 'E_P': p11 * S_11}


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def stroboscopic_states(H, psi0, w, ncycles, progress_bar='tqdm'):
    """States at t = mT, m = 0, ..., ncycles. Returns (tvals, list of kets)."""
    T = 2 * np.pi / w
    tvals = T * np.arange(ncycles + 1)
    result = qt.sesolve(H, psi0, tvals, args={'w': w}, options={'progress_bar': progress_bar})

    return tvals, result.states


def stroboscopic_states_exact(Ncells, Nup, Ndown, V, A, w, psi0, ncycles, zeta=-1, J=1.0, phi=np.pi):
    """
    States at t = mT, m = 0, ..., ncycles, computed with the exact Floquet operator of the square wave
    (same output as `stroboscopic_states`). Returns (tvals, list of kets).
    """
    H_static, H_drive = build_hamiltonian_matrices(Ncells, Nup, Ndown, V, zeta, J, phi)
    U = square_wave_floquet_operator(H_static.full(), H_drive.full(), A, w)
    states = stroboscopic_evolution(U, psi0.full(), ncycles)
    tvals = 2 * np.pi / w * np.arange(ncycles + 1)

    return tvals, [qt.Qobj(psi.reshape(-1, 1)) for psi in states]


def site_densities(states, Ncells, Nup, Ndown):
    """<N_i> for every site i and state. Returns an array of shape (N, len(states))."""
    densities = projected_densities(Ncells, Nup, Ndown)
    return np.array([[qt.expect(n, psi) for psi in states] for n in densities]).real


def fourier_transform(t, y):
    """Power spectrum |FFT(y)|^2 as a function of the angular frequency omega_FT."""
    freqs = 2 * np.pi * np.fft.rfftfreq(len(t), d=abs(t[1] - t[0]))
    return freqs, np.abs(np.fft.rfft(y)) ** 2


def max_entanglement(Ncells, Nup, Ndown, V, A, w, psi0, A_spec, ncycles, zeta=-1, J=1.0, phi=np.pi):
    """
    Maximum over the stroboscopic times t = mT (m <= ncycles) of S_A, E_P(1,1) and S_11.

    Used for the (A, V) colormaps of Fig. A10.
    """
    H = time_dependent_hamiltonian(Ncells, Nup, Ndown, V, A, w, zeta, J, phi)
    _, states = stroboscopic_states(H, psi0, w, ncycles, progress_bar=None)
    obs = entanglement_observables(states, Ncells, Nup, Ndown, A_spec)

    return np.nanmax(obs['S_A']), np.nanmax(obs['E_P']), np.nanmax(obs['S_11'])


def fourier_transform_of_entanglement(Ncells, Nup, Ndown, V, A, w, psi0, A_spec, ncycles, zeta=-1, J=1.0, phi=np.pi):
    """
    Power spectra of S_A(mT) and E_P(1,1)(mT) over `ncycles` driving cycles.

    Used for the colormap of Fig. A9. Returns (freqs, FT_S_A, FT_E_P).
    """
    H = time_dependent_hamiltonian(Ncells, Nup, Ndown, V, A, w, zeta, J, phi)
    tvals, states = stroboscopic_states(H, psi0, w, ncycles, progress_bar=None)
    obs = entanglement_observables(states, Ncells, Nup, Ndown, A_spec)
    freqs, FT_S_A = fourier_transform(tvals, obs['S_A'])
    _, FT_E_P = fourier_transform(tvals, obs['E_P'])

    return freqs, FT_S_A, FT_E_P


# ---------------------------------------------------------------------------
# Two-particle Floquet spectrum
# ---------------------------------------------------------------------------

def mode_occupation(states, mode):
    """
    Occupation <c_a^dag c_a> of the single-particle mode `mode` (vector of length N) in two-fermion states.

    Only for Nup = 2, Ndown = 0: the basis is then {|i, j>, i < j} in the order of itertools.combinations.
    For Slater determinants of single-particle Floquet modes this is the weight of the sectors that contain `mode`.
    """
    N = len(mode)
    pairs = np.array(list(combinations(range(N), 2)))

    occupations = []
    for psi in states:
        wavefunction = np.zeros((N, N), dtype=complex)
        wavefunction[pairs[:, 0], pairs[:, 1]] = psi
        wavefunction -= wavefunction.T
        occupations.append(np.linalg.norm(mode.conj() @ wavefunction) ** 2)

    return np.array(occupations)


def two_particle_quasienergies_vs_V(Vvals, Ncells, Nup, Ndown, A, w, psi0, zeta=-1, J=1.0, phi=np.pi):
    """
    Two-particle quasienergies of the closed chain for every V in `Vvals`, together with, for every Floquet mode:
    - the weight on the end sites C_1 and C_{N+1}, <N_{C_1} + N_{C_{N+1}}> / (Nup + Ndown) (between 0 and 1),
    - the spectral weight of the initial state, |<Phi_beta|psi0>|^2.
    The (sorted) modes are also returned, to evaluate other markers on them (e.g. with `mode_occupation`).

    The levels are followed through crossings with `sort_bands` (by the overlap of the modes at neighboring V),
    so column j is the same state for all V. Used for Fig. A11.

    Returns
    -------
    e_quasi, end_weight, initial_weight : ndarray, shape (len(Vvals), dim)
    modes : ndarray, shape (len(Vvals), dim, dim)
        modes[i, j] is the Floquet mode of level j at Vvals[i].
    """
    e_quasi = []
    modes = np.empty((len(Vvals), len(psi0)), dtype=object)
    for i, V in enumerate(Vvals):
        H_static, H_drive = build_hamiltonian_matrices(Ncells, Nup, Ndown, V, zeta, J, phi)
        e, vecs = floquet_modes(square_wave_floquet_operator(H_static.full(), H_drive.full(), A, w), w)
        e_quasi.append(e)
        for j in range(len(e)):
            modes[i, j] = qt.Qobj(vecs[:, j])

    e_quasi, sorted_modes = sort_bands(np.array(e_quasi), modes)
    modes = np.array([[mode.full().ravel() for mode in modes_V] for modes_V in sorted_modes])   # (V, band, basis)

    densities = projected_densities(Ncells, Nup, Ndown)
    N_ends = (densities[0] + densities[3 * Ncells]).full()
    end_weight = np.einsum('vbi,ik,vbk->vb', modes.conj(), N_ends, modes).real / (Nup + Ndown)
    initial_weight = np.abs(modes.conj() @ psi0) ** 2

    return e_quasi, end_weight, initial_weight, modes
