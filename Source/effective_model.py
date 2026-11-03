"""
High-frequency effective model of the antisymmetrically driven diamond chain and its
mapping to a (decorated) SSH chain.

Notation as in the manuscript: Atilde = A / w, Pi_alpha = Atilde^2 - alpha^2.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Renormalized hoppings
# ---------------------------------------------------------------------------

def Pi(Atilde, alpha):
    """Pole structure Pi_alpha = Atilde^2 - alpha^2."""
    return Atilde ** 2 - alpha ** 2


def compute_Gamma(Atilde, n_max, m_max):
    """
    Dimensionless coefficient Gamma of the second-order effective Hamiltonian, Eq. (gamma_Def) of the Appendix:

        Gamma = sum_{m odd, |m| <= m_max} (1 / m^2) [ sum_{n != 0, m, |n| <= n_max} 2 Atilde^2 (-1)^n m / (n Pi_n Pi_m Pi_{n-m})
                                                      - 3 / Pi_m^2 ]

    The harmonics run over positive and negative values, and the sums are truncated symmetrically:
    both cutoffs are inclusive (|n| <= n_max, odd |m| <= m_max).

    Note
    ----
    The previous version (and the original code `convergence_expansion.ipynb`) only summed n >= 1 and m >= 1,
    which gives Gamma(1.2) = -18.56 instead of -49.01. Only the full sum reproduces the bandwidth of the exact
    Floquet spectrum.
    The terms (m, n) and (-m, -n) are equal, so the sum is evaluated over m > 0 and doubled.
    """
    Gamma = 0.0

    for m in range(1, m_max + 1, 2):
        inner = 0.0
        for n in range(-n_max, n_max + 1):
            if n != 0 and n != m:
                inner += 2 * Atilde ** 2 * (-1) ** n * m / (n * Pi(Atilde, n) * Pi(Atilde, m) * Pi(Atilde, n - m))
        Gamma += 2 * (inner - 3 / Pi(Atilde, m) ** 2) / m ** 2

    return Gamma


def J0(Atilde, J=1.0):
    """Zeroth-order renormalized hopping, J0 = i J / (Atilde pi) [1 - exp(i Atilde pi)]."""
    return 1j * J / (Atilde * np.pi) * (1 - np.exp(1j * Atilde * np.pi))


def J1(Atilde, w, J=1.0, n_max=99, m_max=99):
    """Total effective first-neighbor hopping (zeroth + second order), Eq. (j1) of the manuscript."""
    Gamma = compute_Gamma(Atilde, n_max, m_max)
    second_order = (10 * Gamma / (3 * np.pi ** 3)) * J ** 3 * Atilde / w ** 2 \
        * np.sin(np.pi * Atilde) * (1 + np.exp(1j * Atilde * np.pi))

    return J0(Atilde, J) + second_order


def J2(Atilde, w, J=1.0, n_max=99, m_max=99):
    """Effective second-neighbor hopping, Eq. (j2) of the manuscript."""
    Gamma = compute_Gamma(Atilde, n_max, m_max)

    return (2 * Gamma / np.pi ** 3) * J ** 3 * Atilde / w ** 2 \
        * np.sin(np.pi * Atilde) * (1 + np.exp(1j * Atilde * np.pi))


def Hk_effective(k, j1, j2):
    """
    Second-order effective Bloch Hamiltonian H_eff(k) of the antisymmetric drive, Eq. (h_eff_total) of the manuscript,
    in the basis (C, U, D), for the (complex) hoppings j1 = J1(Atilde, w), j2 = J2(Atilde, w) of Eqs. (j1), (j2).
    """
    h_U = j1 * (1 + np.exp(1j * k)) + j2 * (np.exp(-1j * k) + np.exp(2j * k))
    h_D = np.conj(j1) * (-1 + np.exp(1j * k)) + np.conj(j2) * (np.exp(-1j * k) - np.exp(2j * k))

    return np.array([[0, np.conj(h_U), np.conj(h_D)],
                     [h_U, 0, 0],
                     [h_D, 0, 0]])


# ---------------------------------------------------------------------------
# Diamond chain with first- and second-neighbor hoppings vs SSH chain
# ---------------------------------------------------------------------------

def diamond_chain_second_neighbors(Ncells, J1, J2):
    """
    Closed diamond chain (3 Ncells + 1 sites) with real first- (J1) and second-neighbor (J2)
    hoppings, i.e. the real-space version of the effective Hamiltonian H_eff(k) (Eq. h_eff_total)
    after gauging away the hopping phases.

    Site ordering: {C_1, U_1, D_1, C_2, ...}, index of C_n = 3 (n - 1).
    """
    N = 3 * Ncells + 1
    H = np.zeros((N, N), dtype=complex)

    # First neighbors: C_n-U_n (+J1), C_n-D_n (-J1), U_n-C_{n+1} (+J1), D_n-C_{n+1} (+J1)
    for i in range(N - 1):
        if i % 3 == 0:      # C_n
            H[i, i + 1] = J1      # U_n
            H[i, i + 2] = -J1     # D_n
        elif i % 3 == 1:    # U_n
            H[i, i + 2] = J1      # C_{n+1}
        else:               # D_n
            H[i, i + 1] = J1      # C_{n+1}

    # Second neighbors: C_n-U_{n+1} (+J2), C_n-D_{n+1} (+J2), U_n-C_{n+2} (+J2), D_n-C_{n+2} (-J2)
    for i in range(N - 4):
        if i % 3 == 0:      # C_n
            H[i, i + 4] = J2      # U_{n+1}
            H[i, i + 5] = J2      # D_{n+1}
        elif i % 3 == 1:    # U_n
            H[i, i + 5] = J2      # C_{n+2}
        else:               # D_n
            H[i, i + 4] = -J2     # C_{n+2}

    return H + H.T.conj()


def ssh_chain(Ncells, t1, t2, extra_site=False):
    """
    SSH chain with 2 Ncells sites (+1 if `extra_site`), alternating hoppings t1, t2, starting with t1.
    """
    N = 2 * Ncells + (1 if extra_site else 0)
    H = np.zeros((N, N), dtype=complex)

    for i in range(N - 1):
        H[i, i + 1] = t1 if i % 2 == 0 else t2

    return H + H.T.conj()


def eigenvalues_vs_parameter(hamiltonian, running_key, running_values, **fixed_parameters):
    """Eigenvalues of `hamiltonian(**fixed_parameters, running_key=value)` for every value in `running_values`."""
    mats = np.array([hamiltonian(**{**fixed_parameters, running_key: val}) for val in running_values])
    return np.linalg.eigvalsh(mats)


def effective_band_edges(J1, J2, nk=501):
    """
    Maximum and minimum of the upper effective band, E(k) = 2 sqrt(J1^2 + J2^2 + 2 J1 J2 cos(2k)), Eq. (bulk).

    Returns
    -------
    E_max, E_min : ndarray
        Same shape as J2. The lower band is -E(k).
    """
    kvals = np.linspace(-np.pi, np.pi, nk)
    J2 = np.atleast_1d(J2)
    band = 2 * np.sqrt(J1 ** 2 + J2[:, None] ** 2 + 2 * J1 * J2[:, None] * np.cos(2 * kvals[None, :]))

    return band.max(axis=1), band.min(axis=1)
