"""
Maxwellian energy distributions and related weighting functions.

Kept isolated so if you want to try non-Maxwellian distributions later
(e.g. bi-Maxwellian, Kappa, drifting), you swap the function without
touching compute_q_parallel.
"""

import numpy as np


def maxwellian_energy_pdf(E_keV, T_keV):
    """
    Wingen 2021 Eq. 12:

        p(E; T) = E / T^2 * exp(-E / T)

    E and T must be in the same units (both keV here).  Returns the PDF
    value(s) with the same shape as broadcasting E_keV against T_keV.

    A tiny floor on T prevents divide-by-zero in bins where the profile
    has decayed to essentially nothing.
    """
    E = np.asarray(E_keV, dtype=float)
    T = np.asarray(T_keV, dtype=float)
    T_pos = np.maximum(T, 1e-30)
    return np.maximum(E, 0.0) * np.exp(-E / T_pos) / (T_pos ** 2)
