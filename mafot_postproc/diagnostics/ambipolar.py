"""
Ambipolarity diagnostic.

The current pipeline assumes ambipolarity via c_s and gamma_e=7.  This
module measures how well that assumption holds in the actual output:
back-derive Gamma_i and Gamma_e at each wall location, then integrate
their signed sum to get a cumulative charge flux F_Q(R).

* F_Q(R) flat at zero      -> locally ambipolar (good)
* F_Q(R) returns to zero at R_max -> globally ambipolar, locally not
* F_Q(R) monotonic         -> globally non-ambipolar (something's off)
"""

import numpy as np

from ..constants import QE


def particle_flux_from_q(
    q_MW_per_m2,
    T_keV,
    gamma_sheath,
    *,
    T_floor_keV=1e-6,
):
    """
    Invert  q_|| = gamma * T * Gamma_||  to get particle flux.

    Units: q in MW/m^2, T in keV.  Result in particles / (m^2 * s).

    The floor on T prevents divide-by-zero in bins where the profile has
    decayed to essentially nothing; the resulting Gamma is huge there, but
    those bins are typically masked out by the collapse min_count anyway.
    """
    T_safe = np.maximum(T_keV, T_floor_keV)
    return q_MW_per_m2 / (gamma_sheath * T_safe * QE * 1e-3)


def cumulative_charge_flux(R, Gamma_i, Gamma_e, *, Z_i=1):
    """
    Compute cumulative species fluxes and their charge difference:

        F_i(R) = integral_{R_min}^{R} Gamma_i(R') dR'
        F_e(R) = same, for electrons
        F_Q(R) = Z_i * F_i(R) - F_e(R)

    Note we integrate over dR, not dA = 2*pi*R*dR — so these numbers are
    proportional to real particle fluxes but not literally in particles/s.
    The SHAPE of F_Q(R) is the diagnostic; the endpoint tells you global
    ambipolarity.

    If you need absolute-normalized totals, multiply Gamma_i and Gamma_e
    by (2*pi*R) before passing them in.
    """
    R = np.asarray(R, dtype=float)
    Gi = np.asarray(Gamma_i, dtype=float)
    Ge = np.asarray(Gamma_e, dtype=float)

    Gi_z = np.where(np.isfinite(Gi), Gi, 0.0)
    Ge_z = np.where(np.isfinite(Ge), Ge, 0.0)

    dR = np.diff(R, prepend=R[0])
    F_i = np.cumsum(Gi_z * dR)
    F_e = np.cumsum(Ge_z * dR)
    F_Q = Z_i * F_i - F_e

    ratio = float((Z_i * F_i[-1]) / F_e[-1]) if F_e[-1] != 0 else np.nan

    return {
        "R": R,
        "F_i": F_i,
        "F_e": F_e,
        "F_Q": F_Q,
        "F_i_total": float(F_i[-1]),
        "F_e_total": float(F_e[-1]),
        "F_Q_total": float(F_Q[-1]),
        "global_ambipolarity_ratio": ratio,
    }
