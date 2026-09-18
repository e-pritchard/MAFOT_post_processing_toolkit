"""
Collapse 2D footprints q(phi, s_wall) into 1D profiles q(R), and
periodicity-fold n-lobe footprints (e.g. I-coil n=3) onto a single period.

The two operations compose: for I-coil cases, fold first, then collapse.
For axisymmetric baseline, just collapse.
"""

import numpy as np
from typing import Optional, Tuple

from ..datatypes import Profiles


# ------------------------------------------------------------------
# n-fold periodicity fold
# ------------------------------------------------------------------
def fold_footprint_by_periodicity(
    phi_grid, R_grid, Z_grid, q_grid,
    n_period,
    *,
    phi0_deg=0.0,
    require_exact=True,
    return_std=True,
):
    """
    Fold a 2D footprint q(phi, s_wall) by n-fold toroidal symmetry.

    For an I-coil n=3 RMP case, three identical lobes appear on the wall,
    one per 120-degree period.  This function folds them onto a single
    period and averages:

        q_folded(phi', s) = (1/n) * sum_{k=0}^{n-1} q(phi' + k*360/n, s)

    Returns a dict with folded phi/R/Z/q_mean of shape (Nphi//n, Nt), plus
    q_std across the n folds — a symmetry-breaking diagnostic (near zero
    when the assumption of identical lobes holds).

    Set n_period=1 for a no-op that just returns the input.
    """
    if n_period < 1:
        raise ValueError(f"n_period must be >= 1, got {n_period}")

    if n_period == 1:
        out = {
            "phi": phi_grid, "R": R_grid, "Z": Z_grid,
            "q_mean": q_grid,
            "n_period": 1, "phi0_deg": phi0_deg,
        }
        if return_std:
            out["q_std"] = np.zeros_like(q_grid)
        return out

    Nphi, Nt = phi_grid.shape

    if Nphi % n_period != 0:
        if require_exact:
            raise ValueError(
                f"Nphi={Nphi} is not divisible by n_period={n_period}. "
                f"Either regenerate footprints with a phi-grid that is a "
                f"multiple of {n_period}, or pass require_exact=False to "
                f"interpolate onto a compatible grid."
            )
        Nphi_new = (Nphi // n_period) * n_period
        phi_1d = phi_grid[:, 0]
        phi_new = np.linspace(phi_1d[0], phi_1d[-1], Nphi_new)

        def _interp_along_phi(A):
            out = np.empty((Nphi_new, Nt))
            for j in range(Nt):
                out[:, j] = np.interp(phi_new, phi_1d, A[:, j])
            return out

        phi_grid = phi_new[:, None] * np.ones((1, Nt))
        R_grid = _interp_along_phi(R_grid)
        Z_grid = _interp_along_phi(Z_grid)
        q_grid = _interp_along_phi(q_grid)
        Nphi = Nphi_new

    Nphi_per_period = Nphi // n_period

    # Optional roll so the window starts at phi0_deg
    if phi0_deg != 0.0:
        phi_1d = phi_grid[:, 0]
        phi0_rad = np.deg2rad(phi0_deg % 360.0)
        i0 = int(np.argmin(np.abs(phi_1d - phi0_rad)))
        phi_grid = np.roll(phi_grid, -i0, axis=0)
        R_grid = np.roll(R_grid, -i0, axis=0)
        Z_grid = np.roll(Z_grid, -i0, axis=0)
        q_grid = np.roll(q_grid, -i0, axis=0)

    q_stack = q_grid.reshape(n_period, Nphi_per_period, Nt)
    R_stack = R_grid.reshape(n_period, Nphi_per_period, Nt)
    Z_stack = Z_grid.reshape(n_period, Nphi_per_period, Nt)
    phi_stack = phi_grid.reshape(n_period, Nphi_per_period, Nt)

    out = {
        "phi": phi_stack[0],
        "R": R_stack.mean(axis=0),
        "Z": Z_stack.mean(axis=0),
        "q_mean": q_stack.mean(axis=0),
        "n_period": n_period,
        "phi0_deg": phi0_deg,
    }
    if return_std:
        out["q_std"] = q_stack.std(axis=0)
    return out


# ------------------------------------------------------------------
# Collapse to q(R)
# ------------------------------------------------------------------
def collapse_footprint_to_q_of_R(
    phi_grid, R_grid, q_grid,
    *,
    R_range: Optional[Tuple[float, float]] = None,
    R_bins: int = 200,
    area_weight: bool = False,
    min_count: int = 1,
) -> dict:
    """
    Bin q(phi, s_wall) into q(R) via toroidal averaging.

    Returns a dict with R, q_mean, q_std, counts, edges.  q_std is the
    std across all samples in each R bin — combines toroidal variation
    and wall-position variation.

    Works on either raw or fold-reduced footprints; the math doesn't care.
    """
    if phi_grid.shape != R_grid.shape or phi_grid.shape != q_grid.shape:
        raise ValueError(
            f"Shape mismatch: phi={phi_grid.shape}, R={R_grid.shape}, "
            f"q={q_grid.shape}"
        )

    R_flat = R_grid.ravel()
    q_flat = q_grid.ravel()
    good = np.isfinite(R_flat) & np.isfinite(q_flat)
    R_flat, q_flat = R_flat[good], q_flat[good]
    if R_flat.size == 0:
        raise ValueError("No finite (R, q) points to bin.")

    if R_range is None:
        Rmin, Rmax = float(R_flat.min()), float(R_flat.max())
    else:
        Rmin, Rmax = float(R_range[0]), float(R_range[1])
    if Rmax <= Rmin:
        raise ValueError(f"Invalid R_range: ({Rmin}, {Rmax})")

    edges = np.linspace(Rmin, Rmax, R_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weights = R_flat if area_weight else np.ones_like(R_flat)

    sum_qw, _ = np.histogram(R_flat, bins=edges, weights=q_flat * weights)
    sum_w, _ = np.histogram(R_flat, bins=edges, weights=weights)
    counts, _ = np.histogram(R_flat, bins=edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        q_mean = np.where(sum_w > 0, sum_qw / sum_w, np.nan)
    sum_q2w, _ = np.histogram(R_flat, bins=edges,
                              weights=(q_flat ** 2) * weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        q_mean_sq = np.where(sum_w > 0, sum_q2w / sum_w, np.nan)
        var = q_mean_sq - q_mean ** 2
        var = np.where(var < 0, 0.0, var)
        q_std = np.sqrt(var)

    sparse = counts < min_count
    q_mean = np.where(sparse, np.nan, q_mean)
    q_std = np.where(sparse, np.nan, q_std)

    return {
        "R": centers,
        "q_mean": q_mean,
        "q_std": q_std,
        "counts": counts.astype(int),
        "edges": edges,
    }


def collapse_species_to_q_and_T_of_R(
    phi_grid, R_grid, q_grid, psi_min_grid, profiles: Profiles,
    *,
    R_range=None,
    R_bins=200,
    min_count=1,
):
    """
    Extended collapse that also returns T_mean(R) and n_mean(R), needed to
    back-derive particle flux Gamma(R) for the ambipolar diagnostic.

    Returns a dict with R, q_mean, q_std, T_mean_keV, n_mean, counts, edges.
    """
    for arr, name in [(phi_grid, "phi"), (R_grid, "R"),
                      (q_grid, "q"), (psi_min_grid, "psi_min")]:
        if arr.shape != phi_grid.shape:
            raise ValueError(
                f"{name}_grid shape {arr.shape} != phi_grid shape "
                f"{phi_grid.shape}"
            )

    T_grid = profiles.T_i_keV(psi_min_grid)
    n_grid = profiles.n_i(psi_min_grid)

    R_flat = R_grid.ravel()
    q_flat = q_grid.ravel()
    T_flat = np.asarray(T_grid).ravel()
    n_flat = np.asarray(n_grid).ravel()

    good = (np.isfinite(R_flat) & np.isfinite(q_flat)
            & np.isfinite(T_flat) & np.isfinite(n_flat))
    R_flat = R_flat[good]
    q_flat = q_flat[good]
    T_flat = T_flat[good]
    n_flat = n_flat[good]

    if R_flat.size == 0:
        raise ValueError("No finite (R, q, T, n) points to bin.")

    if R_range is None:
        Rmin, Rmax = float(R_flat.min()), float(R_flat.max())
    else:
        Rmin, Rmax = float(R_range[0]), float(R_range[1])
    if Rmax <= Rmin:
        raise ValueError(f"Invalid R_range: ({Rmin}, {Rmax})")

    edges = np.linspace(Rmin, Rmax, R_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    counts, _ = np.histogram(R_flat, bins=edges)
    sum_q, _ = np.histogram(R_flat, bins=edges, weights=q_flat)
    sum_q2, _ = np.histogram(R_flat, bins=edges, weights=q_flat ** 2)
    sum_T, _ = np.histogram(R_flat, bins=edges, weights=T_flat)
    sum_n, _ = np.histogram(R_flat, bins=edges, weights=n_flat)

    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(counts > 0, counts, 1)
        q_mean = np.where(counts > 0, sum_q / c, np.nan)
        T_mean = np.where(counts > 0, sum_T / c, np.nan)
        n_mean = np.where(counts > 0, sum_n / c, np.nan)
        var = np.where(counts > 0, sum_q2 / c - q_mean ** 2, 0.0)
        var = np.where(var < 0, 0.0, var)
        q_std = np.sqrt(var)

    sparse = counts < min_count
    q_mean = np.where(sparse, np.nan, q_mean)
    q_std = np.where(sparse, np.nan, q_std)
    T_mean = np.where(sparse, np.nan, T_mean)
    n_mean = np.where(sparse, np.nan, n_mean)

    return {
        "R": centers,
        "q_mean": q_mean,
        "q_std": q_std,
        "T_mean_keV": T_mean,
        "n_mean": n_mean,
        "counts": counts.astype(int),
        "edges": edges,
    }
