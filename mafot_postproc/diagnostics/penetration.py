"""
Penetration diagnostics: how deep do orbits reach into the plasma?

All functions take a dict {E [keV]: Footprint} and produce arrays.  Plot
routines live in plotting.py — these return the numbers.
"""

import numpy as np
from typing import Dict, Tuple

from ..constants import QE, LC_MIN_KM_DEFAULT
from ..datatypes import Footprint, Profiles


def penetration_histogram_data(
    footprints_by_energy: Dict[float, Footprint],
    Lc_min_km: float = LC_MIN_KM_DEFAULT,
    nbins: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    All psi_min values from every energy, masked by connection length.

    Returns (bin_centers, densities, all_psi).
    """
    all_psi = []
    for _, fp in footprints_by_energy.items():
        mask = fp.Lc > Lc_min_km
        all_psi.extend(fp.psi_min[mask].ravel())
    all_psi = np.asarray(all_psi)

    counts, edges = np.histogram(all_psi, bins=nbins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts, all_psi


def penetration_by_energy_data(
    footprints_by_energy: Dict[float, Footprint],
    Lc_min_km: float = LC_MIN_KM_DEFAULT,
):
    """Mean and std of psi_min per energy.  Returns (energies, means, stds)."""
    energies = sorted(footprints_by_energy.keys())
    means, stds = [], []
    for E in energies:
        fp = footprints_by_energy[E]
        mask = fp.Lc > Lc_min_km
        psi = fp.psi_min[mask]
        means.append(np.mean(psi) if psi.size else np.nan)
        stds.append(np.std(psi) if psi.size else np.nan)
    return energies, means, stds


def penetration_by_phi_data(
    footprints_by_energy: Dict[float, Footprint],
    Lc_min_km: float = LC_MIN_KM_DEFAULT,
):
    """
    Mean psi_min as a function of phi, averaged over all energies.

    Returns (phi_1d_deg, means, stds).
    """
    fp0 = next(iter(footprints_by_energy.values()))
    Nphi = fp0.phi.shape[0]
    psi_by_phi = [[] for _ in range(Nphi)]
    for _, fp in footprints_by_energy.items():
        mask = fp.Lc > Lc_min_km
        for i in range(Nphi):
            row_mask = mask[i, :]
            psi_by_phi[i].extend(fp.psi_min[i, row_mask])

    phi_1d = np.rad2deg(fp0.phi[:, 0])
    means = np.array([np.mean(p) if p else np.nan for p in psi_by_phi])
    stds = np.array([np.std(p) if p else np.nan for p in psi_by_phi])
    return phi_1d, means, stds


def penetration_weighted_by_heat_flux_data(
    footprints_by_energy: Dict[float, Footprint],
    profiles: Profiles,
    profiles_2: Profiles,
    ion_mass_kg: float,
    Lc_min_km: float = LC_MIN_KM_DEFAULT,
):
    """
    psi_min distribution weighted by each orbit's heat-flux contribution.
    Tells you which depths are actually doing the work.

    Returns (all_psi, all_weights, mean_unweighted, mean_weighted).
    """
    all_psi, all_weights = [], []
    energies = sorted(footprints_by_energy.keys())
    dE = np.gradient(np.array(energies))

    for k, E in enumerate(energies):
        fp = footprints_by_energy[E]
        mask = fp.Lc > Lc_min_km
        psi = fp.psi_min[mask]
        n = profiles.n_i(psi)
        T = profiles.T_i_keV(psi)
        T2 = profiles_2.T_i_keV(psi)
        c_s = np.sqrt(
            1e3 * QE * (np.maximum(T, 0) + np.maximum(T2, 0)) / ion_mass_kg
        )
        pE = np.maximum(E, 0) * np.exp(
            -E / np.maximum(T, 1e-30)
        ) / np.maximum(T, 1e-30) ** 2
        w = 0.5 * n * c_s * E * pE * dE[k]
        all_psi.extend(psi)
        all_weights.extend(w)

    all_psi = np.array(all_psi)
    all_weights = np.array(all_weights)
    mean_unw = float(np.mean(all_psi))
    mean_w = float(np.average(all_psi, weights=all_weights))
    return all_psi, all_weights, mean_unw, mean_w


def lc_vs_psimin_data(
    footprints_by_energy: Dict[float, Footprint],
):
    """
    Joint distribution of Lc and psi_min across all orbits.
    Returns (all_psi, all_lc).
    """
    all_psi, all_lc = [], []
    for _, fp in footprints_by_energy.items():
        mask = fp.Lc > 0
        all_psi.extend(fp.psi_min[mask].ravel())
        all_lc.extend(fp.Lc[mask].ravel())
    return np.asarray(all_psi), np.asarray(all_lc)
