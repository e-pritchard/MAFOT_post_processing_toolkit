"""
Plotting for the pipeline.

Every function follows the same convention:
  * Takes the numbers as arguments (not dicts from datafile parsers)
  * Saves the figure if both filetag and file_path are given
  * Never calls plt.show() — the batch pipeline can't wait for windows

If you want interactive display when running by hand, call plt.show()
explicitly after main() returns, or run in Jupyter with %matplotlib inline.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe for batch use
import matplotlib.pyplot as plt

from .constants import LC_MIN_KM_DEFAULT, QE
from .datatypes import Footprint, Profiles
from .diagnostics.penetration import (
    penetration_histogram_data,
    penetration_by_energy_data,
    penetration_by_phi_data,
    penetration_weighted_by_heat_flux_data,
    lc_vs_psimin_data,
)


PathLike = Union[str, Path, None]


def _save(filetag: Optional[str], file_path: PathLike, suffix: str, **savefig):
    """Helper: save if both filetag and file_path are given, else no-op."""
    if filetag and file_path:
        out = Path(file_path) / f"{filetag}{suffix}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", **savefig)


# ------------------------------------------------------------------
# 2D footprint views
# ------------------------------------------------------------------
def toroidal_slicing(q_perp, phi, Z_div, s_wall, phi_slice=60,
                     *, filetag=None, file_path=None):
    """One phi-slice of q_perp shown vs Z and vs s_wall side by side."""
    phi_1d = phi[:, 0]
    phi_idx = int(np.argmin(np.abs(np.rad2deg(phi_1d) - phi_slice)))
    actual_phi_deg = np.rad2deg(phi_1d[phi_idx])
    print(f"Requested phi = {phi_slice} deg, closest = {actual_phi_deg:.2f} deg")

    Z_slice = Z_div[phi_idx, :]
    q_slice = q_perp[phi_idx, :]
    s_slice = s_wall[phi_idx, :]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(Z_slice, q_slice, "o-", linewidth=2, markersize=4)
    ax1.set_xlabel("Z [m]")
    ax1.set_ylabel(r"$q_\perp$ [MW/m$^2$]")
    ax1.set_title(f"Heat flux vs Z at phi = {actual_phi_deg:.1f} deg")
    ax1.grid(True, alpha=0.3)

    ax2.plot(s_slice, q_slice, "o-", linewidth=2, markersize=4, color="red")
    ax2.set_xlabel("s_wall [m]")
    ax2.set_ylabel(r"$q_\perp$ [MW/m$^2$]")
    ax2.set_title(f"Heat flux vs s_wall at phi = {actual_phi_deg:.1f} deg")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(filetag, file_path, "_toroidal_slicing")
    plt.close(fig)


def heat_flux_map(q_perp, *, filetag=None, file_path=None,
                  extent=(0, 360, 0.5, -0.2)):
    """2D heatmap of q_perp(phi, s_wall)."""
    fig = plt.figure(figsize=[8, 8])
    plt.imshow(q_perp.T, extent=extent, interpolation="nearest", aspect="auto")
    plt.xlabel("phi")
    plt.ylabel("s_wall")
    plt.colorbar(label="q_perp")
    plt.tight_layout()
    _save(filetag, file_path, "_heat_flux_profile")
    plt.close(fig)


# ------------------------------------------------------------------
# Penetration diagnostics
# ------------------------------------------------------------------
def plot_penetration_histogram(
    footprints_by_energy, Lc_min_km=LC_MIN_KM_DEFAULT, nbins=50,
    *, filetag=None, file_path=None,
):
    centers, counts, all_psi = penetration_histogram_data(
        footprints_by_energy, Lc_min_km, nbins
    )
    print(f"Mean psi_min: {np.mean(all_psi):.4f}")
    print(f"Std  psi_min: {np.std(all_psi):.4f}")
    print(f"Median:       {np.median(all_psi):.4f}")

    fig = plt.figure()
    plt.bar(centers, counts, width=centers[1] - centers[0], alpha=0.7)
    plt.xlabel("psi_min")
    plt.ylabel("probability density")
    plt.axvline(1.0, color="red", linestyle="--", label="LCFS")
    plt.axvline(np.mean(all_psi), color="black", linestyle=":",
                label=f"mean = {np.mean(all_psi):.3f}")
    plt.legend()
    _save(filetag, file_path, "_pen_histogram")
    plt.close(fig)
    return centers, counts, all_psi


def plot_penetration_by_energy(
    footprints_by_energy, Lc_min_km=LC_MIN_KM_DEFAULT,
    *, filetag=None, file_path=None,
):
    energies, means, stds = penetration_by_energy_data(
        footprints_by_energy, Lc_min_km
    )
    fig = plt.figure()
    plt.errorbar(energies, means, yerr=stds, marker="o")
    plt.xlabel("E_kin [keV]")
    plt.ylabel("mean psi_min +/- std")
    plt.axhline(1.0, color="red", linestyle="--", label="LCFS")
    _save(filetag, file_path, "_pen_by_energy")
    plt.close(fig)
    return energies, means, stds


def plot_penetration_by_phi(
    footprints_by_energy, Lc_min_km=LC_MIN_KM_DEFAULT,
    *, filetag=None, file_path=None,
):
    phi_1d, means, stds = penetration_by_phi_data(
        footprints_by_energy, Lc_min_km
    )
    fig = plt.figure()
    plt.plot(phi_1d, means, "-")
    plt.fill_between(phi_1d, means - stds, means + stds, alpha=0.3)
    plt.xlabel("phi [deg]")
    plt.ylabel("mean psi_min +/- std")
    plt.axhline(1.0, color="red", linestyle="--")
    _save(filetag, file_path, "_pen_by_phi")
    plt.close(fig)
    return phi_1d, means, stds


def plot_penetration_weighted_by_heat_flux(
    footprints_by_energy, profiles, profiles_2, ion_mass_kg,
    Lc_min_km=LC_MIN_KM_DEFAULT, nbins=100,
    *, filetag=None, file_path=None,
):
    all_psi, all_w, mean_unw, mean_w = penetration_weighted_by_heat_flux_data(
        footprints_by_energy, profiles, profiles_2, ion_mass_kg, Lc_min_km
    )
    fig = plt.figure()
    plt.hist(all_psi, bins=nbins, weights=all_w, density=True, alpha=0.7,
             label="heat-flux weighted")
    plt.hist(all_psi, bins=nbins, density=True, alpha=0.3, label="unweighted")
    plt.xlabel("psi_min")
    plt.ylabel("probability density")
    plt.axvline(1.0, color="red", linestyle="--", label="LCFS")
    plt.legend()
    _save(filetag, file_path, "_weighted_heat_flux")
    plt.close(fig)
    print(f"Unweighted mean psi_min: {mean_unw:.4f}")
    print(f"Heat-flux weighted mean: {mean_w:.4f}")


def plot_lc_vs_psimin(
    footprints_by_energy, lc_min_lim=LC_MIN_KM_DEFAULT,
    *, filetag=None, file_path=None,
):
    all_psi, all_lc = lc_vs_psimin_data(footprints_by_energy)
    fig = plt.figure(figsize=(8, 6))
    plt.hexbin(all_psi, np.log10(np.maximum(all_lc, 1e-6)),
               gridsize=80, cmap="viridis", mincnt=1)
    plt.colorbar(label="count")
    plt.axhline(np.log10(lc_min_lim), color="red", linestyle="--",
                label=f"Lc = {lc_min_lim}")
    plt.axvline(1.0, color="white", linestyle="--", label="LCFS")
    plt.xlabel("psi_min")
    plt.ylabel("log10(Lc [km])")
    plt.legend()
    plt.title("Lc vs psi_min joint distribution")
    plt.tight_layout()
    _save(filetag, file_path, "_lc_vs_psimin")
    plt.close(fig)


# ------------------------------------------------------------------
# Ambipolar three-panel diagnostic
# ------------------------------------------------------------------
def plot_ambipolar_diagnostic(
    ion_result, electron_result, cum_result,
    *,
    Z_i=1,
    filetag=None, file_path=None,
):
    """
    Three-panel:
      top:    q_||(R)  ion, electron, total
      middle: Gamma(R) ion (x Z_i), electron  — should overlay when ambipolar
      bottom: F_Q(R)   cumulative charge flux with zero reference
    """
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    ax_q, ax_G, ax_F = axes

    ax_q.plot(ion_result["R"], ion_result["q_mean"], label="ions")
    ax_q.plot(electron_result["R"], electron_result["q_mean"], label="electrons")
    ax_q.plot(
        ion_result["R"],
        ion_result["q_mean"] + electron_result["q_mean"],
        "k--", alpha=0.7, label="total",
    )
    ax_q.set_ylabel(r"$q_\parallel$ [MW/m$^2$]")
    ax_q.legend()
    ax_q.grid(alpha=0.3)
    ax_q.set_title("Toroidally-averaged heat flux vs R")

    ax_G.plot(ion_result["R"], Z_i * ion_result["Gamma"],
              label=fr"$Z_i \Gamma_i$ ($Z_i={Z_i}$)")
    ax_G.plot(electron_result["R"], electron_result["Gamma"],
              label=r"$\Gamma_e$")
    ax_G.set_ylabel(r"$\Gamma$ [particles/m$^2$/s]")
    ax_G.set_yscale("symlog", linthresh=1e18)
    ax_G.legend()
    ax_G.grid(alpha=0.3)
    ax_G.set_title(
        r"Particle flux by species (overlay $\Leftrightarrow$ local ambipolarity)"
    )

    ax_F.plot(cum_result["R"], cum_result["F_Q"], "C3-", linewidth=1.5)
    ax_F.axhline(0, color="k", linestyle=":", alpha=0.5)
    ax_F.set_xlabel("R [m]")
    ax_F.set_ylabel(r"$F_Q(R)$ [arb]")
    ax_F.grid(alpha=0.3)
    ax_F.set_title(
        r"Cumulative charge flux $F_Q = Z_i F_i - F_e$"
        f"   (endpoint ratio $Z_i F_i/F_e$ = "
        f"{cum_result['global_ambipolarity_ratio']:.3f})"
    )

    plt.tight_layout()
    _save(filetag, file_path, "_ambipolar_diagnostic")
    plt.close(fig)
