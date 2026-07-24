#!/usr/bin/env python3
"""

Last Updated: 7/3/2026

Compute divertor heat-flux footprints following the workflow in the screenshots:

(ii)  trace ion orbits -> provides psi_min(phi, s_wall) for each sampled energy
(iii) apply profiles   -> map psi_min -> n_i(psi_min), T_i(psi_min), v_th
(iv)  sum contributions over energies weighted by Maxwellian p(E, T_i) to get q_parallel
(v)   project onto divertor surface normal:
        q_perp = q_parallel * (n_hat · B_hat)

Assumptions / conventions
-------------------------
- One footprint file per kinetic energy Ekin (in keV).
- T_i(psi) profile is in keV.
- Maxwellian weighting is done in keV consistently:
    p(E;T) = 2/sqrt(pi) * sqrt(E) / T^(3/2) * exp(-E/T)   (E,T in keV)
- Thermal speed uses Joules only in k_B T = (T_keV * 1e3 eV) * e:
    v_th = sqrt( (T_keV*1e3*e) / m_i )
- Projection normal (user-specified):
    n_hat = (-Z, R, 0) / sqrt(R^2 + Z^2)
- Magnetic field components available at divertor impact point:
    B = (B_R, B_phi, B_Z)

Notes on Eq. (13) normalization
-------------------------------
Eq (13) in the screenshot shows factors (1/N) outside and (1/N) inside the sum.
This script reproduces that if `use_extra_1_over_N=True`.
If you'd rather do a plain Riemann-sum approximation of the energy integral, set it False.

Outputs
-------
- Saves an .npz with phi, s_wall, q_parallel, q_perp, cos_incidence.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple, Optional
import argparse
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter

# ----------------------------- Physical constants -----------------------------
QE = 1.602176634e-19   # Coulomb; also J/eV
MP = 1.67262192369e-27 # kg

ELECTRON_TRANSMISSION_COEFF = 7
ION_TRANSMISSION_COEFF = 2.5
LC_MIN_KM_DEFAULT = 0.075
DEFAULT_FILE_TAG = "unnamed_pipeline"

# ----------------------------- Data containers --------------------------------
@dataclass
class Footprint:
    phi: np.ndarray         # (Nphi, Nt)
    s_wall: np.ndarray      # (Nphi, Nt)  wall coordinate (your "length t")
    psi_min: np.ndarray     # (Nphi, Nt)
    R: np.ndarray           # (Nphi, Nt)  [m]
    Z: np.ndarray           # (Nphi, Nt)  [m]
    BR: np.ndarray          # (Nphi, Nt)  [T]     
    BZ: np.ndarray          # (Nphi, Nt)  [T]  
    Bphi: np.ndarray        # (Nphi, Nt)  [T]
    Lc: np.ndarray          # (Nphi, Nt)  [km]
    Lc_psimin: np.ndarray   # (Nphi, Nt)  [km]
    
    meta: Dict[str, float]  # parsed header parameters (best effort)

@dataclass
class Profiles:
    n_i: Callable[[np.ndarray], np.ndarray]   # density at psi
    T_i_keV: Callable[[np.ndarray], np.ndarray]  # Ti at psi in keV

# ----------------------------- Parsing footprint files ------------------------
def _try_parse_float(line: str) -> Optional[Tuple[str, float]]:
    # matches: "# key: value"
    m = re.match(r"^\s*#\s*([^:]+?)\s*:\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)\s*$", line)
    if not m:
        return None
    key = m.group(1).strip()
    val = float(m.group(2))
    return key, val

def read_footprint_file(path: str | Path) -> Footprint:
    """
    Reads one footprint file (one energy) like your snippet.

    Expected data columns:
    phi[rad], length t, N_toroidal, connection length [km], psimin, R[m], Z[m], Lc_at_psimin [km]
    """
    path = Path(path)
    meta: Dict[str, float] = {}

    data_lines = []
    with path.open("r") as f:
        for line in f:
            if line.lstrip().startswith("#"):
                kv = _try_parse_float(line)
                if kv:
                    meta[kv[0]] = kv[1]
            else:
                if line.strip():
                    data_lines.append(line)

    data = np.loadtxt(data_lines)
    if data.ndim == 1:
        data = data[None, :]

    # Columns
    phi = data[:, 0]
    s_wall = data[:, 1]
    Lc = data[:,3]
    psi_min = data[:, 4]
    R = data[:, 5]
    Z = data[:, 6]
    Lc_psimin = data[:, 7] 
    BR = data[:, 8]  
    BZ = data[:, 9]  
    Bphi = data[:, 10]

    # Determine grid sizes: prefer header
    Nphi = int(meta.get("phi-grid", 0))
    Nt = int(meta.get("t-grid", 0))

    if not (Nphi > 0 and Nt > 0 and data.shape[0] == Nphi * Nt):
        # fallback: infer from unique values
        phi_u = np.unique(phi)
        s_u = np.unique(s_wall)
        Nphi, Nt = len(phi_u), len(s_u)
        if Nphi * Nt != data.shape[0]:
            raise ValueError(
                f"Cannot reshape data: got {data.shape[0]} rows, "
                f"but inferred Nphi={Nphi}, Nt={Nt} (product {Nphi*Nt})."
            )

    # Reshape assumption: file ordered with phi varying fastest for each s_wall
    phi2 = phi.reshape(Nphi, Nt, order="F")
    s2   = s_wall.reshape(Nphi, Nt, order="F")
    psi2 = psi_min.reshape(Nphi, Nt, order="F")
    R2   = R.reshape(Nphi, Nt, order="F")
    Z2   = Z.reshape(Nphi, Nt, order="F")
    Lc2 = Lc.reshape(Nphi, Nt, order="F")
    Lc_psimin2 = Lc_psimin.reshape(Nphi, Nt,order="F")
    BR2 = BR.reshape(Nphi, Nt, order="F")
    BZ2 = BZ.reshape(Nphi, Nt, order="F")
    Bphi2 = Bphi.reshape(Nphi, Nt, order="F")


    return Footprint(phi=phi2, s_wall=s2, Lc=Lc2, psi_min=psi2, R=R2, Z=Z2, Lc_psimin=Lc_psimin2, BR=BR2, BZ=BZ2, Bphi=Bphi2, meta=meta)

# ----------------------------- Profiles vs psi --------------------------------
def make_profile_interpolants_keV(
    psi_prof: np.ndarray,
    n_prof: np.ndarray,
    Ti_prof_keV: np.ndarray,
    *,
    kind: str = "linear",
    fill: str = "extrapolate",
) -> Profiles:
    """
    Build interpolants n_i(psi), T_i(psi) with T in keV.
    """
    psi_prof = np.asarray(psi_prof)
    n_prof = np.asarray(n_prof)
    Ti_prof_keV = np.asarray(Ti_prof_keV)

    n_itp = interp1d(psi_prof, n_prof, kind=kind, bounds_error=False, fill_value=fill)
    T_itp = interp1d(psi_prof, Ti_prof_keV, kind=kind, bounds_error=False, fill_value=fill)

    return Profiles(
        n_i=lambda x: np.asarray(n_itp(x)),
        T_i_keV=lambda x: np.asarray(T_itp(x)),
    )


#Wingen's exact maxwellian pdf form
def maxwellian_energy_pdf(E_keV, T_keV):
    """
    Wingen (2021) Eq. (12): p(E,T) = E / T^2 * exp(-E/T)
    E and T must be in consistent units (keV and keV here).
    """
    E = np.asarray(E_keV, dtype=float)
    T = np.asarray(T_keV, dtype=float)
    T_pos = np.maximum(T, 1e-30)
    return np.maximum(E, 0.0) * np.exp(-E / T_pos) / (T_pos ** 2)


def compute_surface_normal(R, Z, axis_s=-1, outward_sign=+1):
    # tangent along s_wall (assumed last axis)
    dRds = np.gradient(R, axis=-1)
    dZds = np.gradient(Z, axis=-1)
    tmag = np.sqrt(dRds**2 + dZds**2)
    tmag_safe = np.where(tmag > 1e-15, tmag, 1.0)
    tR = dRds / tmag_safe
    tZ = dZds / tmag_safe
    # 90-deg rotation of tangent gives poloidal-plane normal
    nR = -tZ         # points to +R for a vertical wall (into plasma)
    nphi = np.zeros_like(nR)
    nZ = tR
    return np.stack((nR, nphi, nZ), axis=-1)

def compute_b_unit(BR: np.ndarray, Bphi: np.ndarray, BZ: np.ndarray) -> np.ndarray:
    """
    Unit magnetic field vector B_hat = B/|B|, array (..., 3)
    """
    Bmag = np.sqrt(BR**2 + Bphi**2 + BZ**2)
    
    # Create mask for non-zero B-field
    mask_good_B = (Bmag > 1e-30)

    B_hat = np.zeros(BR.shape + (3,))
    
    # Use where to safely handle zero division
    safe_Bmag = np.where(mask_good_B, Bmag, 1.0)
    
    B_hat[..., 0] = BR / safe_Bmag
    B_hat[..., 1] = Bphi / safe_Bmag
    B_hat[..., 2] = BZ / safe_Bmag
    
    # Zero out bad points (where we used safe_Bmag=1)
    B_hat[~mask_good_B] = 0.0
    
    return B_hat



def project_parallel_to_perpendicular(
    q_parallel,
    R, Z,
    BR, Bphi, BZ,
    *,
    clip_negative=True,
    outward_sign=+1,
):
    """
    q_perp = q_parallel * |n_hat . B_hat|, using a PROPER poloidal wall normal.
 
    Note: we take the absolute value of the incidence cosine because the sign
    just tells us which side of the wall the field is on; the heat flux
    magnitude depends on the angle, not the sign. (You can also handle this by
    choosing `outward_sign` correctly and then using `clip_negative=True` so
    that points where co-passing particles would not strike the wall are set
    to zero.)
    """
    n_hat = compute_surface_normal(R, Z, axis_s=-1, outward_sign=outward_sign)
 
    Bmag = np.sqrt(BR**2 + Bphi**2 + BZ**2)
    mask_good = Bmag > 1e-12
    Bmag_safe = np.where(mask_good, Bmag, 1.0)
    b_hat = np.stack((BR / Bmag_safe, Bphi / Bmag_safe, BZ / Bmag_safe), axis=-1)
    # Zero out bad points
    b_hat = np.where(mask_good[..., None], b_hat, 0.0)
 
    cos_inc = np.sum(n_hat * b_hat, axis=-1)
 
    if clip_negative:
        # Co-passing ions reach the wall only on the side where n.B has a
        # specific sign. With our convention (n_R > 0 pointing into plasma),
        # field lines that strike this wall section have n.B < 0 (the particle
        # moves toward the wall, opposite to the outward normal). So the
        # intensity is |n.B|; alternatively take max(-cos_inc, 0).
        cos_inc_for_flux = np.maximum(-cos_inc, 0.0)
    else:
        cos_inc_for_flux = np.abs(cos_inc)
 
    q_perp = q_parallel * cos_inc_for_flux
    return q_perp, cos_inc

# ----------------------------- (iv) Sum energies to get q_parallel ------------
def compute_q_parallel(
    footprints_by_energy_keV: Dict[float, Footprint],
    profiles: Profiles,
    *,
    profile_2 = None,
    ion_mass_kg: float,
    energies_keV: Optional[np.ndarray] = None,
    use_extra_1_over_N: bool = False,
    sheath: bool = True,
    Lc_min_km: float = LC_MIN_KM_DEFAULT,
    transmission_coeff = 2.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      phi_grid  (Nphi, Nt)
      s_wall    (Nphi, Nt)
      R_grid    (Nphi, Nt)
      Z_grid    (Nphi, Nt)
      q_parallel(Nphi, Nt)
    """
    if energies_keV is None:
        energies_keV = np.array(sorted(footprints_by_energy_keV.keys()), dtype=float)
    else:
        energies_keV = np.asarray(energies_keV, dtype=float)

    fp0 = footprints_by_energy_keV[float(energies_keV[0])]
    phi_grid = fp0.phi
    s_grid = fp0.s_wall
    R_grid = fp0.R
    Z_grid = fp0.Z

    Nphi, Nt = phi_grid.shape

    # energy bin widths ΔE (in keV)
    dE_keV = np.gradient(energies_keV)


    q = np.zeros((Nphi, Nt), dtype=float)

    # Master B-field grids
    # Initialize to zeros - will fill in from footprints that have particles
    BR_master = np.zeros((Nphi, Nt), dtype=float)
    BZ_master = np.zeros((Nphi, Nt), dtype=float)
    Bphi_master = np.zeros((Nphi, Nt), dtype=float)

    # Track where we have valid B-field data
    mask_B_found = np.zeros((Nphi, Nt), dtype=bool)

    N = len(energies_keV)

    if use_extra_1_over_N:
        pref = (1.0 / N) #if use_extra_1_over_N else 1.0
    else:
        pref = 1.0


    for k, E_keV in enumerate(energies_keV):
        fp = footprints_by_energy_keV[float(E_keV)]
        if fp.psi_min.shape != (Nphi, Nt):
            raise ValueError(f"Grid mismatch for E={E_keV} keV: got {fp.psi_min.shape}, expected {(Nphi, Nt)}")

        # ---- per-energy penetration mask ----
        if Lc_min_km > 0:
            mask_pen = fp.Lc > Lc_min_km
        else:
            mask_pen = np.ones_like(fp.psi_min, dtype=bool)
        
        psi_min = fp.psi_min


        # (iii) Apply profiles at psi_min
        n_i = profiles.n_i(psi_min)          # density
        Ti_keV = profiles.T_i_keV(psi_min)   # keV


        # thermal speed: v_th = sqrt(k_B T / m) with T in keV => k_B T = (T_keV*1e3 eV)*e
        # For ion speeds, thermal velocity is sufficient for calculating heat fluxes

        v_th = np.sqrt(1e3 * QE * np.maximum(Ti_keV, 0.0) / ion_mass_kg)


        # At the divertor plates, electrons acculumate, create a sheath potential, speeding up ions and deceleterating electrons such that
        # electrons and ion fluxes into the plates are equal
        # Therefore the velocity of the electrons driving into the divertor should equal that of the ions (assuming equal densities)
        c_s = np.sqrt(1e3 * QE * 2 * np.maximum(Ti_keV, 0.0) / ion_mass_kg)

        if profile_2:
            T2_keV = profile_2.T_i_keV(psi_min)
            c_s = np.sqrt(1e3 * QE * (np.maximum(Ti_keV, 0.0) + np.maximum(T2_keV, 0)) / ion_mass_kg)


        

        # Maxwellian energy PDF p(E, T) in keV
        pE = maxwellian_energy_pdf(E_keV, np.maximum(Ti_keV, 1e-30))
        # print(f"The probablilty of the {E_keV} energy state is {pE}")

        # (iv) Sum contributions (Eq. 13)
        if sheath:
            contrib = transmission_coeff * pref * 0.5 * n_i * c_s * (E_keV * pE) * dE_keV[k]
        else: 
            contrib = pref * 0.5 * n_i * v_th * (E_keV * pE) * dE_keV[k]

        contrib = np.where(mask_pen, contrib, 0.0)
        q += contrib

        # Fill master B-field from this footprint
        # ---- master B-field (only fill where this orbit actually reached) ----
        has_B = mask_pen & ((np.abs(fp.BR)   > 1e-30) | (np.abs(fp.BZ) > 1e-30) | (np.abs(fp.Bphi) > 1e-30))
        new = has_B & ~mask_B_found
        BR_master[new]   = fp.BR[new]
        BZ_master[new]   = fp.BZ[new]
        Bphi_master[new] = fp.Bphi[new]
        mask_B_found |= has_B
            

    #Technically, q is in units of [q] = keV * m^-2 * s^-1
    #To get MW/m^2, must do: [q] = keV * m^-2 * s^-1 (10^3 * QE) * (10**-6)
    #                                                   [J/keV]     [MW/W] 
    q = q * 10**(-3) * QE 

    return phi_grid, s_grid, R_grid, Z_grid, q, BR_master, BZ_master, Bphi_master


# ----------------------------- Collapse to 1D --------------------------------
def _wrap_deg(angle_deg: float) -> float:
    return float(angle_deg) % 360.0

def _deg2rad(angle_deg: float) -> float:
    return np.deg2rad(_wrap_deg(angle_deg))

def _choose_phi_indices(phi_grid_rad_1d: np.ndarray,
                        phi0_deg: float,
                        periodicity: int) -> np.ndarray:
    """
    Pick indices in the available phi grid closest to the requested angles:
      phi0, phi0+360/p, ..., phi0+(p-1)*360/p
    periodicity p must be >= 1.
    """
    if periodicity < 1:
        raise ValueError("periodicity must be >= 1")

    phi_targets_deg = _wrap_deg(phi0_deg) + (360.0 / periodicity) * np.arange(periodicity)
    phi_targets_rad = np.deg2rad(_wrap_deg(phi_targets_deg))

    # map each target to nearest available phi index
    # assumes phi_grid_rad_1d spans [0, 2pi)
    inds = []
    for ph in phi_targets_rad:
        d = np.angle(np.exp(1j * (phi_grid_rad_1d - ph)))  # wrap-safe diff in [-pi,pi]
        inds.append(int(np.argmin(np.abs(d))))
    return np.array(sorted(set(inds)), dtype=int)


def collapse_footprint_to_q_of_R(
    phi_grid: np.ndarray,     # (Nphi, Nt)
    R_grid: np.ndarray,       # (Nphi, Nt)
    q_grid: np.ndarray,       # (Nphi, Nt) e.g. q_perp or q_parallel
    *,
    phi0_deg: float | None = None,
    periodicity: int = 1,
    average_over_all_phi: bool = True,
    R_bins: int = 400,
    R_range: tuple[float, float] | None = None,
    statistic: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse q(phi, s) to q(R) by selecting toroidal slices and averaging.

    Default behavior (average_over_all_phi=True):
      - continuous toroidal symmetry approximation: use *all* phi rows.

    If average_over_all_phi=False:
      - use phi0_deg and periodicity to choose a subset of phi slices.

    Returns:
      R_centers, q_of_R, counts
    """
    if phi_grid.ndim != 2 or R_grid.ndim != 2 or q_grid.ndim != 2:
        raise ValueError("phi_grid, R_grid, q_grid must all be 2D arrays (Nphi, Nt)")

    Nphi, Nt = phi_grid.shape
    if R_grid.shape != (Nphi, Nt) or q_grid.shape != (Nphi, Nt):
        raise ValueError("R_grid and q_grid must match phi_grid shape")

    # Build 1D phi list (assumes phi constant along s)
    phi_1d = phi_grid[:, 0]

    if average_over_all_phi:
        phi_inds = np.arange(Nphi, dtype=int)
    else:
        if phi0_deg is None:
            raise ValueError("phi0_deg must be provided when average_over_all_phi=False")
        phi_inds = _choose_phi_indices(phi_1d, phi0_deg=phi0_deg, periodicity=periodicity)

    R_sel = R_grid[phi_inds, :].ravel()
    q_sel = q_grid[phi_inds, :].ravel()

    # Optional range
    if R_range is None:
        Rmin, Rmax = np.nanmin(R_sel), np.nanmax(R_sel)
    else:
        Rmin, Rmax = R_range

    # Drop non-finite
    mask = np.isfinite(R_sel) & np.isfinite(q_sel)
    R_sel = R_sel[mask]
    q_sel = q_sel[mask]

    # Bin in R
    edges = np.linspace(Rmin, Rmax, R_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_id = np.digitize(R_sel, edges) - 1

    q_out = np.full(R_bins, np.nan, dtype=float)
    counts = np.zeros(R_bins, dtype=int)

    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")

    for b in range(R_bins):
        m = (bin_id == b)
        if not np.any(m):
            continue
        counts[b] = int(np.sum(m))
        if statistic == "mean":
            q_out[b] = float(np.mean(q_sel[m]))
        else:
            q_out[b] = float(np.median(q_sel[m]))

    return centers, q_out, counts

def toroidal_slicing(q_perp, phi, Z_div, s_wall, phi_slice=60, 
                     *, 
                     filetag=False, file_path=False):
        phi_target_deg = phi_slice
        
        # Convert grid to 1D phi (assumes constant along s_wall)
        phi_1d = phi[:, 0] # WITH TRANSPOSING
 
        
        # Find the closest phi index to your target
        phi_idx = int(np.argmin(np.abs(np.rad2deg(phi_1d) - phi_target_deg)))
        # phi_window = slice(max(0, phi_idx-2), min(600, phi_idx+3))
        # q_perp_slice = np.mean(q_perp[phi_window, :], axis=0)

        actual_phi_deg = np.rad2deg(phi_1d[phi_idx])
        
        print(f"Requested φ = {phi_target_deg}°, closest available φ = {actual_phi_deg:.2f}°")

        Z_slice = Z_div[phi_idx, :]
        q_perp_slice = q_perp[phi_idx, :]
        # q_perp_slice = np.mean(q_perp[phi_window, :], axis=0) # Grabbing short window around 60 toroidal slice
        s_wall_slice = s_wall[phi_idx, :]
        
        # Plot: heat flux vs Z
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: q_perp vs Z
        ax1.plot(Z_slice, q_perp_slice, 'o-', linewidth=2, markersize=4)
        ax1.set_xlabel('Z [m]', fontsize=12)
        ax1.set_ylabel('q⊥ [MW/m²]', fontsize=12)
        ax1.set_title(f'Heat Flux vs Z at φ = {actual_phi_deg:.1f}°')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: q_perp vs s_wall (for comparison)
        ax2.plot(s_wall_slice, q_perp_slice, 'o-', linewidth=2, markersize=4, color='red')
        ax2.set_xlabel('s_wall [m]', fontsize=12)
        ax2.set_ylabel('q⊥ [MW/m²]', fontsize=12)
        ax2.set_title(f'Heat Flux vs s_wall at φ = {actual_phi_deg:.1f}°')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if file_path and filetag:
            plt.savefig(Path(file_path) / f"{filetag}_toroidal_slicing.png", dpi=150, bbox_inches='tight')
        # plt.show()
        
        print(f"Slice shape: Z={Z_slice.shape}, q_perp={q_perp_slice.shape}")
        print(f"Z range: [{Z_slice.min():.4f}, {Z_slice.max():.4f}] m")
        print(f"Peak heat flux: {q_perp_slice.max():.2f} MW/m²")

#My own code for heatmap of q_perp (Figure 6 of wingen)
def heat_flux_map(q_perp,
                  *, 
                filetag=False, file_path=False):
    
    plt.figure(figsize=[8,8])
    plt.imshow(q_perp.T, 
               extent=[0, 360, 0.5, -0.2], 
               interpolation="nearest", aspect="auto")
    # plt.xlim((0, 360))
    plt.xlabel("phi")
    plt.ylabel("S_wall")
    plt.colorbar(label="q_perp")
    plt.tight_layout()
    if file_path and filetag:
        plt.savefig(Path(file_path) / f"{filetag}_heat_flux_profile.png", dpi=150, bbox_inches='tight')
    # if filetag:
    #     plt.savefig(f'{file_path}{filetag}_heat_flux_map.png', dpi=150, bbox_inches='tight')
    # plt.show()


def penetration_histogram(footprints_by_energy, Lc_min_km=LC_MIN_KM_DEFAULT, nbins=50,
                          *, 
                        filetag=False, file_path=False):
    """All psi_min values from all energies, masked by penetration."""
    all_psi = []
    for E, fp in footprints_by_energy.items():
        # mask = fp.Lc_psimin > Lc_min_km
        mask = fp.Lc > Lc_min_km
        all_psi.extend(fp.psi_min[mask].ravel())
    all_psi = np.array(all_psi)
    
    counts, edges = np.histogram(all_psi, bins=nbins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    print(f"Mean psi_min: {np.mean(all_psi):.4f}")
    print(f"Std  psi_min: {np.std(all_psi):.4f}")
    print(f"Median:       {np.median(all_psi):.4f}")
    print(f"5th–95th pct: [{np.percentile(all_psi, 5):.4f}, {np.percentile(all_psi, 95):.4f}]")
    
    plt.figure()
    plt.bar(centers, counts, width=edges[1]-edges[0], alpha=0.7)
    plt.xlabel('psi_min')
    plt.ylabel('probability density')
    plt.axvline(1.0, color='red', linestyle='--', label='LCFS')
    plt.axvline(np.mean(all_psi), color='black', linestyle=':', label=f'mean = {np.mean(all_psi):.3f}')
    plt.legend()
    if file_path and filetag:
        plt.savefig(Path(file_path) / f"{filetag}_toroidal_slicing.png", dpi=150)
    return centers, counts, all_psi

def penetration_by_energy(footprints_by_energy, Lc_min_km=0.075, 
                          *, 
                        filetag=False, file_path=False):
    """Mean and std of psi_min as a function of energy."""
    energies = sorted(footprints_by_energy.keys())
    means, stds = [], []
    for E in energies:
        fp = footprints_by_energy[E]
        # mask = fp.Lc_psimin > Lc_min_km ------------------------
        mask = fp.Lc > Lc_min_km
        psi_E = fp.psi_min[mask]
        means.append(np.mean(psi_E))
        stds.append(np.std(psi_E))
    
    plt.figure()
    plt.errorbar(energies, means, yerr=stds, marker='o')
    plt.xlabel('E_kin [keV]')
    plt.ylabel('mean psi_min ± std')
    plt.axhline(1.0, color='red', linestyle='--', label='LCFS')
    if file_path and filetag:
        plt.savefig(Path(file_path) / f"{filetag}_pen_histogram.png", dpi=150)
    return energies, means, stds

def penetration_by_phi(footprints_by_energy, Lc_min_km=LC_MIN_KM_DEFAULT, 
                        *, 
                     filetag=False, file_path=False):
    """Mean psi_min as a function of phi, averaged over all energies."""
    fp0 = next(iter(footprints_by_energy.values()))
    Nphi = fp0.phi.shape[0]
    
    psi_by_phi = [[] for _ in range(Nphi)]
    for E, fp in footprints_by_energy.items():
        # mask = fp.Lc_psimin > Lc_min_km ===========================================
        mask = fp.Lc > Lc_min_km
        for i in range(Nphi):
            row_mask = mask[i, :]
            psi_by_phi[i].extend(fp.psi_min[i, row_mask])
    
    phi_1d = np.rad2deg(fp0.phi[:, 0])
    means = np.array([np.mean(p) if len(p) > 0 else np.nan for p in psi_by_phi])
    stds  = np.array([np.std(p)  if len(p) > 0 else np.nan for p in psi_by_phi])
    
    plt.figure()
    plt.plot(phi_1d, means, '-')
    plt.fill_between(phi_1d, means-stds, means+stds, alpha=0.3)
    plt.xlabel('phi [deg]')
    plt.ylabel('mean psi_min ± std')
    plt.axhline(1.0, color='red', linestyle='--')
    if file_path and filetag:
        plt.savefig(Path(file_path) / f"{filetag}_pen_by_phi.png", dpi=150)
    return phi_1d, means, stds

def penetration_weighted_by_heat_flux(footprints_by_energy, profiles, profiles_2, ion_mass_kg, Lc_min_km=LC_MIN_KM_DEFAULT, nbins=100,
                                      *, 
                                    filetag=False, file_path=False):
    """psi_min distribution weighted by the heat flux each orbit contributes.
    Tells you which depths are actually doing the work in your heat flux calculation."""
    all_psi = []
    all_weights = []
    energies = sorted(footprints_by_energy.keys())
    dE = np.gradient(np.array(energies))
    
    for k, E in enumerate(energies):
        fp = footprints_by_energy[E]
        # mask = fp.Lc_psimin > Lc_min_km
        mask = fp.Lc > Lc_min_km #======================================================================================
        psi = fp.psi_min[mask]
        n = profiles.n_i(psi)
        T = profiles.T_i_keV(psi)
        T2 = profiles_2.T_i_keV(psi)
        c_s = np.sqrt(1e3 * QE * (np.maximum(T,0) + np.maximum(T2,0)) / ion_mass_kg)
        pE = np.maximum(E, 0) * np.exp(-E / np.maximum(T, 1e-30)) / np.maximum(T, 1e-30)**2
        w = 0.5 * n * c_s * E * pE * dE[k]
        all_psi.extend(psi)
        all_weights.extend(w)
    
    all_psi = np.array(all_psi)
    all_weights = np.array(all_weights)
    
    plt.figure()
    plt.hist(all_psi, bins=nbins, weights=all_weights, density=True, alpha=0.7, label='heat-flux weighted')
    plt.hist(all_psi, bins=nbins, density=True, alpha=0.3, label='unweighted')
    plt.xlabel('psi_min')
    plt.ylabel('probability density')
    plt.axvline(1.0, color='red', linestyle='--', label='LCFS')
    plt.legend()
    if file_path and filetag:
        plt.savefig(Path(file_path) / f"{filetag}_weighted_heat_flux.png", dpi=150)
    
    mean_unw = np.mean(all_psi)
    mean_w   = np.average(all_psi, weights=all_weights)
    print(f"Unweighted mean psi_min: {mean_unw:.4f}")
    print(f"Heat-flux weighted mean: {mean_w:.4f}")

def lc_vs_psimin(footprints_by_energy, lc_min_lim = 0.075,
                *, 
                filetag=False, file_path=False):
    """Show how psi_min correlates with connection length.
    Real lobe orbits should have moderate Lc (~0.1-1 km).
    Suspiciously long Lc with deep psi_min = potential ghost."""
    import matplotlib.pyplot as plt
    all_psi = []
    all_lc = []
    for E, fp in footprints_by_energy.items():
        # mask = fp.Lc_psimin > 0  # all orbits, look at full distribution #======================================================================================
        mask = fp.Lc > 0
        all_psi.extend(fp.psi_min[mask].ravel())
        all_lc.extend(fp.Lc[mask].ravel()) 
        # all_lc.extend(fp.Lc_psimin[mask].ravel())
    all_psi = np.array(all_psi)
    all_lc = np.array(all_lc)
    
    plt.figure(figsize=(8,6))
    plt.hexbin(all_psi, np.log10(np.maximum(all_lc, 1e-6)), 
               gridsize=80, cmap='viridis', mincnt=1)
    plt.colorbar(label='count')
    plt.axhline(np.log10(lc_min_lim), color='red', linestyle='--', label=f'Lc = {lc_min_lim}')
    plt.axvline(1.0, color='white', linestyle='--', label='LCFS')
    plt.xlabel('psi_min')
    plt.ylabel('log10(Lc [km])')
    plt.legend()
    plt.title('Lc vs psi_min joint distribution')
    plt.tight_layout()
    if file_path and filetag:
        plt.savefig(Path(file_path) / f"{filetag}_lc_vs_psimin.png", dpi=150)
    # plt.show()
    
    # Also: print fraction of orbits at each psi_min bin
    print("Fraction of (Lc>75m) orbits in psi_min bins:")
    for lo, hi in [(0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.0), (1.0, 1.05)]:
        m = (all_psi > lo) & (all_psi <= hi) & (all_lc > 0.075)
        print(f"  psi=[{lo},{hi}]: {m.sum()} orbits, mean Lc = {all_lc[m].mean() if m.sum() else float('nan'):.3f} km")


def profile_sanity_check(profiles, profiles_ion,
                        *, 
                     filetag=False, file_path=False):
    import matplotlib.pyplot as plt
    psi = np.linspace(0.7, 1.05, 500)
    ne = profiles.n_i(psi)
    Te = profiles.T_i_keV(psi)
    ni = profiles_ion.n_i(psi)
    Ti = profiles_ion.T_i_keV(psi)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0,0].plot(psi, ne, label='n_e')
    axes[0,0].plot(psi, ni, label='n_i')
    axes[0,0].set_xlabel('psi'); axes[0,0].set_ylabel('density [m^-3]')
    axes[0,0].axvline(1.0, color='red', ls='--'); axes[0,0].legend()
    axes[0,0].set_yscale('log')
    
    axes[0,1].plot(psi, Te, label='T_e')
    axes[0,1].plot(psi, Ti, label='T_i')
    axes[0,1].set_xlabel('psi'); axes[0,1].set_ylabel('T [keV]')
    axes[0,1].axvline(1.0, color='red', ls='--'); axes[0,1].legend()
    
    # Integrand-like quantity at sample energies
    for E in [1, 5, 10, 20]:
        pE = E * np.exp(-E/np.maximum(Ti,1e-6)) / np.maximum(Ti,1e-6)**2
        cs = np.sqrt(1e3*1.602e-19*(Te+Ti)/(2*1.673e-27))
        weight = 0.5 * ni * cs * E * pE
        axes[1,0].plot(psi, weight, label=f'E={E} keV')
    axes[1,0].set_xlabel('psi'); axes[1,0].set_ylabel('ion integrand weight')
    axes[1,0].axvline(1.0, color='red', ls='--'); axes[1,0].legend()
    axes[1,0].set_yscale('log')
    
    axes[1,1].plot(psi, ni*Ti, label='n_i T_i (pressure proxy)')
    axes[1,1].plot(psi, ne*Te, label='n_e T_e')
    axes[1,1].set_xlabel('psi'); axes[1,1].set_ylabel('pressure proxy')
    axes[1,1].axvline(1.0, color='red', ls='--'); axes[1,1].legend()
    
    plt.tight_layout()
    # if filetag and file_path:
    if file_path and filetag:
        plt.savefig(Path(file_path) / f"{filetag}_toroidal_slicing.png", dpi=150)
    # plt.show()



def main(data_dir, file_tag):

    # ---------- Directories + Sub-Dirs ------------------
    DATA_DIR = Path(data_dir)
    SHOT = str(DATA_DIR).split("/")[-1]
    FOOTPRINT_ION_DIR = Path(f"{DATA_DIR}/footprints_ion")
    FOOTPRINT_ELECTRON_DIR = Path(f"{DATA_DIR}/footprints_electrons")

    # Make a directory for post processing products:
    POST_PROCESSING_DIR = Path(f"{DATA_DIR}/post_processing_outputs")
    POST_PROCESSING_DIR.mkdir(exist_ok=True)

    # ---- 1) Provide profiles vs psi (
    # psi_prof must match normalization of psi_min in the footprint files (often psi_N in [0,1]).

    #PROFILES FOR ELECTRONS
    #-----------------------------------------------------------------------------------------------------------
    ne_loaded = np.loadtxt(f"{str(DATA_DIR)}/Profiles/{SHOT}_ne.dat")
    Te_loaded = np.loadtxt(f"{str(DATA_DIR)}/Profiles/{SHOT}_te.dat")
    
    psi_prof = ne_loaded[:,0]
    n_prof = ne_loaded[:,1] * 10**(20) #array is in units of (10^20 / m^3)
    T_prof_keV = Te_loaded[:,1]

    profiles_electron = make_profile_interpolants_keV(psi_prof, n_prof, T_prof_keV)

    #PROFILES FOR IONS
    #-----------------------------------------------------------------------------------------------------------
    ni_loaded = np.loadtxt(f"{str(DATA_DIR)}/Profiles/{SHOT}_ni.dat")
    Ti_loaded = np.loadtxt(f"{str(DATA_DIR)}/Profiles/{SHOT}_ti.dat")

    psi_prof_ions = ni_loaded[:,0]
    n_prof_ions = ni_loaded[:,1] * 10**(20) #array is in units of (10^20 / m^3)
    T_prof_keV_ions= Ti_loaded[:,1]

    profiles_ion = make_profile_interpolants_keV(psi_prof_ions, n_prof_ions, T_prof_keV_ions)


    # ---- 2) Footprint files for each energy -----

    files_electron = []
    files_ion = []
    

    # ION FOOTPRINTS
    #-------------------------------------------------------------------------------------------------------
    for file in FOOTPRINT_ION_DIR.iterdir():
        with open(file) as f:
            lines = f.readlines(1500)
            f.close()

        for line in lines:
            if "Ekin:" in line:
                Ekin = line.split("Ekin: ")[-1].split("\n")[0]
                files_ion.append((file, Ekin))



    if len(files_ion) == 0:
        raise SystemExit("Edit `files = [...]` in main() to point to your footprint files and energies in keV.")

    footprints_by_energy_ion: Dict[float, Footprint] = {}
    for fpath, E_keV in files_ion:
        footprints_by_energy_ion[float(E_keV)] = read_footprint_file(fpath)

    # 3d Fields ELECTRON FOOTPRINTS
    #-------------------------------------------------------------------------------------------------------
    for file in FOOTPRINT_ELECTRON_DIR.iterdir():
        with open(file) as f:
            lines = f.readlines(1500)
            f.close()

        for line in lines:
            if "Ekin:" in line:
                Ekin = line.split("Ekin: ")[-1].split("\n")[0]
                files_electron.append((file, Ekin))
            
    if len(files_electron) == 0:
        raise SystemExit("Edit `files = [...]` in main() to point to your footprint files and energies in keV.")

    footprints_by_energy_electron: Dict[float, Footprint] = {}
    for fpath, E_keV in files_electron:
        footprints_by_energy_electron[float(E_keV)] = read_footprint_file(fpath)

    # ---- 3) Ion mass (choose species) ----
    # Deuterium:
    mi = 2.0 * MP

    me = 9.1093837 * 10**-31 #kg

    # ---- 4_ion) Compute q_parallel(phi, s_wall) ---- FOR IONS!!!!!!
    phi_ion, s_wall_ion, R_div_ion, Z_div_ion, q_parallel_ion, BR_div_ion, BZ_div_ion, Bphi_div_ion = compute_q_parallel(
        footprints_by_energy_ion,
        profiles_ion,
        profile_2 = profiles_electron,
        ion_mass_kg=mi,
        use_extra_1_over_N=False,
        Lc_min_km=LC_MIN_KM_DEFAULT,
        transmission_coeff=ION_TRANSMISSION_COEFF,
        sheath=True
    )

    # ---- 4_electrons) Compute q_parallel(phi, s_wall) ---- FOR ELECTRONS!!!!!!
    phi_electron, s_wall_electron, R_div_electron, Z_div_electron, q_parallel_electron, BR_div_electron, BZ_div_electron, Bphi_div_electron = compute_q_parallel(
        footprints_by_energy_electron,
        profiles_electron,
        profile_2=profiles_ion,
        ion_mass_kg=mi,
        use_extra_1_over_N=False,
        Lc_min_km=LC_MIN_KM_DEFAULT,
        transmission_coeff=ELECTRON_TRANSMISSION_COEFF,
        sheath=True
    )

    

    # ---- 6_ion) (v) Project onto divertor normal: q_perp = q_parallel * (n_hat · B_hat) ----
    q_perp_ion, cos_inc_ion = project_parallel_to_perpendicular(
        q_parallel_ion,
        R_div_ion,
        Z_div_ion,
        BR_div_ion,
        Bphi_div_ion,
        BZ_div_ion,
        clip_negative=False,  
    )
    # ---- 6_electron) (v) Project onto divertor normal: q_perp = q_parallel * (n_hat · B_hat) ----
    q_perp_electron, cos_inc_electron = project_parallel_to_perpendicular(
        q_parallel_electron,
        R_div_electron,
        Z_div_electron,
        BR_div_electron,
        Bphi_div_electron,
        BZ_div_electron,
        clip_negative=False,  
    )

    # Congregate Profiles (Electron + Ion q_perp & q_parallel)

    q_parallel_congre = q_parallel_electron + q_parallel_ion
    q_perp_congre = q_perp_electron + q_perp_ion

    
    #Smoothing the data like wingen
    window_size = 5  # adjust as needed (5 seems good for the most part)
    q_perp_electron = uniform_filter(q_perp_electron, size=window_size, mode='nearest')
    q_perp_ion = uniform_filter(q_perp_ion, size=window_size, mode='nearest')
    q_perp_congre = uniform_filter(q_perp_congre, size=window_size, mode='nearest')

    file_tag = file_tag

    # file_path = "/Users/epritchard/Research/Evan-s_Folders/Plasma Research/Post_Processing/5_15_2026_figures"
    file_path = f"{POST_PROCESSING_DIR}"
    file_tag_electron = file_tag+"_electron"
    file_tag_ion = file_tag+"_ion"
    file_tag_congre = file_tag+"_congre"

    #Electron outputs:
    toroidal_slicing(q_perp_electron, phi_electron, Z_div_electron, s_wall_electron, 60, filetag = file_tag_electron, file_path=file_path)
    heat_flux_map(q_perp_electron, filetag = file_tag_electron, file_path=file_path)
    lc_vs_psimin(footprints_by_energy_electron, lc_min_lim=LC_MIN_KM_DEFAULT, filetag=file_tag_electron, file_path=file_path)

    #Ion outputs:
    toroidal_slicing(q_perp_ion, phi_ion, Z_div_ion, s_wall_ion, 60, filetag = file_tag_ion, file_path=file_path)
    heat_flux_map(q_perp_ion, filetag = file_tag_ion, file_path=file_path)
    lc_vs_psimin(footprints_by_energy_ion, lc_min_lim=LC_MIN_KM_DEFAULT, filetag=file_tag_ion, file_path=file_path)

    # Congre Outputs:
    toroidal_slicing(q_perp_congre, phi_electron, Z_div_electron, s_wall_electron, 60, filetag = file_tag_congre, file_path=file_path)
    heat_flux_map(q_perp_congre, filetag = file_tag_congre, file_path=file_path)


    # ---- 7) Save results ----
    out = Path(f"{POST_PROCESSING_DIR}/{file_tag_electron}_{SHOT}_heat_flux_footprints.npz")
    np.savez(
        out,
        phi=phi_electron,
        s_wall=s_wall_electron,
        R=R_div_electron,
        Z=Z_div_electron,
        q_parallel=q_parallel_electron,
        q_perp=q_perp_electron,
        cos_incidence=cos_inc_electron,
    )

    out = Path(f"{POST_PROCESSING_DIR}/{file_tag_ion}_{SHOT}_heat_flux_footprints.npz")
    np.savez(
        out,
        phi=phi_ion,
        s_wall=s_wall_ion,
        R=R_div_ion,
        Z=Z_div_ion,
        q_parallel=q_parallel_ion,
        q_perp=q_perp_ion,
        cos_incidence=cos_inc_ion,
    )

    out = Path(f"{POST_PROCESSING_DIR}/{file_tag_congre}_{SHOT}_heat_flux_footprints.npz")
    np.savez(
        out,
        phi=phi_electron,
        s_wall=s_wall_electron,
        R=R_div_electron,
        Z=Z_div_electron,
        q_parallel=q_parallel_congre,
        q_perp=q_perp_congre,
        cos_incidence=cos_inc_electron
    )

    print(f"Saved: {out.resolve()}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument(
        "file_tag",
        type=str,
        default=DEFAULT_FILE_TAG,
        help=f"File tag for output naming (default: {DEFAULT_FILE_TAG})",
    )
    args = p.parse_args()
    main(args.data_dir, args.file_tag)
    