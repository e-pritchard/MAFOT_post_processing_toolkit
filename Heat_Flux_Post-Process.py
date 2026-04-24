 #!/usr/bin/env python3
"""
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


Outputs
-------
- Saves an .npz with phi, s_wall, q_parallel, q_perp, cos_incidence.
"""

import re
import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple, Optional

from scipy.interpolate import interp1d

# ----------------------------- Physical constants -----------------------------
QE = 1.602176634e-19   # Coulomb; also J/eV
MP = 1.67262192369e-27 # kg

# In Wingen 2021 sec 4.3: lower limit of connection length within lobes for DIII-D
# corresponds to "at least one full poloidal turn"
LC_MIN_KM_DEFAULT = 0.075 #in km

# ----------------------------- Data containers --------------------------------
@dataclass
class Footprint:
    phi: np.ndarray         # (Nphi, Nt)
    s_wall: np.ndarray      # (Nphi, Nt)  wall coordinate (your "length t")
    psi_min: np.ndarray     # (Nphi, Nt)
    R: np.ndarray           # (Nphi, Nt)  [m]
    Z: np.ndarray           # (Nphi, Nt)  [m]
    #New additions!
    BR: np.ndarray          # (Nphi, Nt)  [T]     
    BZ: np.ndarray          # (Nphi, Nt)  [T]  
    Bphi: np.ndarray        # (Nphi, Nt)  [T]
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
    psi_min = data[:, 4]
    R = data[:, 5]
    Z = data[:, 6]
    #New Additions!
    Lc_psimin = data[:, 7] #added for later masking within the q_parallel function!
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

    # Previous Shaping created a 300x600 matrix, accidently assigning phi coordinates (600) to 
    # the rows. This gives a proper shaping of a 600x300 matrix with phi varying by row 
    phi2 = phi.reshape(Nt, Nphi, order="C").T
    s2   = s_wall.reshape(Nt, Nphi, order="C").T
    psi2 = psi_min.reshape(Nt, Nphi, order="C").T
    R2   = R.reshape(Nt, Nphi, order="C").T
    Z2   = Z.reshape(Nt, Nphi, order="C").T
    Lc_psimin2 = Lc_psimin.reshape(Nt, Nphi, order="C").T
    BR2  = BR.reshape(Nt, Nphi, order="C").T
    BZ2  = BZ.reshape(Nt, Nphi, order="C").T
    Bphi2 = Bphi.reshape(Nt, Nphi, order="C").T

    return Footprint(phi=phi2, s_wall=s2, psi_min=psi2, R=R2, Z=Z2, Lc_psimin=Lc_psimin2, BR=BR2, BZ=BZ2, Bphi=Bphi2, meta=meta)

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

# ----------------------------- Maxwellian weighting ---------------------------
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

#previous maxwellian form
# def maxwellian_energy_pdf(E_keV: np.ndarray, T_keV: np.ndarray) -> np.ndarray:
#     """
#     Maxwellian energy PDF in 3D (energy form), valid in any consistent energy units:
#         p(E;T) = 2/sqrt(pi) * sqrt(E) / T^(3/2) * exp(-E/T)

#     Here E and T are in keV.
#     """
#     E_keV = np.asarray(E_keV)
#     T_keV = np.asarray(T_keV)

#     T_pos = np.maximum(T_keV, 1e-30)
#     return (2.0 / np.sqrt(np.pi)) * np.sqrt(np.maximum(E_keV, 0.0)) * np.exp(-E_keV / T_pos) / (T_pos ** 1.5)

# ----------------------------- (v) Projection utilities -----------------------
def compute_surface_normal(R, Z,axis_s=-1, outward_sign=+1):
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

# Previous surface normal calculation; I think it was switching to polar coordinates
# but not entirely sure --> above keeps it in cylindrical
# def compute_surface_normal(R: np.ndarray, Z: np.ndarray) -> np.ndarray:
#     """
#     User-specified divertor normal approximation:
#         n_hat = (-Z, R, 0) / sqrt(R^2 + Z^2)

#     Returns array (..., 3)
#     """
#     nx = -Z
#     ny = R
#     nz = np.zeros_like(R)

#     norm = np.sqrt(nx**2 + ny**2)
#     norm = np.where(norm == 0.0, 1.0, norm)

#     return np.stack((nx / norm, ny / norm, nz), axis=-1)

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
    q_parallel: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    BR: np.ndarray,
    Bphi: np.ndarray,
    BZ: np.ndarray,
    *,
    clip_negative: bool = False,
    outward_sign=+1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Eq. (14):
        q_perp = q_parallel * (n_hat · B_hat)

    Returns:
        q_perp, cos_incidence
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
    ion_mass_kg: float,
    energies_keV: Optional[np.ndarray] = None,
    use_extra_1_over_N: bool = True,
    Lc_min_km: float = LC_MIN_KM_DEFAULT,
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
    # Will fill these in from footprints that data!
    BR_master = np.zeros((Nphi, Nt), dtype=float)
    BZ_master = np.zeros((Nphi, Nt), dtype=float)
    Bphi_master = np.zeros((Nphi, Nt), dtype=float)

    # Track where we have valid B-field data
    mask_B_found = np.zeros((Nphi, Nt), dtype=bool)


    N = len(energies_keV)
    pref = (1.0 / N) if use_extra_1_over_N else 1.0

    for k, E_keV in enumerate(energies_keV):
        fp = footprints_by_energy_keV[float(E_keV)]
        if fp.psi_min.shape != (Nphi, Nt):
            raise ValueError(f"Grid mismatch for E={E_keV} keV: got {fp.psi_min.shape}, expected {(Nphi, Nt)}")

        psi_min = fp.psi_min

        # (iii) Apply profiles at psi_min
        n_i = profiles.n_i(psi_min)          # density
        Ti_keV = profiles.T_i_keV(psi_min)   # keV

        # thermal speed: v_th = sqrt(k_B T / m) with T in keV => k_B T = (T_keV*1e3 eV)*e
        v_th = np.sqrt(1e3 * QE * np.maximum(Ti_keV, 0.0) / ion_mass_kg)

        # Maxwellian energy PDF p(E, T) in keV
        pE = maxwellian_energy_pdf(E_keV, np.maximum(Ti_keV, 1e-30))

        # (iv) Sum contributions (Eq. 13)
        contrib = pref * 0.5 * n_i * v_th * (E_keV * pE) * dE_keV[k]
        contrib = np.where(mask_pen, contrib, 0.0)
        q += contrib

        # ---- master B-field (only fill where this orbit actually reached) ----
        has_B = mask_pen & ((np.abs(fp.BR)   > 1e-30) | (np.abs(fp.BZ) > 1e-30) | (np.abs(fp.Bphi) > 1e-30))
        new = has_B & ~mask_B_found
        BR_master[new]   = fp.BR[new]
        BZ_master[new]   = fp.BZ[new]
        Bphi_master[new] = fp.Bphi[new]
        mask_B_found |= has_B

    if use_extra_1_over_N:
        q *= (1.0 / N)


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

# ----------------------------- Example driver --------------------------------
def main():

    CODE_PATH = os.path.dirname(os.path.abspath(__file__))
    # ---- 1) Provide profiles vs psi

    #PROFILES FOR IONS
    #-----------------------------------------------------------------------------------------------------------
    ni_loaded = np.loadtxt("Density_Temperature_Profiles/Extended_Profiles/171491_ni_extended.dat")
    Ti_loaded = np.loadtxt("Density_Temperature_Profiles/Extended_Profiles/171491_ti_extended.dat")
    
    psi_prof = ni_loaded[:,0]
    n_prof = ni_loaded[:,1]
    Ti_prof_keV = Ti_loaded[:,1]


    #PROFILES FOR ELECTRONS
    #-----------------------------------------------------------------------------------------------------------
    # ne_loaded = np.loadtxt("Density_Temperature_Profiles/Calculated_Profiles/171491_ne_extended.dat")
    # Te_loaded = np.loadtxt("Density_Temperature_Profiles/Calculated_Profiles/171491_te_extended.dat")
    
    # psi_prof = ne_loaded[:,0]
    # n_prof = ne_loaded[:,1] * 10**(20) #array is in units of (10^20 / m^3)
    # T_prof_keV = Te_loaded[:,1]


    profiles = make_profile_interpolants_keV(psi_prof, n_prof, Ti_prof_keV)

    # ---- 2) Footprint files for each energy ----

    files = []

    ## AXISYMMETRIC ION FOOTPRINTS
    #-------------------------------------------------------------------------------------------------------
    # for file in os.scandir("MAFOT_Footprints/axisymmetric_ion_Footprints/"):
    #     if "foot_in_copass" in file.name:
    #         name_split = file.name.split("copassE")
    #         name_split = name_split[1].split("shot")
    #         files.append((file.path, name_split[0]))

    # AXISYMMETRIC ELECTRON FOOTPRINTS
    #-------------------------------------------------------------------------------------------------------
    # for file in os.scandir("MAFOT_Footprints/axisymmetric_electron_footprints"):
    #     if "foot_in_axisymmetricElectrons" in file.name:
    #         name_split = file.name.split("Electrons")
    #         name_split = name_split[1].split("shot")
    #         files.append((file.path, name_split[0]))

     # 3d Fields ION FOOTPRINTS
    #-------------------------------------------------------------------------------------------------------
    # for file in os.scandir("MAFOT_Footprints/3d_Fields_ion_footprints/"):
    #     if "foot_in_3dFieldsCopass" in file.name:
    #         name_split = file.name.split("CopassE")
    #         name_split = name_split[1].split("shot")
    #         files.append((file.path, name_split[0]))

    # 3d Fields ELECTRON FOOTPRINTS
    #-------------------------------------------------------------------------------------------------------
    for file in os.scandir("MAFOT_Footprints/3d_Fields_electron_footprints/"):
        if "foot_in_3dFieldsElectrons" in file.name:
            name_split = file.name.split("ElectronsE")
            name_split = name_split[1].split("shot")
            files.append((file.path, name_split[0]))
            
    if len(files) == 0:
        raise SystemExit("Edit `files = [...]` in main() to point to your footprint files and energies in keV.")

    footprints_by_energy: Dict[float, Footprint] = {}
    for fpath, E_keV in files:
        footprints_by_energy[float(E_keV)] = read_footprint_file(fpath)

    # ---- 3) Ion mass (choose species) ----
    # Deuterium:
    mi = 2.0 * MP
    me = 9.1093837 * 10**-31

    # ---- 4) Compute q_parallel(phi, s_wall) ----
    phi, s_wall, R_div, Z_div, q_parallel = compute_q_parallel(
        footprints_by_energy,
        profiles,
        ion_mass_kg=mi,
        # ion_mass_kg=me, #i
        use_extra_1_over_N=True,
    )



    # ---- 5) This step used to be reading in B-fields but B-fields are now read in and with footprints and full B-field matrix is filled when computing q_parallel



    # ---- 6) (v) Project onto divertor normal: q_perp = q_parallel * (n_hat · B_hat) ----
    q_perp, cos_inc = project_parallel_to_perpendicular(
        q_parallel,
        R_div,
        Z_div,
        BR_div,
        Bphi_div,
        BZ_div,
        clip_negative=False,  # set True if you want to zero-out negative incidence
    )
    

    # ================= Still Fiddling with these =================

    # ---- 8) Collapse to q(R) ---- 
    # Default: continuous toroidal symmetry approximation (average over all phi)
    # R_cent, qR, counts = collapse_footprint_to_q_of_R(
    #     phi, R_div, q_perp,
    #     average_over_all_phi=True,   
    #     R_bins=400,
    #     statistic="mean",
    # ) #Still working on this
    
    # Example: discrete toroidal sampling
    # phi0 = 60 deg, periodicity=3 -> uses ~phi=60, 180, 300 (nearest grid points), averages them
    # R_cent_60, qR_60, counts_60 = collapse_footprint_to_q_of_R(
    #     phi, R_div, q_perp,
    #     average_over_all_phi=False,
    #     phi0_deg=60.0,
    #     periodicity=3,
    #     R_bins=400,
    #     statistic="mean",
    # ) #

    # ====================================================================


    # Wingen et. al. (2021) does a 2d smoothing of his data:
    # Done here to match his workflow:
    window_size = 5  # adjust as needed (5 seems good for the most part)
    q_perp = uniform_filter(q_perp, size=window_size, mode='constant')



    # ---- Extracting and ploting a single toroidal slice ----
    # (For comparison with Figures 12 and 14)

    # Choose a toroidal angle (in degrees)
    phi_target_deg = 60.0
    
    # Convert grid to 1D phi (assumes constant along s_wall)
    phi_1d = phi[:, 0]  # shape (Nphi,)
    
    # Find the closest phi index to your target
    phi_idx = int(np.argmin(np.abs(np.rad2deg(phi_1d) - phi_target_deg)))
    actual_phi_deg = np.rad2deg(phi_1d[phi_idx])
    
    print(f"Requested φ = {phi_target_deg}°, closest available φ = {actual_phi_deg:.2f}°")
    
    # Extract the slice
    Z_slice = Z_div[phi_idx, :]
    q_perp_slice = q_perp[phi_idx, :]
    s_wall_slice = s_wall[phi_idx, :]
    
    # Plot: heat flux vs Z
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: q_perp vs Z
    ax1.plot(Z_slice, q_perp_slice, 'o-', linewidth=2, markersize=4)
    ax1.set_xlabel('Z [m]', fontsize=12)
    ax1.set_ylabel('q⊥ [MW/m²]', fontsize=12)
    ax1.set_title(f'Heat Flux vs Z at φ = {actual_phi_deg:.1f}°')
    ax1.grid(True, alpha=0.3)
    
    # q_perp vs s_wall (for comparison)
    ax2.plot(s_wall_slice, q_perp_slice, 'o-', linewidth=2, markersize=4, color='red')
    ax2.set_xlabel('s_wall [m]', fontsize=12)
    ax2.set_ylabel('q⊥ [MW/m²]', fontsize=12)
    ax2.set_title(f'Heat Flux vs s_wall at φ = {actual_phi_deg:.1f}°')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{CODE_PATH}/post_processing_figures/3d_Fields_ions_Smoothed(window=5)_Toroidal_Slicing.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Slice shape: Z={Z_slice.shape}, q_perp={q_perp_slice.shape}")
    print(f"Z range: [{Z_slice.min():.4f}, {Z_slice.max():.4f}] m")
    print(f"Peak heat flux: {q_perp_slice.max():.2f} MW/m²")


    # -------- Heatmap of q_perp (Figure 6 of wingen) ------------
    plt.figure(figsize=[8,8])
    plt.imshow(q_perp.T, 
               extent=[0, 360, 0.5, -0.2], 
               interpolation="nearest", aspect="auto")
    plt.xlabel("phi")
    plt.ylabel("S_wall")
    plt.colorbar(label="q_perp")
    plt.tight_layout()
    plt.savefig(f'{CODE_PATH}/post_processing_figures/3d_Fields_ions_Smoothed(window=5)__heat_flux_map.png', dpi=150, bbox_inches='tight')
    plt.show()



    # ---- 7) Save results ----
    out = Path(f"{CODE_PATH}/post_processing_npzfiles/3d_Fields_ions_Smoothed(window=5)_heat_flux_footprints.npz")
    np.savez(
        out,
        phi=phi,
        s_wall=s_wall,
        R=R_div,
        Z=Z_div,
        q_parallel=q_parallel,
        q_perp=q_perp,
        cos_incidence=cos_inc,
    )
    print(f"Saved: {out.resolve()}")
    print("Shapes:",
          "phi", phi.shape,
          "s_wall", s_wall.shape,
          "q_parallel", q_parallel.shape,
          "q_perp", q_perp.shape)

if __name__ == "__main__":
    main()
