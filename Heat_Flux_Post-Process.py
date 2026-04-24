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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple, Optional

from scipy.interpolate import interp1d

# ----------------------------- Physical constants -----------------------------
QE = 1.602176634e-19   # Coulomb; also J/eV
MP = 1.67262192369e-27 # kg

# ----------------------------- Data containers --------------------------------
@dataclass
class Footprint:
    phi: np.ndarray         # (Nphi, Nt)
    s_wall: np.ndarray      # (Nphi, Nt)  wall coordinate (your "length t")
    psi_min: np.ndarray     # (Nphi, Nt)
    R: np.ndarray           # (Nphi, Nt)  [m]
    Z: np.ndarray           # (Nphi, Nt)  [m]
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
    phi2 = phi.reshape(Nphi, Nt, order="C")
    s2   = s_wall.reshape(Nphi, Nt, order="C")
    psi2 = psi_min.reshape(Nphi, Nt, order="C")
    R2   = R.reshape(Nphi, Nt, order="C")
    Z2   = Z.reshape(Nphi, Nt, order="C")

    return Footprint(phi=phi2, s_wall=s2, psi_min=psi2, R=R2, Z=Z2, meta=meta)

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
def maxwellian_energy_pdf(E_keV: np.ndarray, T_keV: np.ndarray) -> np.ndarray:
    """
    Maxwellian energy PDF in 3D (energy form), valid in any consistent energy units:
        p(E;T) = 2/sqrt(pi) * sqrt(E) / T^(3/2) * exp(-E/T)

    Here E and T are in keV.
    """
    E_keV = np.asarray(E_keV)
    T_keV = np.asarray(T_keV)

    T_pos = np.maximum(T_keV, 1e-30)
    return (2.0 / np.sqrt(np.pi)) * np.sqrt(np.maximum(E_keV, 0.0)) * np.exp(-E_keV / T_pos) / (T_pos ** 1.5)

# ----------------------------- (v) Projection utilities -----------------------
def compute_surface_normal(R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    User-specified divertor normal approximation:
        n_hat = (-Z, R, 0) / sqrt(R^2 + Z^2)

    Returns array (..., 3)
    """
    nx = -Z
    ny = R
    nz = np.zeros_like(R)

    norm = np.sqrt(nx**2 + ny**2)
    norm = np.where(norm == 0.0, 1.0, norm)

    return np.stack((nx / norm, ny / norm, nz), axis=-1)

def compute_b_unit(BR: np.ndarray, Bphi: np.ndarray, BZ: np.ndarray) -> np.ndarray:
    """
    Unit magnetic field vector B_hat = B/|B|, array (..., 3)
    """
    Bmag = np.sqrt(BR**2 + Bphi**2 + BZ**2)
    Bmag = np.where(Bmag == 0.0, 1.0, Bmag)
    return np.stack((BR / Bmag, Bphi / Bmag, BZ / Bmag), axis=-1)

def project_parallel_to_perpendicular(
    q_parallel: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    BR: np.ndarray,
    Bphi: np.ndarray,
    BZ: np.ndarray,
    *,
    clip_negative: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Eq. (14):
        q_perp = q_parallel * (n_hat · B_hat)

    Returns:
        q_perp, cos_incidence
    """
    n_hat = compute_surface_normal(R, Z)
    B_hat = compute_b_unit(BR, Bphi, BZ)

    cos_inc = np.sum(n_hat * B_hat, axis=-1)  # (Nphi, Nt)

    if clip_negative:
        cos_inc = np.clip(cos_inc, 0.0, None)

    q_perp = q_parallel * cos_inc
    return q_perp, cos_inc

# ----------------------------- (iv) Sum energies to get q_parallel ------------
def compute_q_parallel(
    footprints_by_energy_keV: Dict[float, Footprint],
    profiles: Profiles,
    *,
    ion_mass_kg: float,
    energies_keV: Optional[np.ndarray] = None,
    use_extra_1_over_N: bool = True,
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
        q += pref * 0.5 * n_i * v_th * (E_keV * pE) * dE_keV[k]

    if use_extra_1_over_N:
        q *= (1.0 / N)

    return phi_grid, s_grid, R_grid, Z_grid, q

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
    # ---- 1) Provide profiles vs psi (YOU REPLACE THESE ARRAYS) ----
    # psi_prof must match normalization of psi_min in the footprint files (often psi_N in [0,1]).
    psi_prof = np.linspace(0.0, 1.0, 200)

    # placeholders (replace): n [m^-3], Ti [keV]
    n_prof = 2e19 * (1 - psi_prof) + 1e18
    Ti_prof_keV = 0.2 * (1 - psi_prof) + 0.02  # 200 eV -> 0.2 keV etc.

    profiles = make_profile_interpolants_keV(psi_prof, n_prof, Ti_prof_keV)

    # ---- 2) Footprint files for each energy (EDIT THIS) ----
    # Example: files = [("footprints/footprint_E100keV.dat", 100.0), ...]
    files = [
        # ("path/to/footprint_E10keV.dat", 10.0),
        # ("path/to/footprint_E20keV.dat", 20.0),
    ]
    if len(files) == 0:
        raise SystemExit("Edit `files = [...]` in main() to point to your footprint files and energies in keV.")

    footprints_by_energy: Dict[float, Footprint] = {}
    for fpath, E_keV in files:
        footprints_by_energy[float(E_keV)] = read_footprint_file(fpath)

    # ---- 3) Ion mass (choose species) ----
    # Deuterium:
    mi = 2.0 * MP

    # ---- 4) Compute q_parallel(phi, s_wall) ----
    phi, s_wall, R_div, Z_div, q_parallel = compute_q_parallel(
        footprints_by_energy,
        profiles,
        ion_mass_kg=mi,
        use_extra_1_over_N=True,
    )

    # ---- 5) Provide B-field components at divertor impact points (EDIT THIS) ----
    # You said to assume we have access to BR, Bphi, BZ evaluated at the incident divertor location.
    # These MUST be shaped (Nphi, Nt) matching q_parallel / R_div / Z_div.
    #
    # Replace placeholders below with your actual arrays.
    BR_div   = np.zeros_like(q_parallel)
    Bphi_div = np.ones_like(q_parallel) * 2.0  # Tesla, placeholder
    BZ_div   = np.ones_like(q_parallel) * 1.0  # Tesla, placeholder

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
    
    # ---- 8) Collapse to q(R) ----
    # Default: continuous toroidal symmetry approximation (average over all phi)
    R_cent, qR, counts = collapse_footprint_to_q_of_R(
        phi, R_div, q_perp,
        average_over_all_phi=True,   # DEFAULT behavior you asked for
        R_bins=400,
        statistic="mean",
    )
    
    # Example: discrete toroidal sampling
    # phi0 = 60 deg, periodicity=3 -> uses ~phi=60, 180, 300 (nearest grid points), averages them
    R_cent_60, qR_60, counts_60 = collapse_footprint_to_q_of_R(
        phi, R_div, q_perp,
        average_over_all_phi=False,
        phi0_deg=60.0,
        periodicity=3,
        R_bins=400,
        statistic="mean",
    )


    # ---- 7) Save results ----
    out = Path("heat_flux_footprints.npz")
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
