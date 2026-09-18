"""
Core physics: compute q_parallel(phi, s_wall) by summing energy bins with
a Maxwellian weight and a pluggable sheath model.

This module is intentionally small.  All the physics choices live in
physics/sheath.py; this file is the numerical machinery that iterates over
energies, applies masks, and assembles the master B-field / psi_min grids.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from .constants import QE, LC_MIN_KM_DEFAULT
from .datatypes import Footprint, Profiles
from .physics import maxwellian_energy_pdf
from .physics.sheath import SheathModel


def compute_q_parallel(
    footprints_by_energy_keV: Dict[float, Footprint],
    profiles: Profiles,
    *,
    sheath_model: SheathModel,
    ion_mass_kg: float,
    profile_2: Optional[Profiles] = None,
    energies_keV: Optional[np.ndarray] = None,
    use_extra_1_over_N: bool = False,
    Lc_min_km: float = LC_MIN_KM_DEFAULT,
) -> Dict[str, np.ndarray]:
    """
    Compute parallel heat flux integrated over the Maxwellian energy spectrum.

    Parameters
    ----------
    footprints_by_energy_keV : dict {E [keV]: Footprint}
        MAFOT footprint files loaded by io.read_footprint_file, keyed by
        the kinetic energy each was run at.
    profiles : Profiles
        Interpolants for this species (n_i, T_i vs psi).
    sheath_model : SheathModel
        Which sheath physics to use.  See physics/sheath.py.
    ion_mass_kg : float
        Species mass in kg (deuterium: 2 * MP).
    profile_2 : Profiles, optional
        Other-species profiles.  Only used by sheath models that need
        both (e.g. AmbipolarSheath uses T_i + T_e for c_s).
    energies_keV : array, optional
        Override which energies to include.  Defaults to sorted keys of
        footprints_by_energy_keV.
    use_extra_1_over_N : bool
        Whether to include the extra 1/N prefactor from Wingen Eq. 13.
    Lc_min_km : float
        Connection-length threshold for masking.  Set to 0 to disable.

    Returns
    -------
    dict with keys:
        phi, s_wall, R, Z  - grids (Nphi, Nt)
        q_parallel         - MW/m^2 (Nphi, Nt)
        BR, BZ, Bphi       - master B-field grids (Nphi, Nt)
        psi_min            - master psi_min grid (Nphi, Nt)
    """
    # --- setup: energies and grid ---
    if energies_keV is None:
        energies_keV = np.array(sorted(footprints_by_energy_keV.keys()),
                                dtype=float)
    else:
        energies_keV = np.asarray(energies_keV, dtype=float)

    fp0 = footprints_by_energy_keV[float(energies_keV[0])]
    Nphi, Nt = fp0.phi.shape

    q = np.zeros((Nphi, Nt), dtype=float)
    BR_master = np.zeros((Nphi, Nt), dtype=float)
    BZ_master = np.zeros((Nphi, Nt), dtype=float)
    Bphi_master = np.zeros((Nphi, Nt), dtype=float)
    psi_min_master = np.full((Nphi, Nt), np.nan)
    mask_B_found = np.zeros((Nphi, Nt), dtype=bool)

    dE_keV = np.gradient(energies_keV)
    N = len(energies_keV)
    pref = (1.0 / N) if use_extra_1_over_N else 1.0

    # --- iterate over energies ---
    for k, E_keV in enumerate(energies_keV):
        fp = footprints_by_energy_keV[float(E_keV)]
        if fp.psi_min.shape != (Nphi, Nt):
            raise ValueError(
                f"Grid mismatch for E={E_keV} keV: got {fp.psi_min.shape}, "
                f"expected {(Nphi, Nt)}"
            )

        # Penetration mask
        if Lc_min_km > 0:
            mask_pen = fp.Lc > Lc_min_km
        else:
            mask_pen = np.ones_like(fp.psi_min, dtype=bool)

        # Profiles at each grid point's psi_min
        psi_min = fp.psi_min
        n_i = profiles.n_i(psi_min)
        T_i_keV = profiles.T_i_keV(psi_min)
        T2_keV = profile_2.T_i_keV(psi_min) if profile_2 is not None else None

        # Maxwellian weight
        pE = maxwellian_energy_pdf(E_keV, np.maximum(T_i_keV, 1e-30))

        # This is the pluggable physics call — everything above is grid setup
        contrib = sheath_model.q_contribution(
            n_i=n_i,
            T_i_keV=T_i_keV,
            T2_keV=T2_keV,
            E_keV=E_keV,
            pE=pE,
            dE=dE_keV[k],
            ion_mass_kg=ion_mass_kg,
            prefactor=pref,
        )

        contrib = np.where(mask_pen, contrib, 0.0)
        q += contrib

        # Master grids: fill from the first footprint (usually lowest energy)
        # that reached each bin.  Later energies don't overwrite.
        has_B = mask_pen & (
            (np.abs(fp.BR) > 1e-30)
            | (np.abs(fp.BZ) > 1e-30)
            | (np.abs(fp.Bphi) > 1e-30)
        )
        new = has_B & ~mask_B_found
        BR_master[new] = fp.BR[new]
        BZ_master[new] = fp.BZ[new]
        Bphi_master[new] = fp.Bphi[new]
        psi_min_master[new] = fp.psi_min[new]
        mask_B_found |= has_B

    # Unit conversion: [keV / m^2 / s] -> [MW/m^2]
    #   1 keV = 1e3 * QE Joules
    #   1 W = 1 J/s
    #   1 MW = 1e6 W
    q_MW = q * 1e-3 * QE

    return {
        "phi": fp0.phi,
        "s_wall": fp0.s_wall,
        "R": fp0.R,
        "Z": fp0.Z,
        "q_parallel": q_MW,
        "BR": BR_master,
        "BZ": BZ_master,
        "Bphi": Bphi_master,
        "psi_min": psi_min_master,
    }
