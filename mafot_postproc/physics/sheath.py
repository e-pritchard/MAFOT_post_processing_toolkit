"""
Sheath models — the plug-in point for physics variants.

Each concrete model implements a single method:

    q_contribution(...) -> ndarray

which returns the per-energy heat flux contribution *in the same units as
Wingen Eq. 13*, i.e. keV / m^2 / s (unit conversion to MW/m^2 happens once
at the end of compute_q_parallel, not here).

To add a new physics variant (like ambipolarity enforcement), subclass
SheathModel and implement q_contribution.  compute_q_parallel doesn't care
which model you pass in.

The three shipped models correspond to:

    WingenSheath          - the original Wingen 2021 paper, no c_s / gamma
    AmbipolarSheath       - assumes ambipolar sheath (c_s from both species
                            temperatures, transmission coefficient gamma)
                            THIS IS THE CURRENT DEFAULT
    EnforcedAmbipolarSheath - placeholder for future work where ambipolarity
                              is enforced as a closure rather than assumed

All models take the same input dict of per-bin quantities, so switching
between them is a one-line change in compute_q_parallel's call site.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..constants import QE


class SheathModel(ABC):
    """
    Abstract base class for sheath physics variants.

    All state (transmission coefficients, closure parameters, etc.) is
    stored on the instance so that main() can pick a variant and pass it
    to compute_q_parallel without a mess of keyword arguments.
    """

    @abstractmethod
    def q_contribution(
        self,
        *,
        n_i: np.ndarray,        # density at psi_min (Nphi, Nt) [m^-3]
        T_i_keV: np.ndarray,    # this-species temperature (Nphi, Nt) [keV]
        T2_keV: Optional[np.ndarray],  # other-species temperature or None
        E_keV: float,           # this bin's energy [keV]
        pE: np.ndarray,         # Maxwellian(E; T_i) (Nphi, Nt) [1/keV]
        dE: float,              # this bin's energy width [keV]
        ion_mass_kg: float,     # kg
        prefactor: float,       # 1 or 1/N depending on Wingen normalization
    ) -> np.ndarray:
        """
        Return this energy bin's heat-flux contribution on the (Nphi, Nt)
        grid, in units of keV / m^2 / s (pre-conversion to MW/m^2).
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable name, used for logging."""
        return type(self).__name__


# ------------------------------------------------------------------
# WingenSheath — the paper's original formulation
# ------------------------------------------------------------------
@dataclass
class WingenSheath(SheathModel):
    """
    Wingen 2021 Eq. 13 as written: uses the thermal speed of a single
    species, no ambipolar boundary condition, no sheath transmission
    coefficient.
    """

    def q_contribution(self, *, n_i, T_i_keV, T2_keV, E_keV,
                       pE, dE, ion_mass_kg, prefactor):
        # v_th = sqrt(k_B T / m_i)  with T in Joules
        v_th = np.sqrt(1e3 * QE * np.maximum(T_i_keV, 0.0) / ion_mass_kg)
        return prefactor * 0.5 * n_i * v_th * E_keV * pE * dE


# ------------------------------------------------------------------
# AmbipolarSheath — current default (c_s, transmission coefficient)
# ------------------------------------------------------------------
@dataclass
class AmbipolarSheath(SheathModel):
    """
    Assumes ambipolar sheath boundary condition:

    * Ion velocity at the sheath is the Bohm speed c_s = sqrt((T_i + T_e) / m_i).
      This is set by the ambipolar constraint that ion and electron fluxes
      into the wall are equal.

    * Heat flux is multiplied by a sheath transmission coefficient gamma
      that accounts for the sheath potential drop.  Standard values:
        gamma = 2.5 for ions
        gamma = 7.0 for electrons

    Note this is 'assumed' ambipolarity, not 'enforced'.  Nothing here
    guarantees the resulting Gamma_i matches Gamma_e at the wall -- The
    diagnostics.ambipolar module measures the discrepancy.
    """
    transmission_coeff: float

    def q_contribution(self, *, n_i, T_i_keV, T2_keV, E_keV,
                       pE, dE, ion_mass_kg, prefactor):
        if T2_keV is None:
            # Fall back to using only this species' T in c_s.  This is
            # equivalent to setting T2 = T_i.
            T_sum = 2.0 * np.maximum(T_i_keV, 0.0)
        else:
            T_sum = np.maximum(T_i_keV, 0.0) + np.maximum(T2_keV, 0.0)

        c_s = np.sqrt(1e3 * QE * T_sum / ion_mass_kg)

        return (self.transmission_coeff * prefactor
                * 0.5 * n_i * c_s * E_keV * pE * dE)


# ------------------------------------------------------------------
# EnforcedAmbipolarSheath — future work placeholder
# ------------------------------------------------------------------
@dataclass
class EnforcedAmbipolarSheath(SheathModel):
    """
    PLACEHOLDER — not yet implemented.

    """
    transmission_coeff: float

    def q_contribution(self, **kwargs):
        raise NotImplementedError(
            "EnforcedAmbipolarSheath is a placeholder"
        )
