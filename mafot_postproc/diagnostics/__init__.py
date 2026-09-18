"""Diagnostic reductions of 2D footprints."""

from .collapse import (
    fold_footprint_by_periodicity,
    collapse_footprint_to_q_of_R,
    collapse_species_to_q_and_T_of_R,
)
from .ambipolar import particle_flux_from_q, cumulative_charge_flux
from .penetration import (
    penetration_histogram_data,
    penetration_by_energy_data,
    penetration_by_phi_data,
    penetration_weighted_by_heat_flux_data,
    lc_vs_psimin_data,
)

__all__ = [
    "fold_footprint_by_periodicity",
    "collapse_footprint_to_q_of_R",
    "collapse_species_to_q_and_T_of_R",
    "particle_flux_from_q",
    "cumulative_charge_flux",
    "penetration_histogram_data",
    "penetration_by_energy_data",
    "penetration_by_phi_data",
    "penetration_weighted_by_heat_flux_data",
    "lc_vs_psimin_data",
]
