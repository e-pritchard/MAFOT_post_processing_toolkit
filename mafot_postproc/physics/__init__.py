"""Physics primitives — sheath models, geometry, distribution functions."""

from .geometry import (
    compute_surface_normal,
    compute_b_unit,
    project_parallel_to_perpendicular,
)
from .maxwellian import maxwellian_energy_pdf
from .sheath import (
    SheathModel,
    WingenSheath,
    AmbipolarSheath,
    EnforcedAmbipolarSheath,
)

__all__ = [
    "compute_surface_normal",
    "compute_b_unit",
    "project_parallel_to_perpendicular",
    "maxwellian_energy_pdf",
    "SheathModel",
    "WingenSheath",
    "AmbipolarSheath",
    "EnforcedAmbipolarSheath",
]
