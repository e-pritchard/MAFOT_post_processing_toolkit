"""
Data containers for the pipeline.

These are pure data classes with no methods and no logic.  If you find
yourself wanting to add a method, ask whether it belongs in physics/,
diagnostics/, or io.py instead.
"""

from dataclasses import dataclass
from typing import Callable, Dict
import numpy as np


@dataclass
class Footprint:
    """
    One MAFOT footprint file, reshaped into 2D grids of shape (Nphi, Nt).

    All arrays share the same shape and correspond to the same set of
    (phi, s_wall) sample points.
    """
    phi: np.ndarray         # (Nphi, Nt) toroidal angle [rad]
    s_wall: np.ndarray      # (Nphi, Nt) wall arc length
    psi_min: np.ndarray     # (Nphi, Nt) minimum psi reached by the orbit
    R: np.ndarray           # (Nphi, Nt) major radius [m]
    Z: np.ndarray           # (Nphi, Nt) height [m]
    BR: np.ndarray          # (Nphi, Nt) B_R at impact point [T]
    BZ: np.ndarray          # (Nphi, Nt) B_Z at impact point [T]
    Bphi: np.ndarray        # (Nphi, Nt) B_phi at impact point [T]
    Lc: np.ndarray          # (Nphi, Nt) total connection length [km]
    Lc_psimin: np.ndarray   # (Nphi, Nt) connection length to psi_min [km]
    meta: Dict[str, float]  # parsed header key/value pairs


@dataclass
class Profiles:
    """
    Callable interpolants for density and temperature vs psi.

    Field names say 'ion' but the same class holds electron profiles too
    (just pass electron data at construction time).  Both interpolants
    accept a psi array of any shape and return an array of the same shape.
    """
    n_i: Callable[[np.ndarray], np.ndarray]      # density [m^-3]
    T_i_keV: Callable[[np.ndarray], np.ndarray]  # temperature [keV]
