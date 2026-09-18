"""
I/O: read MAFOT footprint files, load profile files, save pipeline outputs.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d

from .datatypes import Footprint, Profiles


# ------------------------------------------------------------------
# Footprint file parsing
# ------------------------------------------------------------------
_HEADER_RE = re.compile(
    r"^\s*#\s*([^:]+?)\s*:\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)\s*$"
)


def _try_parse_float(line: str) -> Optional[Tuple[str, float]]:
    m = _HEADER_RE.match(line)
    if not m:
        return None
    return m.group(1).strip(), float(m.group(2))


def read_footprint_file(path: Union[str, Path]) -> Footprint:
    """
    Parse one MAFOT footprint file into a Footprint.

    Expected columns:
      phi[rad], length t, N_toroidal, Lc[km], psi_min, R[m], Z[m],
      Lc_at_psimin[km], B_R, B_Z, B_phi

    Reshape: file is laid out with phi varying fastest for each s_wall,
    so we reshape (Nphi, Nt) with Fortran ordering.
    """
    path = Path(path)
    meta: Dict[str, float] = {}
    data_lines: List[str] = []

    with path.open("r") as f:
        for line in f:
            if line.lstrip().startswith("#"):
                kv = _try_parse_float(line)
                if kv:
                    meta[kv[0]] = kv[1]
            elif line.strip():
                data_lines.append(line)

    data = np.loadtxt(data_lines)
    if data.ndim == 1:
        data = data[None, :]

    # Column layout
    phi, s_wall = data[:, 0], data[:, 1]
    Lc = data[:, 3]
    psi_min = data[:, 4]
    R, Z = data[:, 5], data[:, 6]
    Lc_psimin = data[:, 7]
    BR, BZ, Bphi = data[:, 8], data[:, 9], data[:, 10]

    # Grid sizes — prefer header, fall back to inferring from unique values
    Nphi = int(meta.get("phi-grid", 0))
    Nt = int(meta.get("t-grid", 0))
    if not (Nphi > 0 and Nt > 0 and data.shape[0] == Nphi * Nt):
        Nphi = len(np.unique(phi))
        Nt = len(np.unique(s_wall))
        if Nphi * Nt != data.shape[0]:
            raise ValueError(
                f"Cannot reshape {path.name}: got {data.shape[0]} rows, "
                f"inferred Nphi={Nphi}, Nt={Nt} (product {Nphi*Nt})."
            )

    def _reshape(a):
        return a.reshape(Nphi, Nt, order="F")

    return Footprint(
        phi=_reshape(phi),
        s_wall=_reshape(s_wall),
        psi_min=_reshape(psi_min),
        R=_reshape(R),
        Z=_reshape(Z),
        BR=_reshape(BR),
        BZ=_reshape(BZ),
        Bphi=_reshape(Bphi),
        Lc=_reshape(Lc),
        Lc_psimin=_reshape(Lc_psimin),
        meta=meta,
    )


def load_footprints_from_dir(
    directory: Union[str, Path]
) -> Dict[float, Footprint]:
    """
    Read every footprint file in a directory, keyed by the 'Ekin' value
    parsed from each file's header.
    """
    directory = Path(directory)
    out: Dict[float, Footprint] = {}
    for f in directory.iterdir():
        if not f.is_file():
            continue
        # Peek at the header for Ekin
        with open(f) as fh:
            head = fh.readlines(1500)
        Ekin: Optional[str] = None
        for line in head:
            if "Ekin:" in line:
                Ekin = line.split("Ekin: ")[-1].split("\n")[0]
                break
        if Ekin is None:
            continue
        out[float(Ekin)] = read_footprint_file(f)
    if not out:
        raise FileNotFoundError(
            f"No footprint files with a parseable 'Ekin:' header in "
            f"{directory}"
        )
    return out


# ------------------------------------------------------------------
# Profile loading
# ------------------------------------------------------------------
def make_profile_interpolants_keV(
    psi_prof: np.ndarray,
    n_prof: np.ndarray,
    T_prof_keV: np.ndarray,
    *,
    kind: str = "linear",
    fill: str = "extrapolate",
) -> Profiles:
    """Build a Profiles object from parallel arrays of psi, n, and T."""
    n_itp = interp1d(psi_prof, n_prof, kind=kind,
                     bounds_error=False, fill_value=fill)
    T_itp = interp1d(psi_prof, T_prof_keV, kind=kind,
                     bounds_error=False, fill_value=fill)
    return Profiles(
        n_i=lambda x: np.asarray(n_itp(x)),
        T_i_keV=lambda x: np.asarray(T_itp(x)),
    )


def load_species_profiles(
    profiles_dir: Union[str, Path],
    shot: str,
    species: str,
    *,
    density_scale: float = 1e20,
) -> Profiles:
    """
    Load n and T for one species from files named
    <shot>_n<x>.dat and <shot>_t<x>.dat, where <x> is 'e' or 'i'.

    Assumes density files are in units of 1e20 / m^3 (DIII-D convention).
    Column 0 is psi_N, column 1 is the profile value.
    """
    profiles_dir = Path(profiles_dir)
    x = "e" if species.startswith("e") else "i"
    n_file = profiles_dir / f"{shot}_n{x}.dat"
    T_file = profiles_dir / f"{shot}_t{x}.dat"

    if not n_file.exists():
        raise FileNotFoundError(f"Missing density profile: {n_file}")
    if not T_file.exists():
        raise FileNotFoundError(f"Missing temperature profile: {T_file}")

    n_loaded = np.loadtxt(n_file)
    T_loaded = np.loadtxt(T_file)

    return make_profile_interpolants_keV(
        psi_prof=n_loaded[:, 0],
        n_prof=n_loaded[:, 1] * density_scale,
        T_prof_keV=T_loaded[:, 1],
    )


# ------------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------------
def save_result_npz(path: Union[str, Path], **arrays) -> Path:
    """
    Save a set of named arrays to an .npz.  Thin wrapper around np.savez
    that returns the resolved path for logging.
    """
    path = Path(path)
    np.savez(path, **arrays)
    return path.resolve()
