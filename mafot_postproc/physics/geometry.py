"""
Geometric helpers: wall surface normals, magnetic-field unit vectors, and
projection from parallel to perpendicular heat flux.

Pure geometry — no physics assumptions beyond the wall being a curve in
the poloidal (R, Z) plane at each toroidal angle.
"""

import numpy as np


def compute_surface_normal(R, Z, axis_s=-1, outward_sign=+1):
    """
    Poloidal-plane wall normal at each (phi, s_wall) point.

    Assumes the wall coordinate s runs along axis -1 (last axis of R, Z).
    The tangent along s is (dR/ds, dZ/ds); a 90-degree in-plane rotation
    gives the normal (-dZ/ds, 0, dR/ds).

    The `outward_sign` parameter exists to let callers flip the normal if
    they want it pointing into the plasma vs into the wall.  It is currently
    accepted but not used — the raw normal is returned, and project()
    handles sign via the clip_negative / abs choice.
    """
    dRds = np.gradient(R, axis=-1)
    dZds = np.gradient(Z, axis=-1)
    tmag = np.sqrt(dRds ** 2 + dZds ** 2)
    tmag_safe = np.where(tmag > 1e-15, tmag, 1.0)
    tR = dRds / tmag_safe
    tZ = dZds / tmag_safe

    nR = -tZ
    nphi = np.zeros_like(nR)
    nZ = tR
    return np.stack((nR, nphi, nZ), axis=-1)


def compute_b_unit(BR, Bphi, BZ):
    """
    Return B/|B| as an (..., 3) array.  Points where B is exactly zero
    are returned as (0, 0, 0) rather than NaN, so downstream math doesn't
    poison good bins.
    """
    Bmag = np.sqrt(BR ** 2 + Bphi ** 2 + BZ ** 2)
    mask_good = Bmag > 1e-30
    safe_Bmag = np.where(mask_good, Bmag, 1.0)

    b_hat = np.zeros(BR.shape + (3,))
    b_hat[..., 0] = BR / safe_Bmag
    b_hat[..., 1] = Bphi / safe_Bmag
    b_hat[..., 2] = BZ / safe_Bmag
    b_hat[~mask_good] = 0.0
    return b_hat


def project_parallel_to_perpendicular(
    q_parallel,
    R, Z,
    BR, Bphi, BZ,
    *,
    clip_negative=True,
    outward_sign=+1,
):
    """
    Project field-aligned heat flux onto the wall surface normal:

        q_perp = q_parallel * (n_hat . b_hat)

    Two modes for handling the sign of the cosine:

    * `clip_negative=True`: use max(-cos_inc, 0).  With the surface-normal
      convention here (n_R > 0 points into the plasma), field lines that
      strike the wall have n.B < 0 — so -cos_inc is the physically
      relevant intensity, and clipping zeros out regions where co-passing
      particles wouldn't reach the wall.

    * `clip_negative=False`: use |cos_inc|.  Doesn't distinguish which
      side of the wall the flux comes from; useful when you just want the
      magnitude.
    """
    n_hat = compute_surface_normal(R, Z, axis_s=-1, outward_sign=outward_sign)
    b_hat = compute_b_unit(BR, Bphi, BZ)
    cos_inc = np.sum(n_hat * b_hat, axis=-1)

    if clip_negative:
        cos_inc_for_flux = np.maximum(-cos_inc, 0.0)
    else:
        cos_inc_for_flux = np.abs(cos_inc)

    q_perp = q_parallel * cos_inc_for_flux
    return q_perp, cos_inc
