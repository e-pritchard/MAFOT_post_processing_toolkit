"""
Pipeline orchestration.

Read profiles + footprints, run the physics, save results, generate
diagnostic plots.  All the physics choices live in the two SheathModel
instances constructed at the top of run_pipeline(); everything else is
plumbing.

Invocation:
    python -m mafot_postproc.main <data_dir> <file_tag>
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from .constants import (
    DEFAULT_FILE_TAG,
    DEFAULT_ION_MASS_KG,
    DEFAULT_ELECTRON_MASS_KG,
    DEFAULT_Z_ION,
    ELECTRON_TRANSMISSION_COEFF,
    ION_TRANSMISSION_COEFF,
    LC_MIN_KM_DEFAULT,
)
from .compute import compute_q_parallel
from .physics import (
    AmbipolarSheath,
    project_parallel_to_perpendicular,
    WingenSheath,
)
from .io import (
    load_footprints_from_dir,
    load_species_profiles,
    save_result_npz,
)
from .diagnostics import (
    collapse_species_to_q_and_T_of_R,
    cumulative_charge_flux,
    particle_flux_from_q,
    fold_footprint_by_periodicity,
)
from . import plotting


def run_pipeline(data_dir, file_tag: str, *,
                 smoothing_window: int = 5,
                 periodicity: int = 1):
    """
    Run the full pipeline: ions + electrons + combined + diagnostics.

    Parameters
    ----------
    data_dir : Path-like
        Root directory containing footprints_ion/, footprints_electrons/,
        and Profiles/ subdirectories.  Named after the shot number.
    file_tag : str
        Tag appended to all output filenames.
    smoothing_window : int
        Uniform-filter window size for q_perp smoothing.  Set 0 to disable.
    periodicity : int
        n_period for fold_footprint_by_periodicity.  1 = no fold
        (axisymmetric case), 3 = DIII-D I-coil case.
    """
    # ------------- Directory setup -------------
    data_dir = Path(data_dir)
    shot = data_dir.name

    footprint_ion_dir = data_dir / "footprints_ion"
    footprint_electron_dir = data_dir / "footprints_electrons"
    profiles_dir = data_dir / "Profiles"

    out_dir = data_dir / "post_processing_outputs"
    out_dir.mkdir(exist_ok=True)

    file_path = str(out_dir)   # plotting expects a str/path-like
    tag_electron = file_tag + "_electron"
    tag_ion = file_tag + "_ion"
    tag_congre = file_tag + "_congre"

    print(f"Shot:             {shot}")
    print(f"Data directory:   {data_dir}")
    print(f"Output directory: {out_dir}")
    print(f"File tag:         {file_tag}")

    # ------------- Sheath model choice -------------
    # This is the plug point.  Changing which SheathModel you construct
    # here changes the physics of the whole pipeline in one place.
    
    # Most updated sheath model
    ion_sheath = AmbipolarSheath(transmission_coeff=ION_TRANSMISSION_COEFF)
    electron_sheath = AmbipolarSheath(transmission_coeff=ELECTRON_TRANSMISSION_COEFF)

    # Wingen Sheath
    # ion_sheath = WingenSheath()
    # electron_sheath = WingenSheath()


    print(f"Ion sheath model:      {ion_sheath.name} "
        #   f"(gamma={ion_sheath.transmission_coeff})" #comment this line out if using WingenSheath()
        )
    print(f"Electron sheath model: {electron_sheath.name} "
        #   f"(gamma={electron_sheath.transmission_coeff})" #comment this line out if using WingenSheath()
        )

    # ------------- Load profiles -------------
    profiles_electron = load_species_profiles(profiles_dir, shot, "electron")
    profiles_ion = load_species_profiles(profiles_dir, shot, "ion")

    # ------------- Load footprints -------------
    footprints_ion = load_footprints_from_dir(footprint_ion_dir)
    footprints_electron = load_footprints_from_dir(footprint_electron_dir)
    print(f"Loaded {len(footprints_ion)} ion footprints, "
          f"{len(footprints_electron)} electron footprints")

    # ------------- Compute q_parallel -------------
    ion_out = compute_q_parallel(
        footprints_ion, profiles_ion,
        sheath_model=ion_sheath,
        ion_mass_kg=DEFAULT_ION_MASS_KG,
        profile_2=profiles_electron,
        Lc_min_km=LC_MIN_KM_DEFAULT,
    )
    ele_out = compute_q_parallel(
        footprints_electron, profiles_electron,
        sheath_model=electron_sheath,
        # ion_mass_kg=DEFAULT_ELECTRON_MASS_KG,
        ion_mass_kg=DEFAULT_ION_MASS_KG,
        profile_2=profiles_ion,
        Lc_min_km=LC_MIN_KM_DEFAULT,
    )

    # ------------- Ambipolar diagnostic -------------
    # Uses q_parallel (before wall projection); the sheath physics is
    # defined on parallel flux, so the diagnostic is cleaner there.
    R_min_shared = min(np.nanmin(ion_out["R"]), np.nanmin(ele_out["R"]))
    R_max_shared = max(np.nanmax(ion_out["R"]), np.nanmax(ele_out["R"]))
    R_shared = (R_min_shared, R_max_shared)

    ion_col = collapse_species_to_q_and_T_of_R(
        ion_out["phi"], ion_out["R"], ion_out["q_parallel"],
        ion_out["psi_min"], profiles_ion,
        R_range=R_shared,
    )
    ele_col = collapse_species_to_q_and_T_of_R(
        ele_out["phi"], ele_out["R"], ele_out["q_parallel"],
        ele_out["psi_min"], profiles_electron,
        R_range=R_shared,
)

    ion_col["Gamma"] = particle_flux_from_q(
        ion_col["q_mean"], ion_col["T_mean_keV"],
        gamma_sheath=1,
        # ion_sheath.transmission_coeff,
    )
    ele_col["Gamma"] = particle_flux_from_q(
        ele_col["q_mean"], ele_col["T_mean_keV"], 
        gamma_sheath=1,
        # electron_sheath.transmission_coeff,
    )

    if not np.allclose(ion_col["R"], ele_col["R"]):
        raise RuntimeError(
            "Ion and electron collapse grids differ.  Did you use the "
            "same R_range / R_bins?"
        )

    cum = cumulative_charge_flux(
        ion_col["R"], ion_col["Gamma"], ele_col["Gamma"],
        Z_i=DEFAULT_Z_ION,
    )
    print(f"Global ambipolarity ratio  Z_i*F_i / F_e  = "
          f"{cum['global_ambipolarity_ratio']:.4f}  (want ~1.0)")

    plotting.plot_ambipolar_diagnostic(
        ion_col, ele_col, cum,
        Z_i=DEFAULT_Z_ION,
        filetag=file_tag, file_path=file_path,
    )

    # ------------- Project to q_perp -------------
    q_perp_ion, cos_inc_ion = project_parallel_to_perpendicular(
        ion_out["q_parallel"], ion_out["R"], ion_out["Z"],
        ion_out["BR"], ion_out["Bphi"], ion_out["BZ"],
        clip_negative=False,
    )
    q_perp_ele, cos_inc_ele = project_parallel_to_perpendicular(
        ele_out["q_parallel"], ele_out["R"], ele_out["Z"],
        ele_out["BR"], ele_out["Bphi"], ele_out["BZ"],
        clip_negative=False,
    )
    q_parallel_congre = ion_out["q_parallel"] + ele_out["q_parallel"]
    q_perp_congre = q_perp_ion + q_perp_ele

    # ------------- Smoothing -------------
    if smoothing_window and smoothing_window > 1:
        q_perp_ion = uniform_filter(q_perp_ion, size=smoothing_window,
                                    mode="nearest")
        q_perp_ele = uniform_filter(q_perp_ele, size=smoothing_window,
                                    mode="nearest")
        q_perp_congre = uniform_filter(q_perp_congre, size=smoothing_window,
                                       mode="nearest")

    # ------------- 2D diagnostic plots -------------
    plotting.toroidal_slicing(
        q_perp_ele, ele_out["phi"], ele_out["Z"], ele_out["s_wall"],
        phi_slice=60, filetag=tag_electron, file_path=file_path,
    )
    plotting.heat_flux_map(q_perp_ele,
                           filetag=tag_electron, file_path=file_path)
    plotting.plot_lc_vs_psimin(footprints_electron,
                                lc_min_lim=LC_MIN_KM_DEFAULT,
                                filetag=tag_electron, file_path=file_path)

    plotting.toroidal_slicing(
        q_perp_ion, ion_out["phi"], ion_out["Z"], ion_out["s_wall"],
        phi_slice=60, filetag=tag_ion, file_path=file_path,
    )
    plotting.heat_flux_map(q_perp_ion,
                           filetag=tag_ion, file_path=file_path)
    plotting.plot_lc_vs_psimin(footprints_ion,
                                lc_min_lim=LC_MIN_KM_DEFAULT,
                                filetag=tag_ion, file_path=file_path)

    plotting.toroidal_slicing(
        q_perp_congre, ele_out["phi"], ele_out["Z"], ele_out["s_wall"],
        phi_slice=60, filetag=tag_congre, file_path=file_path,
    )
    plotting.heat_flux_map(q_perp_congre,
                           filetag=tag_congre, file_path=file_path)

    # Optional fold for I-coil cases
    if periodicity > 1:
        folded = fold_footprint_by_periodicity(
            ele_out["phi"], ele_out["R"], ele_out["Z"],
            ele_out["q_parallel"], periodicity,
            require_exact=False,
        )
        plotting.heat_flux_map(
            folded["q_mean"],
            filetag=f"{tag_electron}_folded_n{periodicity}",
            file_path=file_path,
        )

    # ------------- Save arrays -------------
    for tag, out, q_perp, cos_inc in [
        (tag_electron, ele_out, q_perp_ele, cos_inc_ele),
        (tag_ion,      ion_out, q_perp_ion, cos_inc_ion),
    ]:
        path = save_result_npz(
            out_dir / f"{tag}_{shot}_heat_flux_footprints.npz",
            phi=out["phi"], s_wall=out["s_wall"], R=out["R"], Z=out["Z"],
            q_parallel=out["q_parallel"], q_perp=q_perp,
            cos_incidence=cos_inc,
        )
        print(f"Saved: {path}")

    congre_path = save_result_npz(
        out_dir / f"{tag_congre}_{shot}_heat_flux_footprints.npz",
        phi=ele_out["phi"], s_wall=ele_out["s_wall"],
        R=ele_out["R"], Z=ele_out["Z"],
        q_parallel=q_parallel_congre, q_perp=q_perp_congre,
        cos_incidence=cos_inc_ele,
    )
    print(f"Saved: {congre_path}")

    print("Pipeline complete.")


def main():
    p = argparse.ArgumentParser(
        description="MAFOT footprint post-processing pipeline",
    )
    p.add_argument("data_dir", type=Path,
                   help="Root directory containing footprints and Profiles")
    p.add_argument("file_tag", type=str, nargs="?", default=DEFAULT_FILE_TAG,
                   help=f"File tag for outputs (default: {DEFAULT_FILE_TAG})")
    p.add_argument("--smoothing-window", type=int, default=5,
                   help="Uniform-filter window for q_perp (0 = none)")
    p.add_argument("--periodicity", type=int, default=1,
                   help="Toroidal fold: 1 axisymmetric, 3 I-coil")
    args = p.parse_args()

    run_pipeline(args.data_dir, args.file_tag,
                 smoothing_window=args.smoothing_window,
                 periodicity=args.periodicity)


if __name__ == "__main__":
    main()
