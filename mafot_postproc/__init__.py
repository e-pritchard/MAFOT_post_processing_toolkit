"""
MAFOT footprint post-processing pipeline.

Structure:

    constants        - physical constants and defaults
    datatypes        - Footprint, Profiles dataclasses
    io               - reading footprints and profiles, saving outputs
    physics/         - sheath models, geometry, distributions
    compute          - core physics: compute_q_parallel
    diagnostics/     - collapse, fold, ambipolar, penetration
    plotting         - all matplotlib code
    main             - orchestration + CLI entry point

Public entry point:

    from mafot_postproc.main import run_pipeline
    run_pipeline(data_dir, file_tag)

or from the command line:

    python -m mafot_postproc.main <data_dir> <file_tag>
"""

from .main import run_pipeline

__all__ = ["run_pipeline"]
