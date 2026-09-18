#!/usr/bin/env python3
"""
Run MAFOT SDVW energy scan on Omega via Slurm batch submission, then
kick off post-processing automatically once all MAFOT runs finish.

Flow:
    1. Submit N MAFOT jobs to the preemptable queue, one per energy per
       species.  Collect their job IDs.
    2. Submit a post-processing job with --dependency=afterany:<all ids>.
       Slurm holds it until all MAFOT jobs have finished (successful,
       failed, or preempted-and-not-requeued — see afterany vs afterok
       discussion below).
    3. Post-processing job runs on a separate (non-preemptive) queue so
       it can't get killed after all the MAFOT work has completed.

Directory layout expected:
    BASE_DIR/
        _inner_ion.dat
        _inner_electron.dat

Output layout (created if missing):
    DATA_DIR/
        inners/                  per-run _inner files
        sbatch/                  the generated .sbatch scripts (audit)
        logs/                    MAFOT stdout + Slurm .out / .err files
        footprints_ion/          ion foot_in_*.dat
        footprints_electrons/    electron foot_in_*.dat
        Profiles/                (must exist for post-processing to work)
        post_processing_outputs/ (created by post-processor)

Usage:
    # Full sweep + post-processing
    python3 run_mafot_energy_scan_omega.py \\
        --data-dir Products/171491_test \\
        --file-tag energy_scan_test

    # MAFOT only, skip post-processing
    python3 run_mafot_energy_scan_omega.py \\
        --data-dir Products/171491_test \\
        --file-tag energy_scan_test \\
        --no-postproc

    # Post-processing only (assumes MAFOT already ran)
    python3 run_mafot_energy_scan_omega.py \\
        --data-dir Products/171491_test \\
        --file-tag energy_scan_test \\
        --postproc-only

    # Dry run: write sbatch files but don't submit
    python3 run_mafot_energy_scan_omega.py \\
        --data-dir Products/171491_test \\
        --file-tag energy_scan_test \\
        --dry-run

    # Then monitor with:
    #   squeue -u pritcharde
    #   ls -la Products/171491_test/logs/
"""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np


# ==================================================================
# USER SETTINGS - edit these before first run on Omega
# ==================================================================
SHOT = "171491"
DEFAULT_FILE_TAG = "3d_Fields_energy_scan"

# Path to the MAFOT binary on Omega - must exist before you submit anything
MAFOT_BIN = Path("/home/pritcharde/MAFOT/bin/SDVW_dtfoot_mpi")

# Directory containing _inner_ion.dat and _inner_electron.dat
BASE_DIR = Path("/home/pritcharde/mafot_inputs/_inner_pipeline")

# Where the post-processing package lives on Omega.  Either this directory
# (or its parent) must contain the mafot_postproc/ package for the import
# to work.
POSTPROC_PACKAGE_PARENT = Path("/home/pritcharde/")

# ------- Slurm settings for MAFOT jobs -------
SLURM_PARTITION = "preemptable"     # confirm with `sinfo`
SLURM_NTASKS = 6                    # MPI ranks per run
SLURM_TIME = "04:00:00"              # max walltime HH:MM:SS per run
SLURM_REQUEUE = False                 # auto-resubmit if preempted
SLURM_ACCOUNT = None                 # set if your site requires --account

# ------- Slurm settings for the post-processing job -------
# Post-proc is short, single-node, no MPI.  Non-preemptive queue so it
# can't get killed after all your MAFOT work has finished.
POST_PARTITION = "short"             # confirm with `sinfo` - pick a non-preemptive queue
POST_TIME = "00:30:00"
POST_CPUS = 4                        # for numpy / matplotlib

# Module load lines (inserted verbatim into each sbatch script)
MODULE_LOAD_LINES = [
    "module load env/gcc11.x-mpi",
]

# Additional modules needed only for post-processing (python + libs)
POSTPROC_MODULE_LOAD_LINES = [
    "module load env/gcc11.x-mpi",
    "module load python",   # uncomment once you confirm the module name on Omega
]

# ------- Energy grid -------
ENERGIES_PEAK = np.logspace(-3, 0, 25)
ENERGIES_TAIL = np.arange(2, 26, 1)
ENERGIES_KEV = np.append(ENERGIES_PEAK, ENERGIES_TAIL)


# ==================================================================
# Helpers
# ==================================================================
def get_particle_label(input_file: Path) -> str:
    """Extract 'ion' or 'electron' from _inner_{label}.dat."""
    return input_file.stem.split("_")[-1]


def footprint_dir_name(particle_label: str) -> str:
    """Match the post-processor's expected folder names."""
    if particle_label == "ion":
        return "footprints_ion"
    elif particle_label == "electron":
        return "footprints_electrons"
    else:
        raise ValueError(f"Unknown particle label: {particle_label!r}")


def make_input_file(base_input: Path, E: float, particle_label: str,
                    file_tag: str, out_dir: Path) -> Path:
    """Write a modified _inner with overridden Ekin, into out_dir."""
    with open(base_input) as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.strip().startswith("Ekin"):
            new_lines.append(f"Ekin[keV]=      {E}\n")
        else:
            new_lines.append(line)

    new_name = f"_inner_E{E}_{particle_label}_shot{SHOT}_{file_tag}.dat"
    new_path = out_dir / new_name
    with open(new_path, "w") as f:
        f.writelines(new_lines)

    return new_path


def build_mafot_sbatch(*, job_name: str, tag: str, inner_file: Path,
                       data_dir: Path, particle_label: str) -> str:
    """Assemble a MAFOT sbatch script as a string."""
    inner_dir = data_dir / "inners"
    log_dir = data_dir / "logs"
    fp_dir = data_dir / footprint_dir_name(particle_label)

    slurm_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={SLURM_PARTITION}",
        f"#SBATCH --nodes=1",
        f"#SBATCH --ntasks={SLURM_NTASKS}",
        f"#SBATCH --cpus-per-task=1",
        f"#SBATCH --time={SLURM_TIME}",
        f"#SBATCH --output={log_dir}/{job_name}_%j.out",
        f"#SBATCH --error={log_dir}/{job_name}_%j.err",
    ]
    if SLURM_REQUEUE:
        slurm_lines.append("#SBATCH --requeue")
    if SLURM_ACCOUNT:
        slurm_lines.append(f"#SBATCH --account={SLURM_ACCOUNT}")

    body_lines = [
        "",
        "set -euo pipefail",
        "",
        "# ---- environment ----",
        *MODULE_LOAD_LINES,
        "",
        "# ---- run MAFOT ----",
        f'cd "{inner_dir}"',
        f'echo "[$(date)] Starting MAFOT run: {tag}"',
        f"mpirun -n $SLURM_NTASKS {MAFOT_BIN} {inner_file.name} {tag}",
        f'echo "[$(date)] MAFOT run finished"',
        "",
        "# ---- move outputs to per-species dirs ----",
        f'mv "{inner_dir}"/log_dtfoot*{tag}*.dat "{log_dir}/" 2>/dev/null || echo "no log files matched"',
        f'mv "{inner_dir}"/foot_in*{tag}*.dat   "{fp_dir}/"  2>/dev/null || echo "no foot files matched"',
        "",
        'echo "[$(date)] Cleanup complete"',
    ]

    return "\n".join(slurm_lines + body_lines) + "\n"


def build_postproc_sbatch(*, data_dir: Path, file_tag: str,
                          dependency: str = None) -> str:
    """
    Assemble the post-processing sbatch script.

    dependency (optional): a Slurm dependency string like
    "afterany:12345:12346:12347".  If provided, the job stays queued until
    all named jobs have finished (regardless of exit code).
    """
    log_dir = data_dir / "logs"

    slurm_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=maf_postproc",
        f"#SBATCH --partition={POST_PARTITION}",
        f"#SBATCH --nodes=1",
        f"#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={POST_CPUS}",
        f"#SBATCH --time={POST_TIME}",
        f"#SBATCH --output={log_dir}/postproc_%j.out",
        f"#SBATCH --error={log_dir}/postproc_%j.err",
    ]
    if SLURM_ACCOUNT:
        slurm_lines.append(f"#SBATCH --account={SLURM_ACCOUNT}")
    if dependency:
        slurm_lines.append(f"#SBATCH --dependency={dependency}")
        # kill_on_invalid_dep=yes: if the dependency string somehow becomes
        # unsatisfiable (all jobs cancelled before starting), don't leave a
        # zombie post-proc job queued forever.
        slurm_lines.append("#SBATCH --kill-on-invalid-dep=yes")

    body_lines = [
        "",
        "set -euo pipefail",
        "",
        "# ---- environment ----",
        *POSTPROC_MODULE_LOAD_LINES,
        "",
        "# ---- make the post-processing package importable ----",
        f'export PYTHONPATH="{POSTPROC_PACKAGE_PARENT}:${{PYTHONPATH:-}}"',
        "",
        "# ---- run post-processing ----",
        f'echo "[$(date)] Starting post-processing on {data_dir}"',
        f'python3 -m mafot_postproc.main "{data_dir}" "{file_tag}"',
        f'echo "[$(date)] Post-processing complete"',
    ]

    return "\n".join(slurm_lines + body_lines) + "\n"


def submit_script(sbatch_path: Path, dry_run: bool = False):
    """Submit an sbatch script.  Returns job ID on success, None otherwise."""
    if dry_run:
        return None

    result = subprocess.run(
        ["sbatch", str(sbatch_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ! sbatch failed:")
        print(f"    stderr: {result.stderr.strip()}")
        return None
    # sbatch's stdout is typically "Submitted batch job 12345"
    return result.stdout.strip().split()[-1]


def submit_one_mafot_run(base_input: Path, E: float, file_tag: str,
                         data_dir: Path, dry_run: bool = False):
    """Build + submit one MAFOT run.  Returns (job_id, sbatch_path)."""
    particle_label = get_particle_label(base_input)

    inner_dir = data_dir / "inners"
    sbatch_dir = data_dir / "sbatch"

    inner_file = make_input_file(base_input, E, particle_label, file_tag,
                                 inner_dir)

    tag = f"shot{SHOT}_{file_tag}_{particle_label}_E{E}"
    job_name = f"maf_{particle_label[0]}_E{E:.4g}"

    script = build_mafot_sbatch(
        job_name=job_name,
        tag=tag,
        inner_file=inner_file,
        data_dir=data_dir,
        particle_label=particle_label,
    )
    sbatch_path = sbatch_dir / f"{tag}.sbatch"
    sbatch_path.write_text(script)

    job_id = submit_script(sbatch_path, dry_run=dry_run)
    return (job_id, sbatch_path)


def submit_postproc(data_dir: Path, file_tag: str,
                    mafot_job_ids=None, dry_run: bool = False):
    """
    Build + submit the post-processing job.

    If mafot_job_ids is given, uses --dependency=afterany so the job stays
    queued until all listed jobs have finished.  If None, submits with no
    dependency (post-proc runs as soon as it's scheduled).
    """
    sbatch_dir = data_dir / "sbatch"

    if mafot_job_ids:
        # afterany, not afterok: run when all deps FINISH regardless of
        # exit code.  Preempted-and-not-requeued jobs shouldn't block
        # the post-processor.
        dependency = "afterany:" + ":".join(mafot_job_ids)
    else:
        dependency = None

    script = build_postproc_sbatch(
        data_dir=data_dir, file_tag=file_tag, dependency=dependency,
    )
    sbatch_path = sbatch_dir / "postproc.sbatch"
    sbatch_path.write_text(script)

    return submit_script(sbatch_path, dry_run=dry_run)


# ==================================================================
# Main
# ==================================================================
def main(data_dir: Path, file_tag: str, *,
         dry_run: bool = False,
         no_postproc: bool = False,
         postproc_only: bool = False):

    for sub in ("inners", "sbatch", "logs",
                "footprints_ion", "footprints_electrons"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)

    print(f"Data directory:      {data_dir.resolve()}")
    print(f"Base _inner dir:     {BASE_DIR.resolve()}")
    print(f"File tag:            {file_tag}")
    print(f"MAFOT binary:        {MAFOT_BIN}")
    print(f"MAFOT partition:     {SLURM_PARTITION}")
    print(f"Post-proc partition: {POST_PARTITION}")
    print(f"Energy grid:         {len(ENERGIES_KEV)} points from "
          f"{ENERGIES_KEV.min():.4g} to {ENERGIES_KEV.max():.4g} keV")
    if dry_run:
        print("*** DRY RUN - sbatch files written but not submitted ***")
    if no_postproc:
        print("*** --no-postproc: MAFOT only, post-processing skipped ***")
    if postproc_only:
        print("*** --postproc-only: skipping MAFOT submission ***")
    print()

    submitted_mafot = []   # list of (job_id, sbatch_filename)
    failed_mafot = []

    # ---------- MAFOT submissions ----------
    if not postproc_only:
        for base_input in BASE_DIR.iterdir():
            if not base_input.is_file():
                continue
            if not base_input.name.startswith("_inner"):
                continue

            particle_label = get_particle_label(base_input)
            print(f"===== {particle_label.upper()} runs "
                  f"({base_input.name}) =====")

            for E in ENERGIES_KEV:
                job_id, sbatch_path = submit_one_mafot_run(
                    base_input, E, file_tag, data_dir, dry_run=dry_run,
                )
                if job_id:
                    submitted_mafot.append((job_id, sbatch_path.name))
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                          f"submitted E={E:.4g} -> job {job_id}")
                elif dry_run:
                    submitted_mafot.append(("DRY", sbatch_path.name))
                else:
                    failed_mafot.append(sbatch_path.name)
                    print(f"  ! submission failed for E={E:.4g}")

        print("\n----- MAFOT submission summary -----")
        print(f"  Submitted: {len(submitted_mafot)}")
        print(f"  Failed:    {len(failed_mafot)}")
        for name in failed_mafot:
            print(f"    [failed] {name}")

    # ---------- Post-processing submission ----------
    if no_postproc:
        print("\nSkipping post-processing (--no-postproc).")
        post_id = None
    else:
        print("\n===== Post-processing =====")
        if postproc_only:
            mafot_ids = None
        else:
            mafot_ids = [jid for (jid, _) in submitted_mafot
                         if jid and jid != "DRY"]
            if not mafot_ids and not dry_run:
                print("  ! No MAFOT jobs successfully submitted; "
                      "skipping post-processor.")
                post_id = None
                mafot_ids = None

        if mafot_ids or postproc_only or dry_run:
            post_id = submit_postproc(
                data_dir, file_tag,
                mafot_job_ids=mafot_ids,
                dry_run=dry_run,
            )
            if post_id:
                dep_msg = (f"waiting on {len(mafot_ids)} MAFOT jobs"
                           if mafot_ids else "no dependency")
                print(f"  submitted post-processor -> job {post_id} "
                      f"({dep_msg})")
            elif dry_run:
                print(f"  [dry-run] post-processor sbatch written but "
                      f"not submitted")
            else:
                print(f"  ! post-processor submission failed")
        else:
            post_id = None

    if not dry_run and (submitted_mafot or post_id):
        print("\nMonitor with:")
        print(f"  squeue -u $USER")
        print(f"  sacct -u $USER --starttime today")
        print(f"  tail -f {data_dir}/logs/postproc_*.out   # once post-proc starts")
        print(f"Cancel all with:")
        print(f"  scancel -u $USER")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit MAFOT energy scan + dependent post-processing "
                    "to Slurm on Omega",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"Root output dir (default: Products/{SHOT}_<file-tag>)",
    )
    parser.add_argument(
        "--file-tag",
        type=str,
        default=DEFAULT_FILE_TAG,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write sbatch scripts to disk but don't submit them",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--no-postproc",
        action="store_true",
        help="Skip post-processing submission",
    )
    mode.add_argument(
        "--postproc-only",
        action="store_true",
        help="Only submit the post-processor (assumes MAFOT already ran)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or Path(f"Products/{SHOT}_{args.file_tag}")
    main(
        data_dir, args.file_tag,
        dry_run=args.dry_run,
        no_postproc=args.no_postproc,
        postproc_only=args.postproc_only,
    )
