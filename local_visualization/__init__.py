"""Local, read-only adapters of Eric Friesen Waldner's visualization work.

Never imported by the simulation/MPI entry points. Use local_post.py only
after jobs finish and files have been downloaded. Exact originals are archived.
"""
import os


def require_local():
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID"):
        raise RuntimeError("Visualization is local-only; download MPI results first")
    for key in ("OMPI_COMM_WORLD_SIZE","PMI_SIZE","PMIX_SIZE","MPI_LOCALNRANKS"):
        try:
            active = int(os.environ.get(key,"0")) > 0
        except ValueError:
            active = bool(os.environ.get(key))
        if active:
            raise RuntimeError("Do not run local visualization under an MPI launcher")


require_local()
