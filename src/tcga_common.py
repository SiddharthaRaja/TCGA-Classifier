#!/usr/bin/env python3
"""
Shared state, locking and lifecycle helpers for the TCGA-BRCA pipeline.

WHY THIS FILE EXISTS
--------------------
Every job in this pipeline has been killed mid-run at least once. So progress
must be recorded in a way that survives a SIGKILL at any instruction, and must
live somewhere that survives a /scratch wipe.

Two rules follow from that:

  1. STATE LIVES IN $HOME, NOT WITH THE DATA.
     TCGA_STATE_ROOT (default ~/siddhartha/state) holds the completion markers,
     the extracted feature vectors, the small clinical/biospecimen/CNV copies
     and the slide thumbnails. TCGA_DATA_ROOT (default ~/siddhartha/data) holds
     only the bulky raw WSI files, which are disposable by design and may point
     at /scratch. Losing TCGA_DATA_ROOT costs you download time. Losing
     TCGA_STATE_ROOT costs you the project.

  2. EVERY WRITE IS ATOMIC.
     Markers are written to a temp file in the same directory and then
     os.replace()d into position. A marker therefore either does not exist or
     is complete and valid. There is no torn state to reason about after a
     kill, which is what makes "just resubmit it" safe.

A patient moves through exactly three states, and the marker files ARE the
state machine:

    (nothing)            -> not downloaded
    download_complete    -> all modalities on disk, sizes verified
    extract_complete     -> all backbones extracted, features saved

Both scripts read these markers rather than inspecting directory contents, so
deleting a WSI after extraction can never be mistaken for "needs downloading".
"""

import json
import os
import signal
import socket
import time
from pathlib import Path

# --------------------------------------------------------------------- roots

def _root(env_name: str, default_subdir: str) -> Path:
    # TCGA_SCRATCH_ROOT is honoured as a legacy fallback so old job scripts
    # that only export that variable keep working.
    val = os.environ.get(env_name)
    if val:
        return Path(val)
    legacy = os.environ.get("TCGA_SCRATCH_ROOT")
    if legacy and default_subdir == "data":
        return Path(legacy)
    return Path.home() / "siddhartha" / default_subdir


DATA_ROOT = _root("TCGA_DATA_ROOT", "data")     # bulky, disposable, may be /scratch
STATE_ROOT = _root("TCGA_STATE_ROOT", "state")  # small, precious, keep on $HOME

RAW_DIR = DATA_ROOT / "raw"
MANIFEST = STATE_ROOT / "case_manifest.json"
PATIENTS_DIR = STATE_ROOT / "patients"            # per-patient markers + small files
FEATURES_PATCH_ROOT = STATE_ROOT / "features_patch"
FEATURES_POOLED_ROOT = STATE_ROOT / "features_pooled"
THUMBS_DIR = STATE_ROOT / "thumbnails"
LOCKS_DIR = STATE_ROOT / "locks"

DOWNLOAD_MARKER = "download_complete.json"
EXTRACT_MARKER = "extract_complete.json"

LOCK_TIMEOUT_S = 6 * 3600  # a lock older than this is assumed to be from a killed job


def ensure_dirs():
    for d in (RAW_DIR, PATIENTS_DIR, FEATURES_PATCH_ROOT, FEATURES_POOLED_ROOT,
              THUMBS_DIR, LOCKS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def banner(msg):
    print(f"\n{'=' * 70}\n  {msg}\n{'=' * 70}", flush=True)


def describe_roots():
    print(f"  raw WSI (disposable) : {RAW_DIR}")
    print(f"  state + features     : {STATE_ROOT}", flush=True)


# ------------------------------------------------------------------- markers

def patient_state_dir(pid: str) -> Path:
    return PATIENTS_DIR / pid


def _marker_path(pid: str, name: str) -> Path:
    return patient_state_dir(pid) / name


def write_marker(pid: str, name: str, payload: dict):
    """Atomic: the marker either does not exist or is complete and parseable."""
    d = patient_state_dir(pid)
    d.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["written_by_job"] = os.environ.get("SLURM_JOB_ID", "interactive")
    tmp = d / (name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, d / name)


def read_marker(pid: str, name: str):
    p = _marker_path(pid, name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def clear_marker(pid: str, name: str):
    _marker_path(pid, name).unlink(missing_ok=True)


def is_downloaded(pid: str) -> bool:
    return read_marker(pid, DOWNLOAD_MARKER) is not None


def is_extracted(pid: str) -> bool:
    return read_marker(pid, EXTRACT_MARKER) is not None


# --------------------------------------------------------------------- locks

class PatientLock:
    """Advisory per-patient lock so two processes never work the same patient.

    Created with O_EXCL, so acquisition is atomic. A lock left behind by a
    killed job would block that patient forever, so a lock older than
    LOCK_TIMEOUT_S is treated as stale and taken over.

    Used as:  with PatientLock(pid) as lk:
                  if not lk.acquired: continue
    """

    def __init__(self, pid: str, timeout_s: int = LOCK_TIMEOUT_S):
        LOCKS_DIR.mkdir(parents=True, exist_ok=True)
        self.pid = pid
        self.path = LOCKS_DIR / f"{pid}.lock"
        self.timeout_s = timeout_s
        self.acquired = False

    def _try_create(self) -> bool:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as fh:
            json.dump({"job": os.environ.get("SLURM_JOB_ID", "interactive"),
                       "host": socket.gethostname(), "pid": os.getpid(),
                       "ts": time.time()}, fh)
        return True

    def __enter__(self):
        if self._try_create():
            self.acquired = True
            return self
        # Existing lock — steal it only if it is clearly abandoned.
        try:
            age = time.time() - self.path.stat().st_mtime
        except FileNotFoundError:
            age = None
        if age is not None and age > self.timeout_s:
            print(f"      stale lock ({age / 3600:.1f}h old) — taking over", flush=True)
            self.path.unlink(missing_ok=True)
            self.acquired = self._try_create()
        return self

    def __exit__(self, *exc):
        if self.acquired:
            self.path.unlink(missing_ok=True)
        return False


# ------------------------------------------------------- shutdown + time budget

class RunController:
    """Stops the loop cleanly instead of dying mid-patient.

    Two ways a long job ends badly: SLURM kills it at the wall clock, or
    something external cancels it. We can't prevent the second, but we can make
    the first graceful — SIGTERM sets a flag, the current patient finishes, the
    job exits 0 with all markers consistent.

    time_budget_h should be comfortably under the SBATCH --time so the job stops
    starting new patients while there is still time to finish the current one.
    """

    def __init__(self, time_budget_h: float = None):
        self.t_start = time.time()
        self.time_budget_s = time_budget_h * 3600 if time_budget_h else None
        self.stop_requested = False
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame):
        print(f"\n  !! signal {signum} received — finishing current patient, "
              f"then stopping cleanly. Resubmit to resume.\n", flush=True)
        self.stop_requested = True

    def elapsed_h(self) -> float:
        return (time.time() - self.t_start) / 3600

    def should_stop(self) -> tuple:
        if self.stop_requested:
            return True, "signal received"
        if self.time_budget_s and (time.time() - self.t_start) > self.time_budget_s:
            return True, f"time budget reached ({self.elapsed_h():.1f}h)"
        return False, ""


# ------------------------------------------------------------------ manifest

def load_manifest() -> list:
    if not MANIFEST.exists():
        raise SystemExit(
            f"ERROR: {MANIFEST} not found.\n"
            f"Build it first:  python download_tcga.py --manifest"
        )
    return json.loads(MANIFEST.read_text())


def select_patients(cases: list, shard: int, n_shards: int) -> list:
    """Partition by position in the MANIFEST, which never changes, rather than
    by position in a directory listing, which changes as files are written and
    deleted. That instability was why shards previously overlapped and dropped
    patients. Default n_shards=1 means plain sequential processing.
    """
    if n_shards <= 1:
        return list(cases)
    return cases[shard::n_shards]
