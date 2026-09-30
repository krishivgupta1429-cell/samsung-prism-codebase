"""Atomic artifacts and provenance, without importing the ML stack."""

import importlib.metadata
import json
import os
import platform
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from .config import REPO_ROOT, source_fingerprint


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, default=str, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def environment():
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    packages = dict(
        sorted(
            (dist.metadata["Name"], dist.version)
            for dist in importlib.metadata.distributions()
            if dist.metadata["Name"]
        )
    )
    processor = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                processor = line.partition(":")[2].strip()
                break
    try:
        gpu = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .strip()
            .splitlines()
        )
    except (OSError, subprocess.SubprocessError):
        gpu = []
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": processor,
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "nvidia_gpus_name_driver_memory_mib": gpu,
        "thread_environment": {
            name: os.environ[name]
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            if name in os.environ
        },
        "packages": packages,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_fingerprint": source_fingerprint(),
    }


def create_run(root, experiment_id):
    slug = (
        re.sub(r"[^a-zA-Z0-9_-]+", "-", experiment_id).strip("-")[:80] or "experiment"
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(root) / f"{stamp}-{slug}-{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path
