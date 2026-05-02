"""Per-run manifest: git state, environment, timestamp, config snapshot."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config, config_to_dict


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def build_manifest(cfg: Config) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "env": _gpu_info(),
        "config": config_to_dict(cfg),
    }


def write_manifest(run_dir: Path, cfg: Config) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "manifest.json"
    with out.open("w") as f:
        json.dump(build_manifest(cfg), f, indent=2)
    return out
