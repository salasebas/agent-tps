"""Application configuration, paths, and platform defaults for TokPulse."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def get_app_data_dir() -> Path:
    """Returns the platform-specific data directory for TokPulse.

    Matches macOS Application Support, Linux XDG Data Home, and Windows APPDATA.
    """
    custom_dir = os.environ.get("TOKPULSE_DATA_DIR")
    if custom_dir:
        path = Path(custom_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    if sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / "tokpulse"
    elif sys.platform == "win32":
        app_data = os.environ.get("APPDATA")
        base = Path(app_data) if app_data else Path.home() / "AppData" / "Roaming"
        path = base / "tokpulse"
    else:
        # Linux & Unix standard (XDG)
        xdg_data = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg_data) if xdg_data else Path.home() / ".local" / "share"
        path = base / "tokpulse"

    path.mkdir(parents=True, exist_ok=True)
    return path


def get_runs_dir() -> Path:
    """Returns the directory where benchmark run metrics are saved."""
    runs_dir = get_app_data_dir() / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


APP_NAME = "TokPulse"
APP_DESCRIPTION = "High-Velocity Benchmark & Concurrency Profiler for Coding Agents"
VERSION = "0.2.0"
