"""Friendly precondition check for missing optional ML deps.

Importing this module raises SystemExit(2) with an actionable message if torch (or another
named module) isn't importable. Used at the top of every CLI entry point so a user running
`python -m epintlm.cli.train` from a non-activated env gets a clear hint instead of a
ModuleNotFoundError traceback.
"""

from __future__ import annotations

import importlib
import sys


def require(*module_names: str) -> None:
    missing = []
    for name in module_names:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        msg = (
            f"ERROR: required Python module(s) not importable: {', '.join(missing)}.\n"
            f"Activate the project env first:\n"
            f"    module load Anaconda3       # on HPC\n"
            f"    source activate Bioinfomatrics\n"
            f"Or install deps: pip install -r requirements.txt\n"
        )
        sys.stderr.write(msg)
        sys.exit(2)
