"""Pytest configuration and fixtures.

Why:
    Ensures devtools package is importable during test collection.
    The project root must be in sys.path before pytest tries to
    import test modules that depend on devtools.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in path for devtools imports - must happen at import time
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Early hook to ensure devtools is importable before test collection."""
    # Force devtools into sys.modules early
    import devtools  # noqa: F401
