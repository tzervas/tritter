"""Compatibility shim for agent orchestration imports.

DEPRECATED: This module is deprecated. Use `devtools.agents` instead.

This module provides backward compatibility for code that imports from
`tritter.agents`. The actual implementation has moved to `devtools.agents`
as it is development tooling, not part of the core Tritter model.

Why:
    The agent orchestration code (ModelRouter, TaskSpec, etc.) is for Claude Code
    development workflow optimization, not for the ML model itself. Moving it to
    devtools clarifies this distinction. This shim maintains backward compatibility.

Migration:
    Old: from tritter.agents import ModelRouter
    New: from devtools.agents import ModelRouter

Deprecation: This compatibility shim will be removed in a future version.
    Update imports to use `devtools.agents` directly.
"""

from __future__ import annotations

# Re-export from devtools.agents for backward compatibility
# Note: This module is deprecated. Use devtools.agents directly.
from devtools.agents import ModelRouter, TaskComplexity, TaskSpec

__all__ = ["ModelRouter", "TaskComplexity", "TaskSpec"]
