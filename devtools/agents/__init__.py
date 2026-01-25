"""Multi-agent orchestration framework for Tritter development.

Provides model-tiered task delegation for efficient development:
- Opus: Large planning, architecture decisions, research synthesis
- Sonnet: Module implementation, test development, documentation
- Haiku: Bug fixes, config changes, focused execution

Why: Token and cost optimization through intelligent task routing.
Complex tasks need powerful models, simple tasks need fast models.

Note: This module is part of devtools (development tooling), NOT the
core Tritter model. It aids Claude Code orchestration during development
but is not a component of the ML model itself.
"""

from devtools.agents.orchestrator import TaskComplexity, TaskSpec
from devtools.agents.router import ModelRouter

__all__ = ["TaskComplexity", "TaskSpec", "ModelRouter"]
