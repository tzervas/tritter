"""Tritter multi-agent orchestration framework.

Provides model-tiered task delegation for efficient development:
- Opus: Large planning, architecture decisions, research synthesis
- Sonnet: Module implementation, test development, documentation
- Haiku: Bug fixes, config changes, focused execution

Why: Token and cost optimization through intelligent task routing.
Complex tasks need powerful models, simple tasks need fast models.
"""

from tritter.agents.orchestrator import AgentOrchestrator, TaskComplexity
from tritter.agents.router import ModelRouter

__all__ = ["AgentOrchestrator", "TaskComplexity", "ModelRouter"]
