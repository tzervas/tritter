"""Development tooling for Tritter project.

This module provides development utilities separate from the core Tritter model code.
It includes validation runners, project analysis tools, agent orchestration, and CLI
utilities for streamlining the development workflow.

Why:
    The Tritter model code (src/tritter/) should remain clean and focused on the
    ML model implementation. Development tooling, CI/CD helpers, agent orchestration,
    and project analysis utilities belong in a separate namespace to maintain clear
    boundaries between the research artifact and its development infrastructure.

Components:
    - validate: Run validation suite (format, lint, type check, tests)
    - project_info: Analyze project structure and implementation status
    - agents: Multi-agent orchestration for Claude Code task routing
    - cli: Command-line interface for development tasks

Usage:
    # Run full validation
    python -m devtools.validate

    # Get project status
    python -m devtools.project_info

    # CLI interface
    python -m devtools --help
"""

from devtools.agents import ModelRouter, TaskComplexity, TaskSpec
from devtools.project_info import ProjectAnalyzer, get_project_status
from devtools.validate import ValidationRunner, run_validation

__all__ = [
    "ValidationRunner",
    "run_validation",
    "ProjectAnalyzer",
    "get_project_status",
    "ModelRouter",
    "TaskComplexity",
    "TaskSpec",
]

__version__ = "0.1.0"
