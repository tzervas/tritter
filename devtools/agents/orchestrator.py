"""Multi-agent orchestrator using Claude Code native Task tool.

This module documents the orchestration patterns for Tritter development.
Actual execution happens via Claude Code's Task tool with model routing.

Why: Token and cost optimization through intelligent task routing:
- Opus ($15/M input, $75/M output): Architecture, planning, research
- Sonnet ($3/M input, $15/M output): Implementation, tests, docs
- Haiku ($0.25/M input, $1.25/M output): Fixes, configs, focused tasks

Usage Pattern (in Claude Code):
    # Opus for planning
    Task(subagent_type="Plan", model="opus", prompt="Design attention architecture...")

    # Sonnet for implementation
    Task(subagent_type="general-purpose", model="sonnet", prompt="Implement FlexAttention...")

    # Haiku for focused fixes
    Task(subagent_type="Bash", model="haiku", prompt="Run tests and fix lint...")
"""

from dataclasses import dataclass
from enum import Enum


class TaskComplexity(Enum):
    """Task complexity classification for model routing.

    Why: Match model capability to task requirements for cost efficiency.
    Opus is 60x more expensive than Haiku - use it only when needed.
    """

    # Haiku-tier: Fast, focused, simple
    TRIVIAL = "haiku"  # Single-line fixes, formatting
    SIMPLE = "haiku"  # Config changes, small refactors
    FOCUSED = "haiku"  # Bug fixes with clear scope

    # Sonnet-tier: Balanced capability/cost
    MODERATE = "sonnet"  # Module implementation
    COMPLEX = "sonnet"  # Test suite development
    DOCUMENTATION = "sonnet"  # Technical writing

    # Opus-tier: Maximum capability
    ARCHITECTURE = "opus"  # System design decisions
    PLANNING = "opus"  # Multi-phase implementation plans
    RESEARCH = "opus"  # Synthesis of multiple sources


@dataclass
class TaskSpec:
    """Specification for an orchestrated task.

    Attributes:
        description: Short task description (3-5 words)
        prompt: Full task prompt with context
        complexity: TaskComplexity for model routing
        subagent_type: Claude Code subagent type
        dependencies: Task IDs this depends on
        run_in_background: Whether to run async
    """

    description: str
    prompt: str
    complexity: TaskComplexity
    subagent_type: str = "general-purpose"
    dependencies: list[str] | None = None
    run_in_background: bool = False

    @property
    def model(self) -> str:
        """Get model name from complexity."""
        return self.complexity.value


# === Task Templates for Tritter Development ===

TASK_TEMPLATES = {
    "fix_flashattention": TaskSpec(
        description="Fix FlashAttention is_causal",
        prompt="""Fix the FlashAttention causal mask inefficiency in architecture.py.

Current problem (line 132):
- Uses is_causal=False with explicit mask
- Prevents FlashAttention-2 kernel optimization

Required changes:
1. When attention_mask is None, use is_causal=True (no mask)
2. When attention_mask is provided, use is_causal=False with mask
3. Update docstrings to explain the optimization
4. Verify tests still pass

Reference: docs/tritter-comprehensive-implementation-plan.md Part 1""",
        complexity=TaskComplexity.FOCUSED,
        subagent_type="general-purpose",
    ),
    "implement_flex_attention": TaskSpec(
        description="Implement FlexAttention module",
        prompt="""Create FlexAttention integration module.

Create src/tritter/models/flex_attention.py with:
1. Mask primitives: causal, sliding_window, document, streamingllm, prefix_lm
2. Composite mask creation with BlockMask caching
3. FlexAttentionLayer class wrapping torch.nn.attention.flex_attention

Follow the implementation plan in:
docs/tritter-comprehensive-implementation-plan.md Part 2

Include comprehensive docstrings with "Why" explanations per DEVELOPMENT_STANDARDS.md""",
        complexity=TaskComplexity.COMPLEX,
        subagent_type="general-purpose",
    ),
    "add_attention_config": TaskSpec(
        description="Add attention mode config",
        prompt="""Add attention configuration options to TritterConfig.

Add fields:
- attention_mode: str = "causal" (causal, bidirectional, prefix_lm, embedding)
- sliding_window_size: int = 4096
- use_sliding_window: bool = True
- use_attention_sinks: bool = False
- num_sink_tokens: int = 4
- prefix_length: int = 0

Add validation in __post_init__.
Include docstrings explaining each option per DEVELOPMENT_STANDARDS.md.

Reference: docs/tritter-comprehensive-implementation-plan.md Part 3""",
        complexity=TaskComplexity.MODERATE,
        subagent_type="general-purpose",
    ),
    "design_kv_cache": TaskSpec(
        description="Design KV-cache quantization",
        prompt="""Design INT4 KV-cache quantization strategy for 128K context.

Research and plan:
1. Review KIVI paper approach (per-channel keys, per-token values)
2. Calculate memory savings: FP16 vs INT4 at various context lengths
3. Design integration with sliding window attention
4. Plan PagedAttention integration for memory management
5. Define implementation phases

Target: 128K context in ~8GB KV-cache on RTX 5080 16GB

Output: Detailed implementation plan with code structure""",
        complexity=TaskComplexity.ARCHITECTURE,
        subagent_type="Plan",
    ),
    "run_tests": TaskSpec(
        description="Run test suite",
        prompt="Run pytest and report results. Fix any failures.",
        complexity=TaskComplexity.TRIVIAL,
        subagent_type="Bash",
    ),
    "lint_and_format": TaskSpec(
        description="Lint and format code",
        prompt="Run ruff check and ruff format. Fix any issues.",
        complexity=TaskComplexity.TRIVIAL,
        subagent_type="Bash",
    ),
}


def get_phase2_tasks() -> list[TaskSpec]:
    """Get all Phase 2 tasks in execution order.

    Returns:
        List of TaskSpecs for Phase 2 implementation
    """
    return [
        TASK_TEMPLATES["fix_flashattention"],
        TASK_TEMPLATES["add_attention_config"],
        TASK_TEMPLATES["implement_flex_attention"],
        TASK_TEMPLATES["run_tests"],
        TASK_TEMPLATES["lint_and_format"],
    ]


# === Google ADK Integration Notes ===
"""
Google ADK (Agent Development Kit) can be integrated as an alternative backend.

ADK provides:
- Agent blueprints with defined tools and capabilities
- Multi-agent workflows with handoffs
- Built-in Gemini model access

Integration pattern:
1. Define ADK agents matching our TaskComplexity tiers
2. Use ADK for Gemini-powered subtasks
3. Keep Claude Code as primary orchestrator

Example ADK agent definition:
```python
from google.adk import Agent, Tool

@Agent(
    name="tritter_implementer",
    model="gemini-2.0-flash",
    tools=[Tool.code_execution, Tool.file_edit]
)
def implementation_agent(task: str) -> str:
    '''Implement code changes for Tritter.'''
    pass
```

For now, Claude Code's native Task tool provides equivalent functionality.
ADK integration can be added when Gemini-specific capabilities are needed.
"""
