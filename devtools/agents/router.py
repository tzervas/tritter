"""Model router for intelligent task delegation.

Routes tasks to appropriate models based on complexity classification.
Uses Claude Code's native Task tool with model parameter for execution.

Why: Cost optimization through intelligent routing
- 60x cost difference between Opus and Haiku
- Match model capability to task requirements
- Parallel execution for independent tasks
"""

from dataclasses import dataclass
from typing import Literal

ModelTier = Literal["opus", "sonnet", "haiku"]
SubagentType = Literal["Bash", "general-purpose", "Explore", "Plan"]


@dataclass
class ModelRouter:
    """Routes tasks to appropriate models and subagents.

    Why: Centralized routing logic ensures consistent model selection
    and makes cost/capability tradeoffs explicit.

    Usage (conceptual - actual execution via Claude Code Task tool):
        router = ModelRouter()
        model, subagent = router.route(task_type="bug_fix", scope="single_file")
        # Returns: ("haiku", "general-purpose")
    """

    # Cost per million tokens (USD)
    COSTS = {
        "opus": {"input": 15.0, "output": 75.0},
        "sonnet": {"input": 3.0, "output": 15.0},
        "haiku": {"input": 0.25, "output": 1.25},
    }

    # Capability ratings (1-10)
    CAPABILITIES = {
        "opus": {"reasoning": 10, "coding": 10, "planning": 10},
        "sonnet": {"reasoning": 8, "coding": 9, "planning": 7},
        "haiku": {"reasoning": 6, "coding": 7, "planning": 5},
    }

    def route(
        self,
        task_type: str,
        scope: str = "medium",
        requires_research: bool = False,
    ) -> tuple[ModelTier, SubagentType]:
        """Route task to appropriate model and subagent.

        Args:
            task_type: Type of task (bug_fix, implement, design, test, docs)
            scope: Scope of changes (single_file, module, system)
            requires_research: Whether task needs external research

        Returns:
            Tuple of (model_tier, subagent_type)

        Why: Deterministic routing based on task characteristics ensures
        reproducible cost optimization and appropriate capability matching.
        """
        # Research tasks need Opus for synthesis
        if requires_research:
            return ("opus", "Plan")

        # Route by task type and scope
        routing_table = {
            # Bug fixes
            ("bug_fix", "single_file"): ("haiku", "general-purpose"),
            ("bug_fix", "module"): ("sonnet", "general-purpose"),
            ("bug_fix", "system"): ("sonnet", "general-purpose"),
            # Implementation
            ("implement", "single_file"): ("haiku", "general-purpose"),
            ("implement", "module"): ("sonnet", "general-purpose"),
            ("implement", "system"): ("opus", "Plan"),
            # Design/Architecture
            ("design", "single_file"): ("sonnet", "Plan"),
            ("design", "module"): ("opus", "Plan"),
            ("design", "system"): ("opus", "Plan"),
            # Testing
            ("test", "single_file"): ("haiku", "Bash"),
            ("test", "module"): ("sonnet", "general-purpose"),
            ("test", "system"): ("sonnet", "general-purpose"),
            # Documentation
            ("docs", "single_file"): ("haiku", "general-purpose"),
            ("docs", "module"): ("sonnet", "general-purpose"),
            ("docs", "system"): ("sonnet", "general-purpose"),
            # Exploration
            ("explore", "single_file"): ("haiku", "Explore"),
            ("explore", "module"): ("haiku", "Explore"),
            ("explore", "system"): ("sonnet", "Explore"),
        }

        return routing_table.get(
            (task_type, scope),
            ("sonnet", "general-purpose"),  # Default fallback
        )

    def estimate_cost(
        self,
        model: ModelTier,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost for a task.

        Args:
            model: Model tier to use
            input_tokens: Estimated input tokens
            output_tokens: Estimated output tokens

        Returns:
            Estimated cost in USD

        Why: Cost awareness enables informed decisions about model routing
        and helps track development costs over time.
        """
        costs = self.COSTS[model]
        return (input_tokens / 1_000_000) * costs["input"] + (output_tokens / 1_000_000) * costs[
            "output"
        ]


# === Parallel Execution Patterns ===
"""
Claude Code supports parallel Task execution for independent tasks.

Pattern 1: Fan-out (parallel independent tasks)
```
# Send multiple Tasks in single message for parallel execution
Task(model="haiku", prompt="Fix lint in module A", ...)
Task(model="haiku", prompt="Fix lint in module B", ...)
Task(model="haiku", prompt="Fix lint in module C", ...)
```

Pattern 2: Sequential with dependencies
```
# Task 1: Design (Opus)
result1 = Task(model="opus", prompt="Design attention module", ...)

# Task 2: Implement (Sonnet) - depends on Task 1
result2 = Task(model="sonnet", prompt=f"Implement based on: {result1}", ...)

# Task 3: Test (Haiku) - depends on Task 2
result3 = Task(model="haiku", prompt="Run tests", ...)
```

Pattern 3: Background tasks
```
# Start long-running task in background
Task(model="sonnet", prompt="Generate comprehensive docs", run_in_background=True)

# Continue with other work
Task(model="haiku", prompt="Quick fix", ...)

# Check background task later via Read tool on output_file
```
"""


# === Gemini Integration Notes ===
"""
Google Gemini can be integrated via API for specialized subtasks.

Configuration (add to environment or .env):
    GOOGLE_API_KEY=your_api_key
    GEMINI_MODEL=gemini-2.0-flash-exp

Usage pattern:
```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Use for research-heavy subtasks
response = model.generate_content(
    "Summarize the latest FlexAttention benchmarks..."
)
```

Routing decision:
- Use Gemini for: Web research, summarization, quick Q&A
- Use Claude for: Code generation, architecture, complex reasoning
"""
