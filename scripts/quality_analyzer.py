#!/usr/bin/env python3
"""Code quality analysis for dataset curation.

Why: Beyond security, we need to identify code quality anti-patterns.
Poor quality code should be labeled as negative examples with explanations
so the model learns what NOT to do.

Usage:
    from quality_analyzer import QualityAnalyzer

    analyzer = QualityAnalyzer()
    result = analyzer.analyze(code_content, language="python")
"""

import re
from dataclasses import dataclass, field


@dataclass
class QualityIssue:
    """Represents a detected quality issue."""
    issue_type: str
    severity: str  # "high", "medium", "low"
    location: str  # e.g., "line 42" or "function foo"
    description: str
    suggestion: str


@dataclass
class QualityResult:
    """Result of quality analysis."""
    quality_score: float = 1.0  # 0.0 to 1.0
    issues: list[QualityIssue] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    quality_label: str = "positive"

    def to_dict(self) -> dict:
        return {
            "quality_score": self.quality_score,
            "quality_label": self.quality_label,
            "issues": [
                {
                    "type": issue.issue_type,
                    "severity": issue.severity,
                    "location": issue.location,
                    "description": issue.description,
                    "suggestion": issue.suggestion,
                }
                for issue in self.issues
            ],
            "metrics": self.metrics,
        }


# Anti-patterns with explanations
PYTHON_ANTIPATTERNS: list[tuple[str, str, str, str, float]] = [
    # (pattern, name, description, suggestion, severity_score)
    (
        r"except:\s*$|except\s+Exception:\s*$",
        "bare_except",
        "Bare except catches all exceptions including KeyboardInterrupt and SystemExit.",
        "Catch specific exceptions: except ValueError: or at minimum except Exception as e: with logging.",
        0.15,
    ),
    (
        r"except\s+\w+:\s*pass\s*$",
        "silent_exception",
        "Silently swallowing exceptions hides bugs and makes debugging impossible.",
        "At minimum log the exception: except ValueError as e: logger.warning(f'Ignored: {e}')",
        0.2,
    ),
    (
        r"from\s+\w+\s+import\s+\*",
        "star_import",
        "Star imports pollute namespace and make it unclear where names come from.",
        "Import specific names: from module import func1, func2",
        0.1,
    ),
    (
        r"global\s+\w+",
        "global_variable",
        "Global variables create hidden dependencies and make code hard to test.",
        "Pass values as function parameters or use classes to encapsulate state.",
        0.1,
    ),
    (
        r"def\s+\w+\([^)]*=[{\[]\s*[}\]]",
        "mutable_default_arg",
        "Mutable default arguments (list, dict) are shared between calls, causing bugs.",
        "Use None as default and create new object in function: def foo(items=None): items = items or []",
        0.2,
    ),
    (
        r"type\s*\(\s*\w+\s*\)\s*==",
        "type_comparison",
        "Comparing types with == doesn't handle inheritance properly.",
        "Use isinstance(obj, ClassName) instead of type(obj) == ClassName",
        0.1,
    ),
    (
        r"if\s+\w+\s*==\s*(True|False):",
        "boolean_comparison",
        "Comparing to True/False is redundant and less Pythonic.",
        "Use: if condition: instead of if condition == True:",
        0.05,
    ),
    (
        r"len\s*\(\s*\w+\s*\)\s*(>|==)\s*0",
        "len_comparison",
        "Checking len() for emptiness is less Pythonic than truthy check.",
        "Use: if items: instead of if len(items) > 0:",
        0.05,
    ),
    (
        r"print\s*\([^)]*\)(?!.*#.*debug|#.*TODO)",
        "print_statement",
        "Print statements in production code should use proper logging.",
        "Use logging module: import logging; logger = logging.getLogger(__name__)",
        0.05,
    ),
    (
        r"#\s*TODO|#\s*FIXME|#\s*XXX|#\s*HACK",
        "todo_comment",
        "TODO/FIXME comments indicate incomplete or problematic code.",
        "Address the TODO or create a proper issue tracker ticket.",
        0.1,
    ),
    (
        r"time\.sleep\s*\(\s*\d+\s*\)",
        "hardcoded_sleep",
        "Hardcoded sleep values are fragile and slow down code unnecessarily.",
        "Use proper synchronization (events, conditions) or configurable timeouts.",
        0.1,
    ),
]

RUST_ANTIPATTERNS: list[tuple[str, str, str, str, float]] = [
    (
        r"\.clone\(\)\.clone\(\)",
        "double_clone",
        "Double clone is almost always unnecessary and indicates ownership issues.",
        "Review ownership model. Usually one clone or borrowing is sufficient.",
        0.15,
    ),
    (
        r"\.to_string\(\)\.to_string\(\)",
        "double_to_string",
        "Double to_string() is redundant.",
        "A single to_string() is sufficient.",
        0.1,
    ),
    (
        r"&\*",
        "unnecessary_deref_ref",
        "Dereferencing and re-referencing (&*) is usually unnecessary.",
        "Remove the &* or use reborrowing only when needed for type coercion.",
        0.05,
    ),
    (
        r"\.iter\(\)\.map\([^)]+\)\.collect::<Vec<_>>\(\)",
        "unnecessary_collect",
        "Collecting into Vec just to iterate again is inefficient.",
        "Chain iterators or use for loop directly on the iterator.",
        0.1,
    ),
    (
        r"panic!\s*\(",
        "panic_usage",
        "panic! should be reserved for unrecoverable errors, not normal error handling.",
        "Use Result<T, E> for recoverable errors. panic! only for invariant violations.",
        0.1,
    ),
    (
        r"todo!\s*\(\)|unimplemented!\s*\(",
        "incomplete_code",
        "todo! and unimplemented! indicate incomplete code.",
        "Implement the functionality or return appropriate error.",
        0.15,
    ),
    (
        r"#\[allow\(unused",
        "allow_unused",
        "Allowing unused code may indicate dead code or incomplete refactoring.",
        "Remove unused code or complete the implementation.",
        0.1,
    ),
]

# Triton (GPU kernel) quality anti-patterns
TRITON_ANTIPATTERNS: list[tuple[str, str, str, str, float]] = [
    (
        r"@triton\.jit\s*\ndef\s+\w+\([^)]{200,}\)",
        "too_many_parameters",
        "Kernels with many parameters are hard to maintain and may hit driver limits.",
        "Group related parameters into structs or use pointer-based access patterns.",
        0.15,
    ),
    (
        r"tl\.load\([^)]+\)\s*\+\s*tl\.load\([^)]+\)\s*\+\s*tl\.load",
        "sequential_loads",
        "Multiple sequential loads may not coalesce well.",
        "Use vectorized loads or restructure for better memory coalescing.",
        0.1,
    ),
    (
        r"for\s+\w+\s+in\s+range\([^)]+\):\s*\n[^@]*tl\.load",
        "loop_with_loads",
        "Loads inside Python loops may not be optimized by Triton compiler.",
        "Use tl.arange and vectorized operations instead of Python loops where possible.",
        0.1,
    ),
    (
        r"BLOCK_SIZE\s*:\s*tl\.constexpr\s*=\s*\d+[^,\n]*\n[^@]*BLOCK_SIZE\s*:\s*tl\.constexpr",
        "duplicate_constexpr",
        "Duplicate constexpr definitions can cause confusion.",
        "Define each constexpr once and pass as parameter if needed in multiple places.",
        0.05,
    ),
    (
        r"tl\.zeros\([^)]+\)\s*\n[^@]*for",
        "zeros_before_loop",
        "Initializing with tl.zeros then filling in a loop may be suboptimal.",
        "Consider using tl.load with appropriate masking instead.",
        0.05,
    ),
    (
        r"\.to\(tl\.\w+\)\.to\(tl\.\w+\)",
        "double_cast",
        "Double type casting is wasteful and can cause precision issues.",
        "Cast directly to the final type needed.",
        0.1,
    ),
    (
        r"tl\.sum\([^)]+\)\s*\n[^@]*tl\.sum",
        "multiple_reductions",
        "Multiple separate reductions may be fused into one for better performance.",
        "Consider fusing reductions or using tl.reduce with custom combine function.",
        0.1,
    ),
    (
        r"#\s*TODO|#\s*FIXME|#\s*XXX|#\s*HACK",
        "todo_comment",
        "TODO/FIXME comments indicate incomplete or problematic code.",
        "Address the TODO or create a proper issue tracker ticket.",
        0.1,
    ),
    (
        r"print\s*\(",
        "print_in_kernel",
        "Print statements don't work in Triton kernels and will be ignored.",
        "Use tl.device_print() for debugging, remove for production.",
        0.05,
    ),
    (
        r"@triton\.jit[^@]*\n\s*\"\"\"[^\"]{500,}\"\"\"",
        "excessive_docstring",
        "Very long docstrings in kernels add to compile time.",
        "Keep kernel docstrings concise; detailed docs belong in separate documentation.",
        0.05,
    ),
]


class QualityAnalyzer:
    """Analyzes code quality and detects anti-patterns.

    Why: Quality analysis helps identify code that should be labeled as
    negative examples, teaching the model what good vs bad code looks like.
    """

    def __init__(self) -> None:
        self._python_patterns = [
            (re.compile(pattern, re.MULTILINE), *rest)
            for pattern, *rest in PYTHON_ANTIPATTERNS
        ]
        self._rust_patterns = [
            (re.compile(pattern, re.MULTILINE), *rest)
            for pattern, *rest in RUST_ANTIPATTERNS
        ]
        self._triton_patterns = [
            (re.compile(pattern, re.MULTILINE), *rest)
            for pattern, *rest in TRITON_ANTIPATTERNS
        ]

    def analyze(self, content: str, language: str = "python") -> QualityResult:
        """Analyze code quality.

        Args:
            content: Source code to analyze
            language: Programming language

        Returns:
            QualityResult with score, issues, and label
        """
        result = QualityResult()
        lines = content.split("\n")

        # Basic metrics
        result.metrics = self._compute_metrics(content, lines)

        # Anti-pattern detection
        if language == "python":
            patterns = self._python_patterns
        elif language == "rust":
            patterns = self._rust_patterns
        elif language == "triton":
            patterns = self._triton_patterns
        else:
            patterns = self._python_patterns  # Default to Python

        total_penalty = 0.0
        for pattern, name, description, suggestion, penalty in patterns:
            matches = list(pattern.finditer(content))
            for match in matches:
                line_num = content[:match.start()].count("\n") + 1
                result.issues.append(QualityIssue(
                    issue_type=name,
                    severity="high" if penalty > 0.15 else "medium" if penalty > 0.08 else "low",
                    location=f"line {line_num}",
                    description=description,
                    suggestion=suggestion,
                ))
                total_penalty += penalty

        # Metric-based penalties
        metrics = result.metrics

        # Complexity penalty
        if metrics.get("avg_line_length", 0) > 100:
            total_penalty += 0.1
            result.issues.append(QualityIssue(
                issue_type="long_lines",
                severity="low",
                location="multiple",
                description="Average line length exceeds 100 characters.",
                suggestion="Break long lines for readability. Use intermediate variables.",
            ))

        # Deep nesting penalty (rough heuristic)
        max_indent = max(
            (len(line) - len(line.lstrip())) // 4
            for line in lines
            if line.strip()
        ) if lines else 0

        if max_indent > 5:
            total_penalty += 0.15
            result.issues.append(QualityIssue(
                issue_type="deep_nesting",
                severity="high",
                location="multiple",
                description=f"Code has deep nesting (max indent level: {max_indent}).",
                suggestion="Extract nested logic into separate functions. Use early returns.",
            ))

        # Long file penalty
        if metrics.get("total_lines", 0) > 500:
            total_penalty += 0.1
            result.issues.append(QualityIssue(
                issue_type="long_file",
                severity="medium",
                location="file",
                description=f"File has {metrics['total_lines']} lines.",
                suggestion="Consider splitting into multiple modules for maintainability.",
            ))

        # Compute final score
        result.quality_score = max(0.0, 1.0 - total_penalty)

        # Set quality label
        if result.quality_score < 0.5:
            result.quality_label = "negative"
        elif result.quality_score < 0.7:
            result.quality_label = "positive"  # Still usable but not exemplary
        else:
            result.quality_label = "positive"

        return result

    def _compute_metrics(self, content: str, lines: list[str]) -> dict:
        """Compute basic code metrics."""
        non_empty_lines = [l for l in lines if l.strip()]

        return {
            "total_lines": len(lines),
            "code_lines": len(non_empty_lines),
            "blank_lines": len(lines) - len(non_empty_lines),
            "avg_line_length": (
                sum(len(l) for l in non_empty_lines) / len(non_empty_lines)
                if non_empty_lines else 0
            ),
            "max_line_length": max((len(l) for l in lines), default=0),
            "comment_lines": sum(1 for l in lines if l.strip().startswith("#")),
        }

    def format_negative_example(self, content: str, result: QualityResult) -> dict:
        """Format negative example with quality explanations."""
        explanations = []
        for issue in result.issues:
            explanations.append(
                f"{issue.location}: {issue.issue_type} - {issue.description} "
                f"Suggestion: {issue.suggestion}"
            )

        return {
            "text": content,
            "quality_label": "negative",
            "quality_score": result.quality_score,
            "anti_patterns": [issue.issue_type for issue in result.issues],
            "explanation": "\n".join(explanations),
            "metrics": result.metrics,
        }


if __name__ == "__main__":
    # Demo/test
    test_code_good = '''
def calculate_total(items: list[float]) -> float:
    """Calculate the total of a list of prices.

    Args:
        items: List of prices

    Returns:
        Sum of all prices
    """
    if not items:
        return 0.0
    return sum(items)
'''

    test_code_bad = '''
from os import *

def process(data):
    global result
    try:
        for i in range(len(data)):
            if type(data[i]) == str:
                result = eval(data[i])
            else:
                result = data[i]
    except:
        pass
    print(result)
    return result
'''

    analyzer = QualityAnalyzer()

    print("=== Good code ===")
    result = analyzer.analyze(test_code_good)
    print(f"Score: {result.quality_score:.2f}")
    print(f"Label: {result.quality_label}")
    print(f"Issues: {len(result.issues)}")

    print("\n=== Bad code ===")
    result = analyzer.analyze(test_code_bad)
    print(f"Score: {result.quality_score:.2f}")
    print(f"Label: {result.quality_label}")
    print(f"Issues: {len(result.issues)}")
    for issue in result.issues:
        print(f"  - {issue.issue_type}: {issue.description[:50]}...")
