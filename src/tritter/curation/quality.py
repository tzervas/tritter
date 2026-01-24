"""
Code quality analysis for training data curation.

Performs structural analysis of code to compute quality metrics without requiring
external tools. Uses Python's ast module for parsing and analysis.

Why: Quality metrics enable automated labeling of training data:
- High-quality code becomes positive examples (model learns to emulate)
- Poor-quality code becomes negative examples with explanations
- Metrics are objective and reproducible across the dataset

Metrics Computed:
- Function/class count (structural complexity)
- Maximum nesting depth (readability indicator)
- Lines per function (function size)
- Docstring presence ratio (documentation quality)
- Magic number detection (hardcoded constants)
- Long line detection (style violations)

Design Decision: We use Python's built-in ast module rather than external tools
(radon, pylint) to avoid dependencies and enable offline operation. The metrics
are simpler but sufficient for quality gating. For more sophisticated analysis,
integrate external tools at the pipeline level.
"""

import ast
import re
from dataclasses import dataclass, field

__all__ = [
    "QualityMetrics",
    "QualityIssue",
    "QualityAnalyzer",
]


@dataclass
class QualityIssue:
    """A code quality issue found during analysis.

    Why: Structured issue objects enable consistent reporting and aggregation.
    Each issue has a type, location, and explanation that can be used to
    generate training sample explanations.

    Attributes:
        issue_type: Category of quality issue (e.g., "magic_number", "deep_nesting")
        line: Line number where issue was found (1-indexed)
        message: Description of the issue
        severity: How bad the issue is ("minor", "moderate", "major")
    """

    issue_type: str
    line: int
    message: str
    severity: str = "moderate"


@dataclass
class QualityMetrics:
    """Computed quality metrics for a code sample.

    Contains numerical and structural metrics that quantify code quality.
    Used to determine the quality_label for training samples.

    Why: Numerical metrics enable:
    1. Threshold-based labeling (e.g., quality_score < 0.5 -> negative)
    2. Trend analysis across datasets
    3. Correlation with model performance

    Attributes:
        line_count: Total lines of code
        function_count: Number of function definitions
        class_count: Number of class definitions
        max_nesting_depth: Deepest level of nested blocks
        avg_function_lines: Average lines per function
        max_function_lines: Longest function in lines
        docstring_ratio: Fraction of functions/classes with docstrings [0.0, 1.0]
        magic_number_count: Count of hardcoded numeric literals
        long_line_count: Lines exceeding length threshold
        issues: List of detected quality issues
        overall_score: Computed quality score [0.0, 1.0]
    """

    line_count: int = 0
    function_count: int = 0
    class_count: int = 0
    max_nesting_depth: int = 0
    avg_function_lines: float = 0.0
    max_function_lines: int = 0
    docstring_ratio: float = 0.0
    magic_number_count: int = 0
    long_line_count: int = 0
    issues: list[QualityIssue] = field(default_factory=list)
    overall_score: float = 1.0


class QualityAnalyzer:
    """Analyzer for code quality metrics.

    Performs structural analysis using AST parsing to compute quality metrics.
    Supports Python code directly; other languages receive basic line-based analysis.

    Why: AST-based analysis catches structural issues that simple regex cannot:
    - Actual nesting depth (not just indentation)
    - Function boundaries for size calculation
    - Docstring presence detection
    - Magic number detection in code context (not comments)

    The analyzer is configurable with thresholds for what constitutes issues:
    - max_line_length: Lines longer than this trigger issues (default 120)
    - max_nesting_depth: Nesting deeper than this triggers issues (default 5)
    - max_function_lines: Functions longer than this trigger issues (default 100)
    - magic_number_threshold: Numbers larger than this are flagged (default 10)

    Usage:
        analyzer = QualityAnalyzer()
        metrics = analyzer.analyze(code, language="python")
        if metrics.overall_score < 0.5:
            # Label as negative
            pass
    """

    def __init__(
        self,
        max_line_length: int = 120,
        max_nesting_depth: int = 5,
        max_function_lines: int = 100,
        magic_number_threshold: int = 10,
    ) -> None:
        """Initialize quality analyzer with thresholds.

        Why: Configurable thresholds allow adaptation to different coding standards.
        Defaults are based on common style guides (PEP 8, Google style).

        Args:
            max_line_length: Maximum acceptable line length
            max_nesting_depth: Maximum acceptable nesting depth
            max_function_lines: Maximum acceptable function size
            magic_number_threshold: Numbers above this are considered magic
        """
        self.max_line_length = max_line_length
        self.max_nesting_depth = max_nesting_depth
        self.max_function_lines = max_function_lines
        self.magic_number_threshold = magic_number_threshold

    def analyze(self, code: str, language: str) -> QualityMetrics:
        """Analyze code and compute quality metrics.

        Parses code using language-appropriate methods and computes structural
        metrics. Falls back to basic line analysis for unsupported languages.

        Why: Language-specific analysis extracts more meaningful metrics. Python
        gets full AST analysis; other languages get basic heuristics that are
        still useful for quality gating.

        Args:
            code: Source code to analyze
            language: Programming language ("python", "rust", "triton", etc.)

        Returns:
            QualityMetrics with computed values and detected issues
        """
        language = language.lower()

        # Basic line-level metrics (all languages)
        lines = code.split("\n")
        metrics = QualityMetrics(line_count=len(lines))

        # Detect long lines
        metrics.long_line_count = self._count_long_lines(lines)

        if language == "python":
            # Full AST analysis for Python
            try:
                tree = ast.parse(code)
                self._analyze_python_ast(tree, code, metrics)
            except SyntaxError:
                # Fallback to basic analysis if parsing fails
                self._basic_analysis(lines, metrics)
        else:
            # Basic heuristic analysis for other languages
            self._basic_analysis(lines, metrics)

        # Compute overall score
        metrics.overall_score = self._compute_score(metrics)

        return metrics

    def _count_long_lines(self, lines: list[str]) -> int:
        """Count lines exceeding max length threshold.

        Why: Long lines indicate style issues and may hurt readability.
        Common in autogenerated code or poorly formatted manual code.

        Args:
            lines: List of code lines

        Returns:
            Count of lines exceeding max_line_length
        """
        count = 0
        for line in lines:
            if len(line) > self.max_line_length:
                count += 1
        return count

    def _analyze_python_ast(self, tree: ast.AST, code: str, metrics: QualityMetrics) -> None:
        """Perform full AST analysis on Python code.

        Why: AST analysis provides accurate structural metrics:
        - Function definitions are properly identified
        - Nesting depth is computed from actual control flow, not indentation
        - Docstrings are found by examining first statement in function body
        - Magic numbers are identified in code context, not comments

        Args:
            tree: Parsed AST
            code: Original source code (for line-based lookups)
            metrics: QualityMetrics object to populate
        """
        functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        classes: list[ast.ClassDef] = []

        # Collect all functions and classes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node)
            elif isinstance(node, ast.ClassDef):
                classes.append(node)

        metrics.function_count = len(functions)
        metrics.class_count = len(classes)

        # Initialize counters
        function_lines: list[int] = []
        docstrings_found = 0
        class_docstrings = 0

        # Analyze function sizes and docstrings
        for func in functions:
            # Calculate function line count
            if hasattr(func, "end_lineno") and func.end_lineno is not None:
                lines = func.end_lineno - func.lineno + 1
            else:
                # Estimate from body
                lines = len(func.body)
            function_lines.append(lines)

            # Check for docstring
            if func.body and isinstance(func.body[0], ast.Expr):
                if isinstance(func.body[0].value, ast.Constant):
                    if isinstance(func.body[0].value.value, str):
                        docstrings_found += 1

            # Flag long functions
            if lines > self.max_function_lines:
                metrics.issues.append(
                    QualityIssue(
                        issue_type="long_function",
                        line=func.lineno,
                        message=f"Function '{func.name}' has {lines} lines (max: {self.max_function_lines})",
                        severity="moderate",
                    )
                )

        if function_lines:
            metrics.max_function_lines = max(function_lines)
            metrics.avg_function_lines = sum(function_lines) / len(function_lines)

        # Check class docstrings
        for cls in classes:
            if cls.body and isinstance(cls.body[0], ast.Expr):
                if isinstance(cls.body[0].value, ast.Constant):
                    if isinstance(cls.body[0].value.value, str):
                        class_docstrings += 1

        # Calculate docstring ratio
        total_definitions = len(functions) + len(classes)
        total_docstrings = docstrings_found + class_docstrings
        metrics.docstring_ratio = (
            total_docstrings / total_definitions if total_definitions > 0 else 0.0
        )

        # Calculate max nesting depth
        metrics.max_nesting_depth = self._calculate_nesting_depth(tree)

        # Flag deep nesting
        if metrics.max_nesting_depth > self.max_nesting_depth:
            metrics.issues.append(
                QualityIssue(
                    issue_type="deep_nesting",
                    line=1,  # Hard to pinpoint exact line
                    message=f"Maximum nesting depth is {metrics.max_nesting_depth} (max: {self.max_nesting_depth})",
                    severity="moderate",
                )
            )

        # Detect magic numbers
        metrics.magic_number_count = self._count_magic_numbers(tree)

        if metrics.magic_number_count > 5:
            metrics.issues.append(
                QualityIssue(
                    issue_type="magic_numbers",
                    line=1,
                    message=f"Found {metrics.magic_number_count} magic numbers; consider using named constants",
                    severity="minor",
                )
            )

        # Add issues for missing docstrings if ratio is low
        if metrics.docstring_ratio < 0.5 and total_definitions > 0:
            metrics.issues.append(
                QualityIssue(
                    issue_type="missing_docstrings",
                    line=1,
                    message=f"Only {metrics.docstring_ratio:.0%} of definitions have docstrings",
                    severity="minor",
                )
            )

    def _calculate_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth of control structures.

        Why: Deep nesting hurts readability and often indicates code that
        should be refactored into separate functions. Common threshold is 4-5.

        Args:
            tree: AST to analyze

        Returns:
            Maximum nesting depth found
        """
        max_depth = 0

        def visit(node: ast.AST, depth: int) -> None:
            nonlocal max_depth

            # Nodes that increase nesting depth
            nesting_nodes = (
                ast.If,
                ast.For,
                ast.While,
                ast.With,
                ast.Try,
                ast.ExceptHandler,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            )

            new_depth = depth
            if isinstance(node, nesting_nodes):
                new_depth = depth + 1
                max_depth = max(max_depth, new_depth)

            for child in ast.iter_child_nodes(node):
                visit(child, new_depth)

        visit(tree, 0)
        return max_depth

    def _count_magic_numbers(self, tree: ast.AST) -> int:
        """Count magic numbers (hardcoded numeric literals) in code.

        Excludes common acceptable values: 0, 1, 2, -1, and numbers in
        certain contexts (list indices, small loop bounds).

        Why: Magic numbers make code harder to understand and maintain.
        Named constants explain the meaning of values.

        Args:
            tree: AST to analyze

        Returns:
            Count of magic numbers found
        """
        # Acceptable magic numbers (common programming constants)
        acceptable = {0, 1, 2, -1, 10, 100, 1000}

        count = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                    # Skip small values and common constants
                    if (
                        abs(node.value) > self.magic_number_threshold
                        and node.value not in acceptable
                    ):
                        count += 1

        return count

    def _basic_analysis(self, lines: list[str], metrics: QualityMetrics) -> None:
        """Perform basic heuristic analysis for non-Python code.

        Uses regex patterns to estimate metrics without proper parsing.
        Less accurate than AST but provides useful signal.

        Why: Even basic analysis is better than nothing for quality gating.
        Function-like patterns (def, fn, function) identify definitions;
        indentation patterns estimate nesting.

        Args:
            lines: Code lines to analyze
            metrics: QualityMetrics object to populate
        """
        # Count function-like patterns
        function_pattern = re.compile(r"^\s*(def|fn|func|function|pub\s+fn|async\s+fn)\s+\w+")
        class_pattern = re.compile(r"^\s*(class|struct|enum|impl|trait)\s+\w+")

        functions = 0
        classes = 0

        for line in lines:
            if function_pattern.match(line):
                functions += 1
            if class_pattern.match(line):
                classes += 1

        metrics.function_count = functions
        metrics.class_count = classes

        # Estimate nesting from indentation
        max_indent = 0
        for line in lines:
            if line.strip():  # Non-empty lines
                indent = len(line) - len(line.lstrip())
                # Assume 4-space or 2-space indent
                depth = indent // 4 if indent % 4 == 0 else indent // 2
                max_indent = max(max_indent, depth)

        metrics.max_nesting_depth = max_indent

        if max_indent > self.max_nesting_depth:
            metrics.issues.append(
                QualityIssue(
                    issue_type="deep_nesting",
                    line=1,
                    message=f"Estimated nesting depth {max_indent} exceeds threshold {self.max_nesting_depth}",
                    severity="moderate",
                )
            )

        # Count magic numbers with regex (less accurate)
        number_pattern = re.compile(r"\b\d{3,}\b")  # Numbers with 3+ digits
        magic_count = 0
        for line in lines:
            # Skip comments
            if "#" in line:
                line = line[: line.index("#")]
            if "//" in line:
                line = line[: line.index("//")]
            magic_count += len(number_pattern.findall(line))

        metrics.magic_number_count = magic_count

        # Estimate docstring ratio (very rough for non-Python)
        doc_pattern = re.compile(r'^\s*("""|\'\'\'|///|/\*\*)')
        doc_count = sum(1 for line in lines if doc_pattern.match(line))
        total_defs = functions + classes
        metrics.docstring_ratio = min(doc_count / max(total_defs, 1), 1.0)

    def _compute_score(self, metrics: QualityMetrics) -> float:
        """Compute overall quality score from individual metrics.

        The score is a weighted combination of factors that indicate quality.
        Higher is better, range is [0.0, 1.0].

        Why: A single score simplifies threshold-based labeling while
        preserving individual metrics for detailed analysis.

        Scoring factors:
        - Docstring ratio (25%): Good documentation is important
        - Nesting depth penalty (25%): Deep nesting hurts readability
        - Long lines penalty (15%): Style violations
        - Magic numbers penalty (15%): Hardcoded constants
        - Function size penalty (20%): Long functions are hard to maintain

        Args:
            metrics: Computed quality metrics

        Returns:
            Overall quality score [0.0, 1.0]
        """
        score = 1.0

        # Docstring ratio contribution (0-25%)
        score *= 0.75 + (0.25 * metrics.docstring_ratio)

        # Nesting depth penalty
        if metrics.max_nesting_depth > self.max_nesting_depth:
            excess = metrics.max_nesting_depth - self.max_nesting_depth
            score *= max(0.5, 1.0 - (0.1 * excess))

        # Long lines penalty (max 15% reduction)
        if metrics.line_count > 0:
            long_ratio = metrics.long_line_count / metrics.line_count
            score *= 1.0 - (0.15 * min(long_ratio, 1.0))

        # Magic numbers penalty (max 15% reduction)
        if metrics.magic_number_count > 5:
            excess = metrics.magic_number_count - 5
            score *= max(0.85, 1.0 - (0.03 * excess))

        # Function size penalty
        if metrics.max_function_lines > self.max_function_lines:
            excess_ratio = (
                metrics.max_function_lines - self.max_function_lines
            ) / self.max_function_lines
            score *= max(0.8, 1.0 - (0.2 * min(excess_ratio, 1.0)))

        # Issue count penalty
        major_issues = sum(1 for issue in metrics.issues if issue.severity == "major")
        moderate_issues = sum(1 for issue in metrics.issues if issue.severity == "moderate")
        score *= max(0.7, 1.0 - (0.1 * major_issues) - (0.05 * moderate_issues))

        return max(0.0, min(1.0, score))
