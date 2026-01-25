"""
Main curation pipeline for training data processing.

Combines all scanners (secrets, security, quality) into a unified pipeline that
processes code samples and produces quality-labeled training data.

Why: A unified pipeline ensures consistent processing of all training data:
1. Secrets are ALWAYS rejected (never train on secrets)
2. Security issues result in negative labels with explanations
3. Quality issues result in negative labels with explanations
4. High-quality, secure code becomes positive examples

The pipeline implements the processing stages from SPEC-007:
    Raw Code
        |
        v
    Secret Scanner  --REJECT--> /dev/null (never train on secrets)
        |
        | pass
        v
    Security Scan   --issues--> Label as NEGATIVE + explanation
        |
        | clean
        v
    Quality Metrics --poor--> Label as NEGATIVE + issues
        |
        | good
        v
    Label POSITIVE
    Add metadata

Contrastive Learning: Negative examples are valuable - they teach the model
to identify and explain bad code. The explanation field is critical for this.
"""

from dataclasses import dataclass
from typing import Literal

from .quality import QualityAnalyzer, QualityMetrics
from .schema import CuratedSample, QualityMetadata, SourceMetadata
from .secrets import SecretMatch, SecretScanner
from .security import SecurityIssue, SecurityScanner

__all__ = [
    "CurationResult",
    "CurationPipeline",
]


@dataclass
class CurationResult:
    """Result of processing a code sample through the curation pipeline.

    Contains the quality determination and all supporting information needed
    to create a CuratedSample for training.

    Why: Separating the result from the final sample format allows callers to
    inspect intermediate results (security issues, quality metrics) before
    committing to a final label. This enables custom post-processing logic.

    Attributes:
        quality_label: Final quality classification
        quality_score: Numerical quality score [0.0, 1.0]
        security_issues: List of security vulnerabilities found
        quality_issues: List of quality problems detected
        explanation: Human-readable explanation of the determination
        rejected_reason: If rejected, why (e.g., "contains_secrets")
        secret_matches: Details of any secrets found
        quality_metrics: Full quality metrics object
    """

    quality_label: Literal["positive", "negative", "rejected"]
    quality_score: float
    security_issues: list[SecurityIssue]
    quality_issues: list[str]
    explanation: str | None
    rejected_reason: str | None
    secret_matches: list[SecretMatch]
    quality_metrics: QualityMetrics | None


class CurationPipeline:
    """Unified pipeline for curating training data.

    Processes code samples through secret detection, security scanning, and
    quality analysis to produce labeled training data.

    Why: Centralizing curation logic ensures:
    1. Consistent processing across all data
    2. Correct ordering (secrets first, then security, then quality)
    3. Proper explanation generation for negative examples
    4. Configurable thresholds for different use cases

    Processing Order (critical):
    1. Secret detection - REJECT immediately (highest priority)
    2. Security scan - Label as negative if issues found
    3. Quality analysis - Label as negative if score below threshold
    4. All passed - Label as positive

    Usage:
        pipeline = CurationPipeline()
        result = pipeline.process(code, language="python", source_metadata={...})

        if result.quality_label == "rejected":
            # Never include in training
            pass
        elif result.quality_label == "negative":
            # Include as negative example with explanation
            sample = pipeline.create_sample(code, language, result, source_metadata)
        else:
            # Include as positive example
            sample = pipeline.create_sample(code, language, result, source_metadata)
    """

    def __init__(
        self,
        quality_threshold: float = 0.5,
        reject_on_critical_security: bool = False,
        min_lines: int = 5,
        max_lines: int = 10000,
    ) -> None:
        """Initialize curation pipeline.

        Why: Configurable thresholds allow adaptation to different data sources
        and quality requirements. Strict thresholds produce fewer but higher
        quality samples; loose thresholds maximize data volume.

        Args:
            quality_threshold: Minimum quality score for positive label [0.0, 1.0]
            reject_on_critical_security: If True, critical security issues cause
                rejection instead of negative labeling
            min_lines: Minimum lines for a valid sample (too short = not useful)
            max_lines: Maximum lines for a valid sample (too long = memory issues)
        """
        self.quality_threshold = quality_threshold
        self.reject_on_critical_security = reject_on_critical_security
        self.min_lines = min_lines
        self.max_lines = max_lines

        # Initialize scanners
        self.secret_scanner = SecretScanner()
        self.security_scanner = SecurityScanner()
        self.quality_analyzer = QualityAnalyzer()

    def process(
        self,
        code: str,
        language: str,
        source_metadata: dict | None = None,
    ) -> CurationResult:
        """Process a code sample through the curation pipeline.

        Applies all quality gates in order: secrets -> security -> quality.
        Returns a result with the quality determination and supporting data.

        Why: This is the main entry point. Processing order is critical:
        1. Secrets must be detected first (highest priority rejection)
        2. Security issues inform negative labeling
        3. Quality metrics provide the final determination for clean code

        Args:
            code: Source code to process
            language: Programming language ("python", "rust", "triton")
            source_metadata: Optional metadata about code source

        Returns:
            CurationResult with quality label and supporting data
        """
        # Initialize result
        security_issues: list[SecurityIssue] = []
        quality_issues: list[str] = []
        explanation: str | None = None
        rejected_reason: str | None = None
        quality_metrics: QualityMetrics | None = None

        # Stage 0: Basic validation
        lines = code.split("\n")
        line_count = len(lines)

        if line_count < self.min_lines:
            return CurationResult(
                quality_label="rejected",
                quality_score=0.0,
                security_issues=[],
                quality_issues=[f"Too short: {line_count} lines (min: {self.min_lines})"],
                explanation=None,
                rejected_reason="too_short",
                secret_matches=[],
                quality_metrics=None,
            )

        if line_count > self.max_lines:
            return CurationResult(
                quality_label="rejected",
                quality_score=0.0,
                security_issues=[],
                quality_issues=[f"Too long: {line_count} lines (max: {self.max_lines})"],
                explanation=None,
                rejected_reason="too_long",
                secret_matches=[],
                quality_metrics=None,
            )

        # Stage 1: Secret detection (ALWAYS REJECT)
        secret_matches = self.secret_scanner.scan(code)
        if secret_matches:
            rejected_reason = "contains_secrets"
            pattern_names = list({m.pattern_name for m in secret_matches})
            explanation = f"Rejected: contains secrets ({', '.join(pattern_names)})"

            return CurationResult(
                quality_label="rejected",
                quality_score=0.0,
                security_issues=[],
                quality_issues=[],
                explanation=explanation,
                rejected_reason=rejected_reason,
                secret_matches=secret_matches,
                quality_metrics=None,
            )

        # Stage 2: Security scanning
        security_issues = self.security_scanner.scan(code, language)

        # Check for critical issues (optional rejection)
        if self.reject_on_critical_security:
            critical = [i for i in security_issues if i.severity == "critical"]
            if critical:
                issue_types = list({i.issue_type for i in critical})
                explanation = f"Rejected: critical security issues ({', '.join(issue_types)})"

                return CurationResult(
                    quality_label="rejected",
                    quality_score=0.0,
                    security_issues=security_issues,
                    quality_issues=[],
                    explanation=explanation,
                    rejected_reason="critical_security_issue",
                    secret_matches=[],
                    quality_metrics=None,
                )

        # Stage 3: Quality analysis
        quality_metrics = self.quality_analyzer.analyze(code, language)

        # Collect quality issues as strings
        for issue in quality_metrics.issues:
            quality_issues.append(f"{issue.issue_type}: {issue.message}")

        # Determine final label
        if security_issues:
            # Security issues -> negative label
            explanation = self._generate_security_explanation(security_issues)

            # Penalize quality score based on security severity
            adjusted_score = quality_metrics.overall_score
            for issue in security_issues:
                if issue.severity == "critical":
                    adjusted_score *= 0.3
                elif issue.severity == "high":
                    adjusted_score *= 0.5
                elif issue.severity == "medium":
                    adjusted_score *= 0.7

            return CurationResult(
                quality_label="negative",
                quality_score=adjusted_score,
                security_issues=security_issues,
                quality_issues=quality_issues,
                explanation=explanation,
                rejected_reason=None,
                secret_matches=[],
                quality_metrics=quality_metrics,
            )

        elif quality_metrics.overall_score < self.quality_threshold:
            # Poor quality -> negative label
            explanation = self._generate_quality_explanation(quality_metrics)

            return CurationResult(
                quality_label="negative",
                quality_score=quality_metrics.overall_score,
                security_issues=[],
                quality_issues=quality_issues,
                explanation=explanation,
                rejected_reason=None,
                secret_matches=[],
                quality_metrics=quality_metrics,
            )

        else:
            # All checks passed -> positive label
            return CurationResult(
                quality_label="positive",
                quality_score=quality_metrics.overall_score,
                security_issues=[],
                quality_issues=[],
                explanation=None,
                rejected_reason=None,
                secret_matches=[],
                quality_metrics=quality_metrics,
            )

    def _generate_security_explanation(self, issues: list[SecurityIssue]) -> str:
        """Generate explanation for security-related negative labeling.

        Why: Explanations enable contrastive learning - the model learns WHY
        code is bad, not just that it is bad.

        Args:
            issues: List of security issues found

        Returns:
            Human-readable explanation with recommendations
        """
        lines = ["This code has security issues:"]

        # Group by issue type
        by_type: dict[str, list[SecurityIssue]] = {}
        for issue in issues:
            if issue.issue_type not in by_type:
                by_type[issue.issue_type] = []
            by_type[issue.issue_type].append(issue)

        for _issue_type, type_issues in by_type.items():
            # Take first issue for representative message
            representative = type_issues[0]
            count = len(type_issues)

            if count == 1:
                lines.append(f"- {representative.message}")
                lines.append(f"  Fix: {representative.recommendation}")
            else:
                lines.append(f"- {representative.message} ({count} occurrences)")
                lines.append(f"  Fix: {representative.recommendation}")

        return "\n".join(lines)

    def _generate_quality_explanation(self, metrics: QualityMetrics) -> str:
        """Generate explanation for quality-related negative labeling.

        Why: Quality explanations help the model understand what makes code
        maintainable and readable.

        Args:
            metrics: Quality metrics with issues

        Returns:
            Human-readable explanation of quality problems
        """
        lines = [f"This code has quality issues (score: {metrics.overall_score:.2f}):"]

        for issue in metrics.issues:
            lines.append(f"- {issue.message}")

        # Add specific metric-based feedback
        if metrics.docstring_ratio < 0.3:
            lines.append(
                f"- Low documentation: only {metrics.docstring_ratio:.0%} of definitions have docstrings"
            )

        if metrics.max_nesting_depth > 5:
            lines.append(
                f"- Deep nesting (depth {metrics.max_nesting_depth}): consider extracting functions"
            )

        if metrics.max_function_lines > 100:
            lines.append(
                f"- Long functions ({metrics.max_function_lines} lines): consider breaking into smaller functions"
            )

        if metrics.magic_number_count > 5:
            lines.append(
                f"- {metrics.magic_number_count} magic numbers: consider using named constants"
            )

        return "\n".join(lines)

    def create_sample(
        self,
        code: str,
        language: str,
        result: CurationResult,
        source_metadata: dict | None = None,
    ) -> CuratedSample:
        """Create a CuratedSample from processing result.

        Why: Separating processing from sample creation allows callers to
        inspect results before committing to a sample, and to customize
        source metadata.

        Args:
            code: Source code
            language: Programming language
            result: CurationResult from process()
            source_metadata: Optional source information

        Returns:
            CuratedSample ready for training
        """
        # Build source metadata
        source_dict = source_metadata or {}
        source = SourceMetadata(
            repo=source_dict.get("repo", ""),
            path=source_dict.get("path", ""),
            license=source_dict.get("license", "unknown"),
            stars=source_dict.get("stars", 0),
            commit_sha=source_dict.get("commit_sha", ""),
            url=source_dict.get("url", ""),
        )

        # Build quality metadata
        if result.quality_metrics:
            metadata = QualityMetadata(
                complexity=0.0,  # Would need radon for cyclomatic complexity
                maintainability=result.quality_score * 100,  # Approximate
                type_coverage=0.0,  # Would need mypy for this
                lint_score=result.quality_score * 10,  # Approximate
                docstring_ratio=result.quality_metrics.docstring_ratio,
                line_count=result.quality_metrics.line_count,
                function_count=result.quality_metrics.function_count,
                class_count=result.quality_metrics.class_count,
                max_nesting_depth=result.quality_metrics.max_nesting_depth,
            )
        else:
            metadata = QualityMetadata(
                line_count=len(code.split("\n")),
            )

        # Convert security issues to dicts
        security_dicts = [
            {
                "type": issue.issue_type,
                "severity": issue.severity,
                "line": issue.line,
                "message": issue.message,
                "recommendation": issue.recommendation,
            }
            for issue in result.security_issues
        ]

        # Detect anti-patterns from issues
        anti_patterns: list[str] = []
        for issue in result.quality_issues:
            if "nesting" in issue.lower():
                anti_patterns.append("deep_nesting")
            if "magic" in issue.lower():
                anti_patterns.append("magic_numbers")
            if "docstring" in issue.lower():
                anti_patterns.append("missing_documentation")
            if "function" in issue.lower() and "long" in issue.lower():
                anti_patterns.append("long_functions")

        # Map label (CurationResult uses 3 labels, CuratedSample uses 4)
        quality_label: Literal["positive", "negative", "educational", "rejected"]
        if result.quality_label == "rejected":
            quality_label = "rejected"
        elif result.quality_label == "negative":
            quality_label = "negative"
        else:
            quality_label = "positive"

        return CuratedSample(
            text=code,
            language=language,
            quality_label=quality_label,
            quality_score=result.quality_score,
            security_issues=security_dicts,
            quality_issues=result.quality_issues,
            anti_patterns=list(set(anti_patterns)),
            explanation=result.explanation,
            source=source,
            metadata=metadata,
        )
