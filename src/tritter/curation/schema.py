"""
Data labeling schema for curated training samples.

Defines the structure for curated training data including quality labels, security
issues, and metadata. All samples flow through the curation pipeline and are labeled
according to this schema before being used for training.

Why: A well-defined schema ensures consistency across the curation pipeline and enables
downstream tools (training, evaluation) to rely on predictable data formats. The schema
captures both the sample content and rich metadata about its quality, security posture,
and provenance. This metadata enables contrastive learning where the model learns both
good and bad patterns, with explicit labels explaining why bad code is bad.

The schema follows SPEC-007 Dataset Quality Gates specification, supporting:
- Positive examples (high-quality, secure code)
- Negative examples (poor quality with explicit issues and explanations)
- Educational examples (refactoring, code reviews)
- Rejected samples (secrets - never included in training)
"""

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "SourceMetadata",
    "QualityMetadata",
    "CuratedSample",
    "QualityLabel",
]

# Type alias for quality labels
QualityLabel = Literal["positive", "negative", "educational", "rejected"]


@dataclass
class SourceMetadata:
    """Metadata about the source of a code sample.

    Tracks provenance information to enable reproducibility, license compliance,
    and quality filtering based on repository reputation.

    Why: Knowing where code comes from is essential for:
    1. License compliance - only train on permissively licensed code
    2. Quality signals - code from high-star repos tends to be better reviewed
    3. Reproducibility - ability to trace training data back to source
    4. Deduplication - avoid training on the same code from multiple forks

    Attributes:
        repo: Repository identifier (e.g., "author/repo" for GitHub)
        path: File path within the repository
        license: SPDX license identifier (e.g., "MIT", "Apache-2.0")
        stars: Repository star count as quality signal
        commit_sha: Specific commit hash for exact reproducibility
        url: Optional URL to the source file
    """

    repo: str
    path: str
    license: str = "unknown"
    stars: int = 0
    commit_sha: str = ""
    url: str = ""


@dataclass
class QualityMetadata:
    """Computed quality metrics for a code sample.

    Contains numerical scores from static analysis tools that quantify code quality.
    These metrics inform the quality_label decision and are stored for analysis.

    Why: Numerical metrics enable:
    1. Threshold-based labeling - consistent quality decisions across the dataset
    2. Training weighting - samples with extreme scores can be weighted differently
    3. Trend analysis - track overall dataset quality over time
    4. Model evaluation - correlate model outputs with objective quality measures

    Attributes:
        complexity: Cyclomatic complexity score (lower is better, < 10 is good)
        maintainability: Maintainability index (higher is better, > 20 is good)
        type_coverage: Fraction of functions with type hints [0.0, 1.0]
        lint_score: Lint score normalized to [0, 10] (higher is better)
        docstring_ratio: Fraction of functions/classes with docstrings [0.0, 1.0]
        line_count: Total lines in the sample
        function_count: Number of functions defined
        class_count: Number of classes defined
        max_nesting_depth: Maximum nesting depth (< 5 is good)
    """

    complexity: float = 0.0
    maintainability: float = 100.0
    type_coverage: float = 0.0
    lint_score: float = 10.0
    docstring_ratio: float = 0.0
    line_count: int = 0
    function_count: int = 0
    class_count: int = 0
    max_nesting_depth: int = 0


@dataclass
class CuratedSample:
    """A fully curated training sample with quality labels and metadata.

    This is the final output format of the curation pipeline. Each sample includes
    the code text, its language, quality assessment, and rich metadata about issues
    and provenance.

    Why: The CuratedSample format enables contrastive learning where the model learns
    both good and bad code patterns:
    - positive samples: the model learns to emulate this code
    - negative samples: the model learns to identify and explain issues
    - educational samples: the model learns to refactor and improve code
    - rejected samples: NEVER included in training (contain secrets)

    The explanation field is critical for negative/educational samples - it tells the
    model WHY the code is bad, enabling it to provide helpful feedback to users.

    Attributes:
        text: The code content
        language: Programming language (python, rust, triton, etc.)
        quality_label: Overall quality classification
        quality_score: Numerical quality score [0.0, 1.0] (1.0 = perfect)
        security_issues: List of security vulnerabilities found
        quality_issues: List of quality problems (anti-patterns, style issues)
        anti_patterns: Specific anti-patterns detected (god_class, magic_numbers, etc.)
        explanation: Human-readable explanation of why this is good/bad code
        source: Provenance metadata
        metadata: Computed quality metrics
    """

    text: str
    language: str
    quality_label: QualityLabel
    quality_score: float
    security_issues: list[dict[str, str]] = field(default_factory=list)
    quality_issues: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)
    explanation: str | None = None
    source: SourceMetadata = field(default_factory=lambda: SourceMetadata("", ""))
    metadata: QualityMetadata = field(default_factory=QualityMetadata)

    def to_dict(self) -> dict:
        """Convert sample to dictionary for JSON serialization.

        Why: JSON is the standard interchange format for training data. This method
        ensures consistent serialization across the pipeline.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "text": self.text,
            "language": self.language,
            "quality_label": self.quality_label,
            "quality_score": self.quality_score,
            "security_issues": self.security_issues,
            "quality_issues": self.quality_issues,
            "anti_patterns": self.anti_patterns,
            "explanation": self.explanation,
            "source": {
                "repo": self.source.repo,
                "path": self.source.path,
                "license": self.source.license,
                "stars": self.source.stars,
                "commit_sha": self.source.commit_sha,
                "url": self.source.url,
            },
            "metadata": {
                "complexity": self.metadata.complexity,
                "maintainability": self.metadata.maintainability,
                "type_coverage": self.metadata.type_coverage,
                "lint_score": self.metadata.lint_score,
                "docstring_ratio": self.metadata.docstring_ratio,
                "line_count": self.metadata.line_count,
                "function_count": self.metadata.function_count,
                "class_count": self.metadata.class_count,
                "max_nesting_depth": self.metadata.max_nesting_depth,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CuratedSample":
        """Create CuratedSample from dictionary.

        Why: Enables loading samples from JSON files for training or analysis.

        Args:
            data: Dictionary with sample fields

        Returns:
            CuratedSample instance
        """
        source_data = data.get("source", {})
        source = SourceMetadata(
            repo=source_data.get("repo", ""),
            path=source_data.get("path", ""),
            license=source_data.get("license", "unknown"),
            stars=source_data.get("stars", 0),
            commit_sha=source_data.get("commit_sha", ""),
            url=source_data.get("url", ""),
        )

        metadata_data = data.get("metadata", {})
        metadata = QualityMetadata(
            complexity=metadata_data.get("complexity", 0.0),
            maintainability=metadata_data.get("maintainability", 100.0),
            type_coverage=metadata_data.get("type_coverage", 0.0),
            lint_score=metadata_data.get("lint_score", 10.0),
            docstring_ratio=metadata_data.get("docstring_ratio", 0.0),
            line_count=metadata_data.get("line_count", 0),
            function_count=metadata_data.get("function_count", 0),
            class_count=metadata_data.get("class_count", 0),
            max_nesting_depth=metadata_data.get("max_nesting_depth", 0),
        )

        return cls(
            text=data.get("text", ""),
            language=data.get("language", "unknown"),
            quality_label=data.get("quality_label", "positive"),
            quality_score=data.get("quality_score", 0.0),
            security_issues=data.get("security_issues", []),
            quality_issues=data.get("quality_issues", []),
            anti_patterns=data.get("anti_patterns", []),
            explanation=data.get("explanation"),
            source=source,
            metadata=metadata,
        )
