"""
Dataset curation module for training data quality gates.

Provides tools for curating high-quality training data through secret detection,
security scanning, code quality analysis, and deduplication. Implements SPEC-007
Dataset Quality Gates specification.

Why: Training data quality directly impacts model quality. This module ensures:
1. Secrets are NEVER included in training data (high leakage risk)
2. Security vulnerabilities are labeled as negative examples with explanations
3. Poor quality code is labeled as negative examples for contrastive learning
4. Near-duplicates are detected to prevent memorization and bias

Philosophy (Contrastive Learning):
The model should learn both good and bad patterns:
- POSITIVE examples: high-quality, secure code the model should emulate
- NEGATIVE examples: poor/insecure code WITH explanations of what's wrong
- REJECTED samples: secrets that should NEVER appear in training

This enables the model to not just write good code, but also identify and explain
problems in existing code.

Usage:
    from tritter.curation import CurationPipeline, SecretScanner, SecurityScanner

    # Full pipeline
    pipeline = CurationPipeline()
    result = pipeline.process(code, language="python")
    if result.quality_label == "positive":
        # High-quality training sample
        sample = pipeline.create_sample(code, "python", result)

    # Individual scanners
    if SecretScanner().has_secrets(code):
        # REJECT - never train on this
        pass

    issues = SecurityScanner().scan(code, "python")
    for issue in issues:
        print(f"{issue.severity}: {issue.message}")
"""

from .dedup import DATASKETCH_AVAILABLE, MinHashDeduplicator, MinHashSignature
from .pipeline import CurationPipeline, CurationResult
from .quality import QualityAnalyzer, QualityIssue, QualityMetrics
from .schema import CuratedSample, QualityLabel, QualityMetadata, SourceMetadata
from .secrets import SecretMatch, SecretScanner
from .security import SecurityIssue, SecurityScanner, Severity

__all__ = [
    # Pipeline
    "CurationPipeline",
    "CurationResult",
    # Schema
    "CuratedSample",
    "SourceMetadata",
    "QualityMetadata",
    "QualityLabel",
    # Secret Detection
    "SecretScanner",
    "SecretMatch",
    # Security Scanning
    "SecurityScanner",
    "SecurityIssue",
    "Severity",
    # Quality Analysis
    "QualityAnalyzer",
    "QualityMetrics",
    "QualityIssue",
    # Deduplication
    "MinHashDeduplicator",
    "MinHashSignature",
    "DATASKETCH_AVAILABLE",
]
