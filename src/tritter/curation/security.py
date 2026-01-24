"""
Security vulnerability detection for training data curation.

Scans code for security vulnerabilities and unsafe patterns that should be labeled
as negative examples with explanations. Unlike secrets (which are rejected entirely),
security issues become contrastive learning opportunities - the model learns to
identify and explain these problems.

Why: Security vulnerabilities in code are educational - the model should learn:
1. To recognize dangerous patterns (eval, shell=True, unsafe deserialization)
2. To explain WHY they're dangerous
3. To suggest safer alternatives

This is different from secrets, which must never appear in training data. Security
issues are labeled as "negative" with explanations, enabling the model to help users
write more secure code.

Supported Languages:
- Python: eval, exec, subprocess shell=True, pickle, unsafe yaml
- Rust: unsafe blocks, unwrap() usage
- Triton: unmasked loads/stores, excessive block sizes
"""

import re
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "SecurityIssue",
    "SecurityScanner",
    "Severity",
]

# Type alias for severity levels
Severity = Literal["critical", "high", "medium", "low", "info"]


@dataclass
class SecurityIssue:
    """A detected security vulnerability in source code.

    Contains information about the vulnerability type, location, and recommended fix.
    Used to label code as negative examples with explanations.

    Why: Rich metadata about security issues enables:
    1. Severity-based filtering (reject critical, warn on medium)
    2. Generating explanations for negative training samples
    3. Tracking common vulnerability patterns in datasets

    Attributes:
        issue_type: Category of the vulnerability (e.g., "code_injection")
        severity: How dangerous this vulnerability is
        line: Line number where issue was found (1-indexed)
        message: Description of what's wrong
        recommendation: Suggested fix or safer alternative
        pattern_name: Name of the detection pattern that triggered
        matched_text: The code snippet that triggered the detection
    """

    issue_type: str
    severity: Severity
    line: int
    message: str
    recommendation: str
    pattern_name: str = ""
    matched_text: str = ""


class SecurityScanner:
    """Scanner for security vulnerabilities in code.

    Detects common security anti-patterns for multiple languages. Issues are categorized
    by severity to enable nuanced handling:
    - critical: Always label as negative, high training weight
    - high: Label as negative with strong warnings
    - medium: Label as negative with moderate warnings
    - low: Label as negative, informational
    - info: Note in metadata but may still be positive

    Why: Pattern-based detection catches common vulnerabilities efficiently. While not
    as comprehensive as tools like Bandit or Semgrep, this scanner covers the most
    critical patterns without external dependencies. For production use, consider
    integrating with proper security scanning tools.

    Usage:
        scanner = SecurityScanner()
        issues = scanner.scan(code, language="python")
        for issue in issues:
            print(f"{issue.severity}: {issue.message}")
            print(f"Fix: {issue.recommendation}")
    """

    # Python security patterns
    # Format: (pattern_name, regex, issue_type, severity, message, recommendation)
    PYTHON_PATTERNS: list[tuple[str, re.Pattern[str], str, Severity, str, str]] = [
        (
            "eval_usage",
            re.compile(r"\beval\s*\("),
            "code_injection",
            "critical",
            "eval() executes arbitrary code and is extremely dangerous with untrusted input",
            "Use ast.literal_eval() for safe parsing of literals, or implement proper parsing",
        ),
        (
            "exec_usage",
            re.compile(r"\bexec\s*\("),
            "code_injection",
            "critical",
            "exec() executes arbitrary code and should be avoided with untrusted input",
            "Avoid dynamic code execution; use safer alternatives like importlib",
        ),
        (
            "subprocess_shell",
            re.compile(r"subprocess\.[a-z_]+\([^)]*shell\s*=\s*True"),
            "command_injection",
            "high",
            "subprocess with shell=True is vulnerable to command injection",
            "Use shell=False with a list of arguments instead of a shell command string",
        ),
        (
            "os_system",
            re.compile(r"\bos\.system\s*\("),
            "command_injection",
            "high",
            "os.system() is vulnerable to command injection",
            "Use subprocess.run() with shell=False and a list of arguments",
        ),
        (
            "pickle_load",
            re.compile(r"pickle\.loads?\s*\("),
            "unsafe_deserialization",
            "high",
            "pickle can execute arbitrary code during deserialization",
            "Use safer formats like JSON, or restrict pickle to trusted sources only",
        ),
        (
            "yaml_unsafe_load",
            re.compile(r"yaml\.load\s*\([^)]*\)(?!\s*,\s*Loader\s*=\s*yaml\.SafeLoader)"),
            "unsafe_deserialization",
            "high",
            "yaml.load() without SafeLoader can execute arbitrary code",
            "Use yaml.safe_load() or yaml.load(data, Loader=yaml.SafeLoader)",
        ),
        (
            "yaml_full_load",
            re.compile(r"yaml\.(full_load|unsafe_load)\s*\("),
            "unsafe_deserialization",
            "high",
            "yaml.full_load()/unsafe_load() can execute arbitrary code",
            "Use yaml.safe_load() instead",
        ),
        (
            "marshal_load",
            re.compile(r"marshal\.loads?\s*\("),
            "unsafe_deserialization",
            "medium",
            "marshal can execute code when loading malicious data",
            "Use safer formats like JSON for untrusted data",
        ),
        (
            "shelve_open",
            re.compile(r"shelve\.open\s*\("),
            "unsafe_deserialization",
            "medium",
            "shelve uses pickle internally, which can execute arbitrary code",
            "Use safer storage formats for untrusted data",
        ),
        (
            "temp_file_race",
            re.compile(r"open\s*\([^)]*['\"][^'\"]*tmp[^'\"]*['\"]"),
            "race_condition",
            "medium",
            "Hardcoded temp file paths are vulnerable to race conditions",
            "Use tempfile.mkstemp() or tempfile.NamedTemporaryFile() for secure temp files",
        ),
        (
            "sql_format_string",
            re.compile(r"(execute|cursor)\s*\([^)]*['\"].*%[sd].*['\"].*%"),
            "sql_injection",
            "high",
            "String formatting in SQL queries is vulnerable to SQL injection",
            "Use parameterized queries with placeholders",
        ),
        (
            "bare_except",
            re.compile(r"\bexcept\s*:"),
            "error_handling",
            "low",
            "Bare except catches all exceptions including KeyboardInterrupt and SystemExit",
            "Catch specific exceptions or use 'except Exception:'",
        ),
        (
            "assert_security",
            re.compile(r"\bassert\s+[^,\n]*(?:auth|password|permission|access|secure)"),
            "security_bypass",
            "medium",
            "Assert statements can be disabled with -O flag, not suitable for security checks",
            "Use if statements with proper error handling for security checks",
        ),
        (
            "hardcoded_bind",
            re.compile(r"\.bind\s*\(\s*['\"]0\.0\.0\.0['\"]"),
            "network_exposure",
            "medium",
            "Binding to 0.0.0.0 exposes the service to all network interfaces",
            "Bind to specific interfaces (127.0.0.1 for local-only) unless intended",
        ),
        (
            "debug_true",
            re.compile(r"(?i)debug\s*=\s*True"),
            "information_exposure",
            "low",
            "Debug mode enabled may expose sensitive information",
            "Disable debug mode in production",
        ),
    ]

    # Rust security patterns
    RUST_PATTERNS: list[tuple[str, re.Pattern[str], str, Severity, str, str]] = [
        (
            "unsafe_block",
            re.compile(r"\bunsafe\s*\{"),
            "unsafe_code",
            "medium",
            "Unsafe block bypasses Rust's safety guarantees",
            "Document safety invariants; minimize unsafe code; prefer safe abstractions",
        ),
        (
            "unwrap_usage",
            re.compile(r"\.unwrap\s*\(\s*\)"),
            "panic_risk",
            "low",
            "unwrap() will panic on None/Err, causing program termination",
            "Use pattern matching, unwrap_or(), unwrap_or_else(), or the ? operator",
        ),
        (
            "expect_usage",
            re.compile(r"\.expect\s*\("),
            "panic_risk",
            "info",
            "expect() will panic with message on None/Err",
            "Consider returning Result or using ? operator for recoverable errors",
        ),
        (
            "transmute",
            re.compile(r"\btransmute\s*[:<]"),
            "unsafe_code",
            "high",
            "transmute bypasses type safety and can cause undefined behavior",
            "Use safe alternatives like From/Into traits or as casts where possible",
        ),
        (
            "raw_pointer_deref",
            re.compile(r"\*[a-zA-Z_][a-zA-Z0-9_]*\s*(?:as\s+\*|\.)"),
            "unsafe_code",
            "medium",
            "Raw pointer dereference requires unsafe and can cause memory issues",
            "Prefer references and smart pointers; document safety when raw pointers are necessary",
        ),
        (
            "format_string_panic",
            re.compile(r"format!\s*\([^)]*\{\}[^)]*\)"),
            "panic_risk",
            "info",
            "format! can panic if argument count doesn't match placeholders",
            "Verify format string matches argument count; consider using format_args!",
        ),
        (
            "todo_macro",
            re.compile(r"\btodo!\s*\("),
            "incomplete_code",
            "low",
            "todo!() will panic at runtime",
            "Implement the functionality or use unimplemented!() with documentation",
        ),
    ]

    # Triton (GPU kernel) security/correctness patterns
    TRITON_PATTERNS: list[tuple[str, re.Pattern[str], str, Severity, str, str]] = [
        (
            "unmasked_load",
            re.compile(r"tl\.load\s*\([^)]*(?:mask\s*=\s*None|(?!mask))"),
            "memory_safety",
            "high",
            "Unmasked tl.load can read out-of-bounds memory",
            "Always use mask parameter for bounds checking: tl.load(ptr, mask=mask, other=0.0)",
        ),
        (
            "unmasked_store",
            re.compile(r"tl\.store\s*\([^)]*(?:mask\s*=\s*None|(?!mask))"),
            "memory_safety",
            "high",
            "Unmasked tl.store can write out-of-bounds memory",
            "Always use mask parameter: tl.store(ptr, value, mask=mask)",
        ),
        (
            "excessive_block_size",
            re.compile(r"BLOCK_SIZE\s*=\s*(\d{5,})"),
            "resource_limit",
            "medium",
            "Block size may exceed GPU shared memory limits",
            "Keep BLOCK_SIZE <= 1024 for most GPUs; check against GPU shared memory limits",
        ),
        (
            "excessive_warps",
            re.compile(r"num_warps\s*=\s*(?:1[7-9]|[2-9]\d+)"),
            "resource_limit",
            "medium",
            "num_warps > 16 is usually suboptimal; typical range is 1-8",
            "Use num_warps between 1-8 for most kernels; profile to find optimal value",
        ),
        (
            "division_in_where",
            re.compile(r"tl\.where\s*\([^)]*\s*/\s*[^)]*,"),
            "correctness",
            "medium",
            "Division in tl.where may cause issues if divisor can be zero in false branch",
            "Perform division outside tl.where or ensure divisor is never zero",
        ),
        (
            "precision_loss",
            re.compile(r"\.to\s*\(\s*tl\.float16\s*\)(?!.*\.to\s*\(\s*tl\.float32)"),
            "precision",
            "low",
            "Converting to float16 without accumulation in float32 may cause precision loss",
            "Accumulate in float32 for numerical stability, then convert final result",
        ),
        (
            "atomic_without_mask",
            re.compile(r"tl\.atomic_[a-z]+\s*\([^)]*\)(?!.*mask)"),
            "memory_safety",
            "medium",
            "Atomic operations without masks may cause out-of-bounds access",
            "Use mask parameter to guard atomic operations at boundaries",
        ),
        (
            "hardcoded_num_stages",
            re.compile(r"num_stages\s*=\s*[89]|num_stages\s*=\s*\d{2,}"),
            "resource_limit",
            "low",
            "num_stages > 7 may not be supported on all GPUs",
            "Use num_stages between 2-5 for best portability; higher values need newer GPUs",
        ),
    ]

    def __init__(self) -> None:
        """Initialize security scanner with all language patterns.

        Why: Pre-compiling patterns at initialization ensures they're ready for fast
        scanning. The scanner is stateless and thread-safe.
        """
        self.python_patterns = list(self.PYTHON_PATTERNS)
        self.rust_patterns = list(self.RUST_PATTERNS)
        self.triton_patterns = list(self.TRITON_PATTERNS)

    def scan(self, code: str, language: str) -> list[SecurityIssue]:
        """Scan code for security vulnerabilities.

        Applies language-specific patterns to detect security issues. Returns all
        matches with severity, message, and recommendations.

        Why: Full scan returns all issues for comprehensive reporting. The caller can
        filter by severity or aggregate issues for the sample's quality label.

        Args:
            code: Source code to scan
            language: Programming language ("python", "rust", "triton")

        Returns:
            List of SecurityIssue objects for each detected vulnerability
        """
        language = language.lower()

        # Select patterns based on language
        if language == "python":
            patterns = self.python_patterns
        elif language == "rust":
            patterns = self.rust_patterns
        elif language == "triton":
            patterns = self.triton_patterns
        else:
            # Unknown language - return empty list
            return []

        issues: list[SecurityIssue] = []

        # Pre-compute line starts for line number calculation
        line_starts = [0]
        for i, char in enumerate(code):
            if char == "\n":
                line_starts.append(i + 1)

        def get_line_number(pos: int) -> int:
            """Get 1-indexed line number for character position."""
            for i, _start in enumerate(line_starts):
                if i + 1 < len(line_starts) and line_starts[i + 1] > pos:
                    return i + 1
            return len(line_starts)

        # Scan with each pattern
        for pattern_name, pattern, issue_type, severity, message, recommendation in patterns:
            for match in pattern.finditer(code):
                issue = SecurityIssue(
                    issue_type=issue_type,
                    severity=severity,
                    line=get_line_number(match.start()),
                    message=message,
                    recommendation=recommendation,
                    pattern_name=pattern_name,
                    matched_text=match.group()[:100],  # Truncate long matches
                )
                issues.append(issue)

        return issues

    def has_critical_issues(self, code: str, language: str) -> bool:
        """Quick check for critical security issues.

        More efficient than full scan when you only need to know if critical
        issues exist. Returns immediately upon finding first critical issue.

        Why: Fast rejection for critical issues enables efficient pre-filtering
        of large datasets before full analysis.

        Args:
            code: Source code to check
            language: Programming language

        Returns:
            True if any critical/high severity issue found
        """
        issues = self.scan(code, language)
        return any(issue.severity in ("critical", "high") for issue in issues)

    def get_severity_counts(self, issues: list[SecurityIssue]) -> dict[str, int]:
        """Count issues by severity level.

        Why: Summary statistics help with dataset analysis and quality reporting.

        Args:
            issues: List of security issues

        Returns:
            Dictionary mapping severity to count
        """
        counts: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for issue in issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        return counts
