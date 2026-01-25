#!/usr/bin/env python3
"""Security scanning module for dataset curation.

Why: Insecure code must either be rejected (secrets) or explicitly labeled
as negative examples with explanations. This module implements the security
gates defined in SPEC-007.

Usage:
    from security_scanner import SecurityScanner

    scanner = SecurityScanner()
    result = scanner.scan(code_content, language="python")

    if result.has_secrets:
        # REJECT - never train on secrets
        pass
    elif result.security_issues:
        # Label as negative example with explanations
        pass
"""

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SecurityIssue:
    """Represents a detected security issue.

    Why: Structured representation enables consistent handling and
    explanation generation for negative training examples.
    """
    issue_type: str
    severity: str  # "critical", "high", "medium", "low"
    line_number: int | None
    code_snippet: str
    explanation: str
    fix_suggestion: str


@dataclass
class ScanResult:
    """Result of security scan on code content.

    Why: Aggregates all findings to support quality labeling decisions.
    """
    has_secrets: bool = False
    security_issues: list[SecurityIssue] = field(default_factory=list)
    quality_label: str = "positive"  # positive, negative, reject

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "has_secrets": self.has_secrets,
            "security_issues": [
                {
                    "type": issue.issue_type,
                    "severity": issue.severity,
                    "line": issue.line_number,
                    "snippet": issue.code_snippet[:100],
                    "explanation": issue.explanation,
                    "fix": issue.fix_suggestion,
                }
                for issue in self.security_issues
            ],
            "quality_label": self.quality_label,
        }


# Secret patterns - ALWAYS reject, never train on these
SECRET_PATTERNS = [
    # API Keys
    (r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"][a-zA-Z0-9_\-]{20,}['\"]", "api_key"),
    (r"(?i)(secret|password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{8,}['\"]", "password"),

    # Service-specific tokens
    (r"ghp_[a-zA-Z0-9]{36}", "github_token"),
    (r"gho_[a-zA-Z0-9]{36}", "github_oauth"),
    (r"github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59}", "github_pat"),
    (r"sk-[a-zA-Z0-9]{48}", "openai_key"),
    (r"sk-proj-[a-zA-Z0-9\-_]{80,}", "openai_project_key"),
    (r"AKIA[0-9A-Z]{16}", "aws_access_key"),
    (r"(?i)aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*['\"][a-zA-Z0-9/+=]{40}['\"]", "aws_secret"),
    (r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}", "slack_token"),
    (r"(?i)bearer\s+[a-zA-Z0-9\-_]{20,}\.[a-zA-Z0-9\-_]{20,}", "bearer_token"),

    # Private keys
    (r"-----BEGIN\s+(RSA|DSA|EC|OPENSSH|PGP)\s+PRIVATE\s+KEY-----", "private_key"),
    (r"-----BEGIN\s+PRIVATE\s+KEY-----", "private_key"),
]

# Vulnerability patterns - label as negative with explanation
PYTHON_VULNERABILITY_PATTERNS: list[tuple[str, str, str, str]] = [
    # (pattern, issue_type, explanation, fix)
    (
        r"\beval\s*\([^)]*\)",
        "code_injection",
        "eval() executes arbitrary code. If the input comes from users, attackers can execute malicious code.",
        "Use ast.literal_eval() for safe parsing of Python literals, or implement proper parsing for your data format.",
    ),
    (
        r"\bexec\s*\([^)]*\)",
        "code_injection",
        "exec() executes arbitrary Python code. This is extremely dangerous with untrusted input.",
        "Avoid dynamic code execution. Use proper data structures or configuration instead.",
    ),
    (
        r"subprocess\.[a-z_]+\([^)]*shell\s*=\s*True",
        "command_injection",
        "shell=True passes the command through the shell, enabling command injection if input is untrusted.",
        "Use shell=False with a list of arguments: subprocess.run(['cmd', 'arg1', 'arg2'])",
    ),
    (
        r"os\.system\s*\([^)]*\)",
        "command_injection",
        "os.system() runs commands through the shell, enabling injection attacks.",
        "Use subprocess.run() with shell=False and a list of arguments.",
    ),
    (
        r"pickle\.loads?\s*\([^)]*\)",
        "unsafe_deserialization",
        "pickle can execute arbitrary code during deserialization. Never use with untrusted data.",
        "Use JSON, MessagePack, or other safe serialization formats. If you must use pickle, only load data you created.",
    ),
    (
        r"yaml\.load\s*\([^)]*\)(?!\s*,\s*Loader\s*=\s*yaml\.SafeLoader)",
        "unsafe_deserialization",
        "yaml.load() without SafeLoader can execute arbitrary Python code.",
        "Always use yaml.safe_load() or explicitly specify Loader=yaml.SafeLoader.",
    ),
    (
        r"marshal\.loads?\s*\([^)]*\)",
        "unsafe_deserialization",
        "marshal can execute code during deserialization. Not safe for untrusted data.",
        "Use JSON or other safe serialization formats.",
    ),
    (
        r"__import__\s*\([^)]*\)",
        "code_injection",
        "__import__ with dynamic input can load malicious modules.",
        "Use explicit imports or importlib with validated module names.",
    ),
    (
        r"request\.(GET|POST|args|form|values)\[['\"][^'\"]+['\"]\]",
        "missing_input_validation",
        "Directly using request parameters without validation can lead to injection attacks.",
        "Validate and sanitize all user input before use. Use type coercion and allowlists.",
    ),
    (
        r"\.format\([^)]*\)|f['\"].*\{[^}]*\}.*['\"]",
        "potential_injection",
        "String formatting with user input can lead to injection in SQL, shell, or template contexts.",
        "Use parameterized queries for SQL, and avoid string formatting with untrusted input.",
    ),
    (
        r"sqlite3\.execute\s*\([^)]*%|sqlite3\.execute\s*\([^)]*\.format",
        "sql_injection",
        "String formatting in SQL queries enables SQL injection attacks.",
        "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
    ),
    (
        r"cursor\.execute\s*\([^)]*%|cursor\.execute\s*\([^)]*\.format",
        "sql_injection",
        "String formatting in SQL queries enables SQL injection attacks.",
        "Use parameterized queries with placeholders (? or %s depending on database).",
    ),
    (
        r"hashlib\.md5\s*\(|hashlib\.sha1\s*\(",
        "weak_crypto",
        "MD5 and SHA1 are cryptographically broken and should not be used for security purposes.",
        "Use SHA-256 or better: hashlib.sha256(). For passwords, use bcrypt, scrypt, or argon2.",
    ),
    (
        r"random\.(random|randint|choice|randrange)\s*\(",
        "insecure_random",
        "The random module is not cryptographically secure. Don't use for security-sensitive operations.",
        "Use secrets module for security: secrets.token_hex(), secrets.choice(), etc.",
    ),
    (
        r"tempfile\.mktemp\s*\(",
        "race_condition",
        "mktemp() has a race condition between name generation and file creation.",
        "Use tempfile.mkstemp() or tempfile.NamedTemporaryFile() which create the file atomically.",
    ),
    (
        r"assert\s+[^,\n]+,?\s*['\"]",
        "assert_in_production",
        "Assertions can be disabled with -O flag. Don't use for security checks.",
        "Use explicit if statements and raise exceptions for security-critical checks.",
    ),
]

RUST_VULNERABILITY_PATTERNS: list[tuple[str, str, str, str]] = [
    (
        r"unsafe\s*\{",
        "unsafe_block",
        "Unsafe blocks bypass Rust's safety guarantees. Must be carefully audited.",
        "Document safety invariants. Consider if safe alternatives exist. Minimize unsafe scope.",
    ),
    (
        r"\.unwrap\(\)",
        "panic_on_error",
        "unwrap() panics on None/Err, which can crash the program or cause denial of service.",
        "Use proper error handling: match, if let, ?, or unwrap_or_default().",
    ),
    (
        r"\.expect\([^)]*\)",
        "panic_on_error",
        "expect() panics with a message on None/Err. Better than unwrap but still causes crashes.",
        "Use proper error handling for recoverable errors. expect() is OK for unrecoverable states.",
    ),
    (
        r"std::mem::transmute",
        "memory_unsafety",
        "transmute reinterprets bits as different type. Extremely dangerous if types don't match.",
        "Use safe casts, From/Into traits, or bytemuck for well-defined transmutes.",
    ),
    (
        r"std::ptr::null\(\)|std::ptr::null_mut\(\)",
        "null_pointer",
        "Null pointers in Rust require unsafe to dereference and can cause crashes.",
        "Use Option<NonNull<T>> or Option<&T> to represent optional pointers safely.",
    ),
    (
        r"#\[no_mangle\]",
        "ffi_exposure",
        "no_mangle exposes function to FFI. Ensure proper safety documentation.",
        "Document safety requirements for FFI callers. Consider using safer FFI wrappers.",
    ),
    (
        r"Command::new\([^)]*\)\.arg\(",
        "command_injection",
        "Building shell commands requires careful input validation to prevent injection.",
        "Validate and sanitize all user input. Avoid shell=true patterns. Use allowlists.",
    ),
]

# Triton (OpenAI GPU kernel language) vulnerability patterns
TRITON_VULNERABILITY_PATTERNS: list[tuple[str, str, str, str]] = [
    (
        r"tl\.load\([^)]*,\s*mask\s*=\s*None",
        "unmasked_load",
        "Unmasked loads can read out-of-bounds memory, causing undefined behavior or crashes.",
        "Always provide a mask for boundary conditions: tl.load(ptr, mask=offs < n, other=0.0)",
    ),
    (
        r"tl\.store\([^)]*,\s*mask\s*=\s*None",
        "unmasked_store",
        "Unmasked stores can write out-of-bounds, corrupting memory or causing crashes.",
        "Always provide a mask: tl.store(ptr, value, mask=offs < n)",
    ),
    (
        r"tl\.atomic_add\([^)]*\)(?!.*mask)",
        "unmasked_atomic",
        "Unmasked atomic operations on out-of-bounds addresses cause undefined behavior.",
        "Use mask parameter: tl.atomic_add(ptr, val, mask=condition)",
    ),
    (
        r"BLOCK_SIZE\s*=\s*\d{5,}",
        "excessive_block_size",
        "Block sizes > 10000 can exceed shared memory limits, causing kernel launch failures.",
        "Use block sizes appropriate for your GPU (typically 128-1024). Profile to optimize.",
    ),
    (
        r"tl\.arange\([^)]*\)\s*\*\s*tl\.arange",
        "quadratic_indexing",
        "Quadratic index computation can overflow or cause excessive register pressure.",
        "Use linear indexing where possible. Check for overflow with large tensors.",
    ),
    (
        r"@triton\.jit[^@]*def\s+\w+[^:]*:\s*\n\s*(?!.*tl\.program_id)",
        "missing_program_id",
        "Kernels without tl.program_id() don't parallelize across GPU blocks.",
        "Use pid = tl.program_id(axis=0) to distribute work across blocks.",
    ),
    (
        r"tl\.where\([^)]*,\s*[^,)]*\s*/\s*[^,)]*,",
        "division_in_where",
        "Division inside tl.where still executes both branches, risking divide-by-zero.",
        "Guard division before tl.where: safe_div = tl.where(denom != 0, num / denom, 0.0)",
    ),
    (
        r"\.to\(tl\.float16\)(?!.*tl\.float32)",
        "precision_loss",
        "Converting to float16 without intermediate float32 can cause precision loss in accumulations.",
        "Accumulate in float32, convert to float16 only for final output.",
    ),
    (
        r"num_warps\s*=\s*(?:1[7-9]|[2-9]\d|\d{3,})",
        "excessive_warps",
        "num_warps > 16 is rarely beneficial and wastes resources on most GPUs.",
        "Use num_warps between 1-8 for most kernels. Profile to find optimal value.",
    ),
    (
        r"tl\.debug_barrier\(\)",
        "debug_barrier",
        "Debug barriers should not be in production code; they serialize execution.",
        "Remove tl.debug_barrier() for production. Use only for debugging race conditions.",
    ),
]


class SecurityScanner:
    """Scans code for security issues and secrets.

    Why: Central class for all security scanning, supporting multiple languages
    and providing structured results for training data labeling.
    """

    def __init__(self) -> None:
        """Initialize scanner with compiled patterns."""
        self._secret_patterns = [
            (re.compile(pattern), name)
            for pattern, name in SECRET_PATTERNS
        ]
        self._python_patterns = [
            (re.compile(pattern), issue_type, explanation, fix)
            for pattern, issue_type, explanation, fix in PYTHON_VULNERABILITY_PATTERNS
        ]
        self._rust_patterns = [
            (re.compile(pattern), issue_type, explanation, fix)
            for pattern, issue_type, explanation, fix in RUST_VULNERABILITY_PATTERNS
        ]
        self._triton_patterns = [
            (re.compile(pattern), issue_type, explanation, fix)
            for pattern, issue_type, explanation, fix in TRITON_VULNERABILITY_PATTERNS
        ]

    def scan(self, content: str, language: str = "python") -> ScanResult:
        """Scan code content for security issues.

        Args:
            content: Source code to scan
            language: Programming language (python, rust)

        Returns:
            ScanResult with findings and quality label
        """
        result = ScanResult()
        lines = content.split("\n")

        # Check for secrets first - these are always rejected
        for pattern, secret_type in self._secret_patterns:
            for i, line in enumerate(lines, 1):
                if pattern.search(line):
                    result.has_secrets = True
                    result.quality_label = "reject"
                    result.security_issues.append(SecurityIssue(
                        issue_type=f"secret_{secret_type}",
                        severity="critical",
                        line_number=i,
                        code_snippet=line.strip()[:100],
                        explanation=f"Hardcoded {secret_type} detected. Never commit secrets to code.",
                        fix_suggestion="Use environment variables or a secrets manager.",
                    ))

        # If secrets found, don't bother with other checks
        if result.has_secrets:
            return result

        # Language-specific vulnerability patterns
        if language == "python":
            patterns = self._python_patterns
        elif language == "rust":
            patterns = self._rust_patterns
        elif language == "triton":
            patterns = self._triton_patterns
        else:
            patterns = self._python_patterns  # Default to Python

        for pattern, issue_type, explanation, fix in patterns:
            for i, line in enumerate(lines, 1):
                match = pattern.search(line)
                if match:
                    result.security_issues.append(SecurityIssue(
                        issue_type=issue_type,
                        severity="high" if "injection" in issue_type else "medium",
                        line_number=i,
                        code_snippet=match.group(0)[:100],
                        explanation=explanation,
                        fix_suggestion=fix,
                    ))

        # Set quality label based on findings
        if result.security_issues:
            high_severity = any(
                issue.severity in ("critical", "high")
                for issue in result.security_issues
            )
            result.quality_label = "negative" if high_severity else "positive"

        return result

    def format_negative_example(self, content: str, result: ScanResult) -> dict:
        """Format a negative example with explanations for training.

        Why: Negative examples must include explicit explanations so the model
        learns WHY the code is bad, not just that it's bad.

        Args:
            content: Original code content
            result: Scan result with issues

        Returns:
            Formatted dictionary for training data
        """
        explanations = []
        for issue in result.security_issues:
            explanations.append(
                f"Line {issue.line_number}: {issue.issue_type} - {issue.explanation} "
                f"Fix: {issue.fix_suggestion}"
            )

        return {
            "text": content,
            "quality_label": "negative",
            "security_issues": [issue.issue_type for issue in result.security_issues],
            "explanation": "\n".join(explanations),
            "issue_count": len(result.security_issues),
        }


def scan_file(filepath: str, language: str = "python") -> ScanResult:
    """Convenience function to scan a file.

    Args:
        filepath: Path to file to scan
        language: Programming language

    Returns:
        ScanResult
    """
    with open(filepath) as f:
        content = f.read()

    scanner = SecurityScanner()
    return scanner.scan(content, language)


if __name__ == "__main__":
    # Demo/test
    import sys

    test_code_safe = '''
def greet(name: str) -> str:
    """Return a greeting."""
    return f"Hello, {name}!"
'''

    test_code_unsafe = '''
import os
import pickle

def dangerous_eval(user_input):
    return eval(user_input)  # BAD: code injection

def run_command(cmd):
    os.system(cmd)  # BAD: command injection

def load_data(data):
    return pickle.loads(data)  # BAD: unsafe deserialization
'''

    test_code_secrets = '''
API_KEY = "sk-1234567890abcdefghijklmnopqrstuvwxyz123456"
password = "super_secret_password_123"
'''

    scanner = SecurityScanner()

    print("=== Safe code ===")
    result = scanner.scan(test_code_safe)
    print(f"Label: {result.quality_label}")
    print(f"Issues: {len(result.security_issues)}")

    print("\n=== Unsafe code ===")
    result = scanner.scan(test_code_unsafe)
    print(f"Label: {result.quality_label}")
    print(f"Issues: {len(result.security_issues)}")
    for issue in result.security_issues:
        print(f"  - {issue.issue_type}: {issue.explanation[:50]}...")

    print("\n=== Code with secrets ===")
    result = scanner.scan(test_code_secrets)
    print(f"Label: {result.quality_label}")
    print(f"Has secrets: {result.has_secrets}")

    sys.exit(0 if not result.has_secrets else 1)
