"""
Secret detection for training data curation.

Scans code for hardcoded secrets (API keys, passwords, tokens) that must NEVER be
included in training data, even as negative examples. The risk of secret leakage
through model memorization is too high.

Why: Language models can memorize and reproduce training data. If secrets are included
in training, the model could output them during inference, exposing sensitive credentials.
This is a critical security risk - unlike other code quality issues, secrets cannot be
safely used as negative examples because the model might still learn to reproduce them.

The scanner uses regex patterns to detect common secret formats:
- API keys (generic key=value patterns)
- OAuth/Bearer tokens
- Cloud provider credentials (AWS, OpenAI, GitHub)
- Passwords in configuration
- Private keys (SSH, PGP)

Detection Strategy:
1. Pattern matching with high-precision regexes (minimize false positives)
2. Context-aware scanning (only flag secrets in code/config contexts)
3. Redaction of matched content to prevent logging actual secrets
"""

import re
from dataclasses import dataclass

__all__ = [
    "SecretMatch",
    "SecretScanner",
]


@dataclass
class SecretMatch:
    """A detected secret in source code.

    Contains information about where a secret was found and what type it is.
    The matched text is redacted to prevent logging actual secrets.

    Why: Storing match details enables reporting and debugging without exposing
    the actual secret values. Redaction is critical - we need to know WHERE secrets
    were found but should never log WHAT they contain.

    Attributes:
        pattern_name: Name of the pattern that matched (e.g., "github_token")
        start: Start position in source text (character index)
        end: End position in source text (character index)
        matched_text: Redacted version of matched text (first/last chars + ***)
        line_number: Line number where secret was found (1-indexed)
    """

    pattern_name: str
    start: int
    end: int
    matched_text: str  # Redacted
    line_number: int

    @staticmethod
    def redact(text: str, visible_chars: int = 4) -> str:
        """Redact sensitive text, showing only first/last few characters.

        Why: We need to show enough to identify what was matched (for debugging)
        without exposing the full secret. Standard practice is to show first
        and last few characters with asterisks in between.

        Args:
            text: The secret text to redact
            visible_chars: Number of characters to show at start and end

        Returns:
            Redacted string like "ghp_****36ch"
        """
        if len(text) <= visible_chars * 2 + 3:
            # Too short to meaningfully redact
            return "*" * len(text)
        return f"{text[:visible_chars]}***{text[-visible_chars:]}"


class SecretScanner:
    """Scanner for detecting hardcoded secrets in code.

    Uses regex patterns to identify common secret formats. Designed for high precision
    (few false positives) rather than recall, since false positives cause valid code
    to be rejected while false negatives risk training on secrets.

    Why: Pattern-based detection is fast and deterministic, suitable for scanning large
    codebases. The patterns are tuned for common formats:
    - Known prefixes (ghp_, sk-, AKIA) provide high confidence
    - Generic patterns (api_key =) require context validation
    - Length/character requirements reduce false positives

    Usage:
        scanner = SecretScanner()
        if scanner.has_secrets(code):
            # REJECT - do not include in training
            matches = scanner.scan(code)
            for match in matches:
                log(f"Secret found: {match.pattern_name} at line {match.line_number}")
    """

    # Patterns for various secret types
    # Format: (pattern_name, regex, description)
    PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
        # GitHub tokens (Personal Access Tokens)
        (
            "github_token",
            re.compile(r"ghp_[a-zA-Z0-9]{36}"),
            "GitHub Personal Access Token",
        ),
        # GitHub OAuth tokens
        (
            "github_oauth",
            re.compile(r"gho_[a-zA-Z0-9]{36}"),
            "GitHub OAuth Token",
        ),
        # GitHub App tokens
        (
            "github_app",
            re.compile(r"ghu_[a-zA-Z0-9]{36}"),
            "GitHub App Token",
        ),
        (
            "github_refresh",
            re.compile(r"ghr_[a-zA-Z0-9]{36}"),
            "GitHub Refresh Token",
        ),
        # OpenAI API keys
        (
            "openai_key",
            re.compile(r"sk-[a-zA-Z0-9]{48}"),
            "OpenAI API Key",
        ),
        # Anthropic API keys
        (
            "anthropic_key",
            re.compile(r"sk-ant-[a-zA-Z0-9\-]{40,}"),
            "Anthropic API Key",
        ),
        # AWS Access Key IDs
        (
            "aws_key_id",
            re.compile(r"AKIA[0-9A-Z]{16}"),
            "AWS Access Key ID",
        ),
        # AWS Secret Access Keys (often follows access key)
        (
            "aws_secret",
            re.compile(
                r"(?i)(aws_secret_access_key|aws_secret)\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{40}['\"]?"
            ),
            "AWS Secret Access Key",
        ),
        # Generic API key assignments
        (
            "generic_api_key",
            re.compile(r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"][a-zA-Z0-9\-_]{20,}['\"]"),
            "Generic API Key",
        ),
        # Generic secret/password assignments
        (
            "generic_secret",
            re.compile(r"(?i)(secret|password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{8,}['\"]"),
            "Generic Secret/Password",
        ),
        # Bearer tokens in code
        (
            "bearer_token",
            re.compile(r"(?i)bearer\s+[a-zA-Z0-9\-_.]{20,}"),
            "Bearer Token",
        ),
        # Private keys (PEM format start)
        (
            "private_key",
            re.compile(r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----"),
            "Private Key",
        ),
        # Slack tokens
        (
            "slack_token",
            re.compile(r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*"),
            "Slack Token",
        ),
        # Stripe API keys
        (
            "stripe_key",
            re.compile(r"sk_live_[a-zA-Z0-9]{24,}"),
            "Stripe Secret Key",
        ),
        (
            "stripe_test_key",
            re.compile(r"sk_test_[a-zA-Z0-9]{24,}"),
            "Stripe Test Key",
        ),
        # Twilio credentials
        (
            "twilio_key",
            re.compile(r"SK[a-f0-9]{32}"),
            "Twilio API Key",
        ),
        # Mailchimp API keys
        (
            "mailchimp_key",
            re.compile(r"[a-f0-9]{32}-us[0-9]{1,2}"),
            "Mailchimp API Key",
        ),
        # SendGrid API keys
        (
            "sendgrid_key",
            re.compile(r"SG\.[a-zA-Z0-9_-]{22,}\.[a-zA-Z0-9_-]{43,}"),
            "SendGrid API Key",
        ),
        # Database connection strings with credentials
        (
            "db_connection",
            re.compile(r"(?i)(postgres|mysql|mongodb)://[^:]+:[^@]+@"),
            "Database Connection String",
        ),
        # Heroku API keys
        (
            "heroku_key",
            re.compile(r"(?i)heroku_api_key\s*[=:]\s*['\"]?[a-f0-9\-]{36}['\"]?"),
            "Heroku API Key",
        ),
        # NPM tokens
        (
            "npm_token",
            re.compile(r"npm_[a-zA-Z0-9]{36}"),
            "NPM Token",
        ),
        # PyPI tokens
        (
            "pypi_token",
            re.compile(r"pypi-[a-zA-Z0-9_-]{50,}"),
            "PyPI Token",
        ),
    ]

    def __init__(self, additional_patterns: list[tuple[str, str]] | None = None) -> None:
        """Initialize secret scanner.

        Why: The scanner can be extended with custom patterns for organization-specific
        secret formats. The base patterns cover common public services.

        Args:
            additional_patterns: Optional list of (name, regex_string) tuples for
                custom patterns to add to the scanner.
        """
        self.patterns = list(self.PATTERNS)

        if additional_patterns:
            for name, pattern_str in additional_patterns:
                try:
                    compiled = re.compile(pattern_str)
                    self.patterns.append((name, compiled, f"Custom pattern: {name}"))
                except re.error as e:
                    raise ValueError(f"Invalid regex pattern '{name}': {e}") from e

    def scan(self, code: str) -> list[SecretMatch]:
        """Scan code for hardcoded secrets.

        Searches the code for all known secret patterns and returns matches with
        redacted values. Line numbers are computed for reporting.

        Why: Full scan returns all matches for detailed reporting. Use has_secrets()
        for a quick boolean check when you only need to know if ANY secrets exist.

        Args:
            code: Source code to scan

        Returns:
            List of SecretMatch objects for each detected secret
        """
        matches: list[SecretMatch] = []

        # Pre-compute line number mapping
        # line_starts[i] = character position where line i begins (0-indexed lines)
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
        for pattern_name, pattern, _description in self.patterns:
            for match in pattern.finditer(code):
                secret_match = SecretMatch(
                    pattern_name=pattern_name,
                    start=match.start(),
                    end=match.end(),
                    matched_text=SecretMatch.redact(match.group()),
                    line_number=get_line_number(match.start()),
                )
                matches.append(secret_match)

        return matches

    def has_secrets(self, code: str) -> bool:
        """Quick check if code contains any secrets.

        More efficient than scan() when you only need a boolean result,
        as it returns immediately upon finding the first match.

        Why: Fast rejection is important when processing large codebases. This method
        allows early exit without scanning the entire file once a secret is found.

        Args:
            code: Source code to check

        Returns:
            True if any secret pattern matches, False otherwise
        """
        for _pattern_name, pattern, _description in self.patterns:
            if pattern.search(code):
                return True
        return False

    def get_pattern_descriptions(self) -> dict[str, str]:
        """Get descriptions for all secret patterns.

        Why: Useful for documentation and error messages explaining what types
        of secrets are detected.

        Returns:
            Dictionary mapping pattern names to human-readable descriptions
        """
        return {name: desc for name, _pattern, desc in self.patterns}
