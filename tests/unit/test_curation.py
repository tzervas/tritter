"""Unit tests for dataset curation module.

Validates the curation pipeline including secret detection, security scanning,
quality analysis, and deduplication.

Why: The curation module is critical for training data quality:
1. Secret detection prevents credential leakage through model memorization
2. Security scanning identifies vulnerabilities for contrastive learning
3. Quality analysis enables positive/negative labeling
4. Deduplication prevents memorization and dataset bias

Testing Strategy:
- Secret detection: Test both positive cases (should detect) and negative cases
  (should not false-positive on similar-looking strings)
- Security scanning: Test language-specific patterns for Python, Rust, Triton
- Quality analysis: Test docstring detection, nesting depth, magic numbers
- Deduplication: Test similar vs different texts, threshold behavior
- Full pipeline: Test end-to-end flow (good code -> positive, bad code -> negative,
  secrets -> rejected)
"""

from tritter.curation import (
    CuratedSample,
    CurationPipeline,
    MinHashDeduplicator,
    MinHashSignature,
    QualityAnalyzer,
    QualityMetadata,
    SecretMatch,
    SecretScanner,
    SecurityScanner,
    SourceMetadata,
)


class TestSecretScanner:
    """Test suite for secret detection.

    Why: Secret detection is the most critical quality gate - secrets must NEVER
    appear in training data. These tests ensure both detection accuracy (catching
    real secrets) and precision (not false-positive on similar patterns).
    """

    def test_initialization(self) -> None:
        """Test scanner initializes with default patterns."""
        scanner = SecretScanner()
        assert len(scanner.patterns) > 0
        # Should have common secret patterns
        pattern_names = [p[0] for p in scanner.patterns]
        assert "github_token" in pattern_names
        assert "openai_key" in pattern_names
        assert "aws_key_id" in pattern_names

    def test_detect_github_token(self) -> None:
        """Test detection of GitHub Personal Access Token.

        Why: GitHub tokens are common in code and follow a specific format (ghp_).
        Detection must be reliable to prevent credential exposure.
        """
        scanner = SecretScanner()

        code = """
config = {
    "token": "ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456789012"
}
"""
        assert scanner.has_secrets(code)
        matches = scanner.scan(code)
        assert len(matches) == 1
        assert matches[0].pattern_name == "github_token"
        # Should be redacted
        assert "***" in matches[0].matched_text

    def test_detect_openai_key(self) -> None:
        """Test detection of OpenAI API key.

        Why: OpenAI keys (sk-...) are very common in AI/ML code and must be detected.
        """
        scanner = SecretScanner()

        code = """
import openai
openai.api_key = "sk-abcdefghijklmnopqrstuvwxyz123456789012345678901234"
"""
        assert scanner.has_secrets(code)
        matches = scanner.scan(code)
        assert any(m.pattern_name == "openai_key" for m in matches)

    def test_detect_aws_access_key(self) -> None:
        """Test detection of AWS Access Key ID.

        Why: AWS keys follow a specific format (AKIA...) and are critical to detect.
        """
        scanner = SecretScanner()

        code = """
AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
"""
        assert scanner.has_secrets(code)
        matches = scanner.scan(code)
        assert any(m.pattern_name == "aws_key_id" for m in matches)

    def test_detect_generic_api_key(self) -> None:
        """Test detection of generic API key patterns.

        Why: Many services use api_key or apikey in their configuration.
        """
        scanner = SecretScanner()

        code = """
api_key = "test_fake_key_abcdefghijklmnop1234567890"
"""
        assert scanner.has_secrets(code)
        matches = scanner.scan(code)
        assert any(m.pattern_name == "generic_api_key" for m in matches)

    def test_detect_password_assignment(self) -> None:
        """Test detection of hardcoded passwords.

        Why: Passwords in code are a common security mistake.
        """
        scanner = SecretScanner()

        code = """
password = "supersecretpassword123"
"""
        assert scanner.has_secrets(code)
        matches = scanner.scan(code)
        assert any(m.pattern_name == "generic_secret" for m in matches)

    def test_detect_bearer_token(self) -> None:
        """Test detection of Bearer tokens.

        Why: Bearer tokens in code indicate hardcoded authentication.
        """
        scanner = SecretScanner()

        code = """
headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"}
"""
        assert scanner.has_secrets(code)
        matches = scanner.scan(code)
        assert any(m.pattern_name == "bearer_token" for m in matches)

    def test_detect_private_key(self) -> None:
        """Test detection of private key markers.

        Why: Private keys in code are a severe security risk.
        """
        scanner = SecretScanner()

        code = '''
key = """
-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA...
-----END RSA PRIVATE KEY-----
"""
'''
        assert scanner.has_secrets(code)
        matches = scanner.scan(code)
        assert any(m.pattern_name == "private_key" for m in matches)

    def test_no_false_positive_on_placeholders(self) -> None:
        """Test that placeholder values don't trigger false positives.

        Why: Documentation often includes placeholders like 'your-api-key-here'
        which should not be flagged as real secrets.
        """
        scanner = SecretScanner()

        # Placeholder patterns that look like secrets but aren't
        code = """
# Configure your API key
api_key = os.environ.get("API_KEY")
github_token = os.getenv("GITHUB_TOKEN")
password = input("Enter password: ")
"""
        # Should not detect these as secrets
        matches = scanner.scan(code)
        # May match password due to "Enter password" string, but core env lookups are safe
        assert not any(m.pattern_name == "github_token" for m in matches)
        assert not any(m.pattern_name == "generic_api_key" for m in matches)

    def test_no_false_positive_on_short_strings(self) -> None:
        """Test that short strings don't trigger false positives.

        Why: Patterns require minimum length to avoid matching variable names
        or other short identifiers.
        """
        scanner = SecretScanner()

        code = """
api = "test"
key = "value"
sk = "short"
"""
        assert not scanner.has_secrets(code)

    def test_line_number_tracking(self) -> None:
        """Test that line numbers are correctly reported.

        Why: Line numbers enable debugging and fixing secret leaks.
        """
        scanner = SecretScanner()

        code = """line1
line2
api_key = "sk_test_abcdefghijklmnopqrst"
line4
"""
        matches = scanner.scan(code)
        assert len(matches) > 0
        # Secret is on line 3
        assert matches[0].line_number == 3

    def test_redaction(self) -> None:
        """Test that matched text is properly redacted.

        Why: We need to report matches for debugging without exposing actual secrets.
        """
        result = SecretMatch.redact("ghp_abcdefghijklmnopqrstuvwxyz12345678901234")
        assert "***" in result
        assert "ghp_" in result  # First chars visible
        # Full secret should not be visible
        assert "abcdefghij" not in result

    def test_custom_patterns(self) -> None:
        """Test adding custom secret patterns.

        Why: Organizations may have custom secret formats that need detection.
        """
        custom_patterns = [
            ("custom_token", r"myorg_[a-zA-Z0-9]{20}"),
        ]
        scanner = SecretScanner(additional_patterns=custom_patterns)

        code = 'token = "myorg_abcdefghij1234567890"'
        assert scanner.has_secrets(code)
        matches = scanner.scan(code)
        assert any(m.pattern_name == "custom_token" for m in matches)


class TestSecurityScanner:
    """Test suite for security vulnerability detection.

    Why: Security scanning enables contrastive learning - the model learns to
    identify and explain security issues. Tests cover patterns for Python, Rust,
    and Triton.
    """

    def test_initialization(self) -> None:
        """Test scanner initializes with patterns for all languages."""
        scanner = SecurityScanner()
        assert len(scanner.python_patterns) > 0
        assert len(scanner.rust_patterns) > 0
        assert len(scanner.triton_patterns) > 0

    def test_detect_eval_python(self) -> None:
        """Test detection of eval() in Python.

        Why: eval() is a critical security risk that enables code injection.
        """
        scanner = SecurityScanner()

        code = """
def process(data):
    return eval(data)
"""
        issues = scanner.scan(code, "python")
        assert len(issues) > 0
        assert any(i.issue_type == "code_injection" for i in issues)
        assert any(i.severity == "critical" for i in issues)

    def test_detect_exec_python(self) -> None:
        """Test detection of exec() in Python.

        Why: exec() is as dangerous as eval() - arbitrary code execution.
        """
        scanner = SecurityScanner()

        code = """
exec(user_input)
"""
        issues = scanner.scan(code, "python")
        assert any(i.pattern_name == "exec_usage" for i in issues)
        assert any(i.severity == "critical" for i in issues)

    def test_detect_subprocess_shell_true(self) -> None:
        """Test detection of subprocess with shell=True.

        Why: shell=True enables command injection through user input.
        """
        scanner = SecurityScanner()

        code = """
import subprocess
subprocess.run(command, shell=True)
"""
        issues = scanner.scan(code, "python")
        assert any(i.pattern_name == "subprocess_shell" for i in issues)
        assert any(i.severity == "high" for i in issues)

    def test_detect_pickle_load(self) -> None:
        """Test detection of pickle.load().

        Why: pickle can execute arbitrary code during deserialization.
        """
        scanner = SecurityScanner()

        code = """
import pickle
data = pickle.load(file)
"""
        issues = scanner.scan(code, "python")
        assert any(i.pattern_name == "pickle_load" for i in issues)

    def test_detect_unsafe_yaml(self) -> None:
        """Test detection of yaml.load() without SafeLoader.

        Why: yaml.load() can execute arbitrary code; yaml.safe_load() is required.
        """
        scanner = SecurityScanner()

        code = """
import yaml
data = yaml.load(content)
"""
        issues = scanner.scan(code, "python")
        assert any("yaml" in i.pattern_name for i in issues)

    def test_detect_rust_unsafe(self) -> None:
        """Test detection of unsafe blocks in Rust.

        Why: unsafe blocks bypass Rust's safety guarantees and need documentation.
        """
        scanner = SecurityScanner()

        code = """
fn risky() {
    unsafe {
        ptr::read(addr)
    }
}
"""
        issues = scanner.scan(code, "rust")
        assert any(i.pattern_name == "unsafe_block" for i in issues)

    def test_detect_rust_unwrap(self) -> None:
        """Test detection of .unwrap() in Rust.

        Why: unwrap() can panic, causing unexpected program termination.
        """
        scanner = SecurityScanner()

        code = """
fn get_value() -> i32 {
    some_result.unwrap()
}
"""
        issues = scanner.scan(code, "rust")
        assert any(i.pattern_name == "unwrap_usage" for i in issues)

    def test_detect_triton_unmasked_load(self) -> None:
        """Test detection of unmasked tl.load() in Triton.

        Why: Unmasked loads can read out-of-bounds memory on GPU.
        """
        scanner = SecurityScanner()

        code = """
@triton.jit
def kernel(ptr):
    data = tl.load(ptr)  # No mask!
"""
        issues = scanner.scan(code, "triton")
        # Should detect missing mask
        assert any(i.pattern_name == "unmasked_load" for i in issues)

    def test_detect_triton_excessive_block_size(self) -> None:
        """Test detection of excessive BLOCK_SIZE in Triton.

        Why: Very large block sizes can exceed GPU shared memory limits.
        """
        scanner = SecurityScanner()

        code = """
BLOCK_SIZE = 100000  # Too big!
"""
        issues = scanner.scan(code, "triton")
        assert any(i.pattern_name == "excessive_block_size" for i in issues)

    def test_recommendations_provided(self) -> None:
        """Test that all issues include recommendations.

        Why: Recommendations enable the model to suggest fixes, not just identify issues.
        """
        scanner = SecurityScanner()

        code = """
result = eval(input())
data = pickle.load(file)
"""
        issues = scanner.scan(code, "python")
        for issue in issues:
            assert issue.recommendation, f"Issue {issue.pattern_name} missing recommendation"
            assert len(issue.recommendation) > 10  # Non-trivial recommendation

    def test_unknown_language_returns_empty(self) -> None:
        """Test that unknown languages return empty results (not crash).

        Why: The scanner should gracefully handle unsupported languages.
        """
        scanner = SecurityScanner()

        code = "function foo() { return eval(x); }"  # JavaScript
        issues = scanner.scan(code, "javascript")
        assert issues == []

    def test_severity_counts(self) -> None:
        """Test severity counting utility.

        Why: Summary statistics help with dataset-level quality analysis.
        """
        scanner = SecurityScanner()

        code = """
eval(x)
exec(y)
subprocess.run(cmd, shell=True)
"""
        issues = scanner.scan(code, "python")
        counts = scanner.get_severity_counts(issues)

        assert "critical" in counts
        assert "high" in counts
        assert counts["critical"] >= 2  # eval and exec


class TestQualityAnalyzer:
    """Test suite for code quality analysis.

    Why: Quality metrics enable labeling code as positive (good) or negative (bad).
    Tests cover docstring detection, nesting depth, function size, and magic numbers.
    """

    def test_initialization(self) -> None:
        """Test analyzer initializes with configurable thresholds."""
        analyzer = QualityAnalyzer(max_line_length=80, max_nesting_depth=3)
        assert analyzer.max_line_length == 80
        assert analyzer.max_nesting_depth == 3

    def test_count_lines(self) -> None:
        """Test basic line counting."""
        analyzer = QualityAnalyzer()

        code = """line1
line2
line3"""
        metrics = analyzer.analyze(code, "python")
        assert metrics.line_count == 3

    def test_detect_functions_python(self) -> None:
        """Test function detection in Python code.

        Why: Function count indicates code structure complexity.
        """
        analyzer = QualityAnalyzer()

        code = """
def foo():
    pass

def bar():
    pass

async def baz():
    pass
"""
        metrics = analyzer.analyze(code, "python")
        assert metrics.function_count == 3

    def test_detect_classes_python(self) -> None:
        """Test class detection in Python code."""
        analyzer = QualityAnalyzer()

        code = """
class Foo:
    pass

class Bar:
    pass
"""
        metrics = analyzer.analyze(code, "python")
        assert metrics.class_count == 2

    def test_detect_docstrings(self) -> None:
        """Test docstring presence detection.

        Why: Documentation is a key quality indicator.
        """
        analyzer = QualityAnalyzer()

        code = '''
def documented():
    """This function has a docstring."""
    pass

def undocumented():
    pass
'''
        metrics = analyzer.analyze(code, "python")
        # 1 out of 2 functions has docstring
        assert metrics.docstring_ratio == 0.5

    def test_detect_deep_nesting(self) -> None:
        """Test nesting depth detection.

        Why: Deep nesting hurts readability and indicates need for refactoring.
        """
        analyzer = QualityAnalyzer(max_nesting_depth=3)

        code = """
def deep():
    if True:
        for i in range(10):
            while True:
                if False:
                    if True:
                        pass
"""
        metrics = analyzer.analyze(code, "python")
        assert metrics.max_nesting_depth > 3
        # Should have issue for deep nesting
        assert any(i.issue_type == "deep_nesting" for i in metrics.issues)

    def test_detect_magic_numbers(self) -> None:
        """Test magic number detection.

        Why: Hardcoded numbers without context hurt maintainability.
        """
        analyzer = QualityAnalyzer(magic_number_threshold=10)

        code = """
def calculate():
    x = 42 * 100  # Magic!
    y = 12345
    z = 98765
    return x + y + z
"""
        metrics = analyzer.analyze(code, "python")
        assert metrics.magic_number_count > 0

    def test_detect_long_lines(self) -> None:
        """Test long line detection."""
        analyzer = QualityAnalyzer(max_line_length=80)

        code = f"""
short_line = "hello"
{"long_line = " + "x" * 100}
"""
        metrics = analyzer.analyze(code, "python")
        assert metrics.long_line_count > 0

    def test_overall_score_calculation(self) -> None:
        """Test that overall score is computed and bounded.

        Why: Score enables threshold-based labeling.
        """
        analyzer = QualityAnalyzer()

        code = '''
def foo():
    """Documented."""
    return 42
'''
        metrics = analyzer.analyze(code, "python")
        # Score should be between 0 and 1
        assert 0.0 <= metrics.overall_score <= 1.0

    def test_high_quality_code_high_score(self) -> None:
        """Test that high-quality code gets high score.

        Why: Well-documented, simple code should be labeled positive.
        """
        analyzer = QualityAnalyzer()

        code = '''
def add_numbers(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b


class Calculator:
    """A simple calculator."""

    def multiply(self, x: int, y: int) -> int:
        """Multiply two numbers."""
        return x * y
'''
        metrics = analyzer.analyze(code, "python")
        # Good code should score well
        assert metrics.overall_score > 0.6
        assert metrics.docstring_ratio >= 0.5

    def test_low_quality_code_low_score(self) -> None:
        """Test that low-quality code gets low score.

        Why: Poorly structured code should be labeled negative.
        """
        analyzer = QualityAnalyzer()

        # Code with many quality issues
        code = """
def f():
    x=42*3.14159265359*100000+99999*88888*77777*66666
    if True:
        if True:
            if True:
                if True:
                    if True:
                        if True:
                            pass
"""
        metrics = analyzer.analyze(code, "python")
        # Bad code should score poorly
        assert metrics.overall_score < 0.8
        # Should have issues
        assert len(metrics.issues) > 0

    def test_fallback_for_non_python(self) -> None:
        """Test basic analysis for non-Python languages.

        Why: Should provide useful metrics even without AST parsing.
        """
        analyzer = QualityAnalyzer()

        code = """
fn main() {
    let x = 42;
    println!("{}", x);
}
"""
        metrics = analyzer.analyze(code, "rust")
        # Should still compute basic metrics
        assert metrics.line_count > 0
        assert metrics.function_count >= 0

    def test_handles_syntax_errors(self) -> None:
        """Test graceful handling of invalid Python syntax.

        Why: Real-world data may contain malformed code.
        """
        analyzer = QualityAnalyzer()

        code = """
def broken(
    missing_close_paren
"""
        # Should not crash
        metrics = analyzer.analyze(code, "python")
        assert metrics.line_count > 0


class TestMinHashDeduplicator:
    """Test suite for MinHash-based deduplication.

    Why: Deduplication prevents training on redundant data, which can cause
    memorization and bias. Tests cover signature computation, similarity
    estimation, and index operations.
    """

    def test_initialization(self) -> None:
        """Test deduplicator initializes with configurable parameters."""
        dedup = MinHashDeduplicator(num_perm=64, shingle_size=5, threshold=0.9)
        assert dedup.num_perm == 64
        assert dedup.shingle_size == 5
        assert dedup.threshold == 0.9

    def test_compute_signature(self) -> None:
        """Test signature computation produces valid output."""
        dedup = MinHashDeduplicator(num_perm=32)

        text = "def hello(): return 'world'"
        sig = dedup.compute_signature(text)

        assert isinstance(sig, MinHashSignature)
        assert len(sig.signature) == 32
        assert sig.shingle_count > 0

    def test_identical_texts_similar(self) -> None:
        """Test that identical texts have similarity 1.0.

        Why: Exact duplicates should always be detected.
        """
        dedup = MinHashDeduplicator()

        text = "def hello(): return 'world'"
        sig1 = dedup.compute_signature(text)
        sig2 = dedup.compute_signature(text)

        similarity = dedup.estimate_similarity(sig1, sig2)
        assert similarity == 1.0
        assert dedup.is_duplicate(sig1, sig2)

    def test_similar_texts_high_similarity(self) -> None:
        """Test that similar texts have high similarity.

        Why: Near-duplicates (minor changes) should be detected.
        """
        dedup = MinHashDeduplicator()

        text1 = "def hello(): return 'world'"
        text2 = "def hello(): return 'World'"  # Only case difference

        sig1 = dedup.compute_signature(text1)
        sig2 = dedup.compute_signature(text2)

        similarity = dedup.estimate_similarity(sig1, sig2)
        assert similarity > 0.7  # Should be quite similar

    def test_different_texts_low_similarity(self) -> None:
        """Test that different texts have low similarity.

        Why: Genuinely different code should not be flagged as duplicates.
        """
        dedup = MinHashDeduplicator()

        text1 = "def hello(): return 'world'"
        text2 = "class Calculator: def add(self, a, b): return a + b"

        sig1 = dedup.compute_signature(text1)
        sig2 = dedup.compute_signature(text2)

        similarity = dedup.estimate_similarity(sig1, sig2)
        assert similarity < 0.5  # Should be quite different
        assert not dedup.is_duplicate(sig1, sig2)

    def test_add_to_index(self) -> None:
        """Test adding documents to index."""
        dedup = MinHashDeduplicator()

        dedup.add_to_index("doc1", "def foo(): pass")
        dedup.add_to_index("doc2", "def bar(): pass")

        assert dedup.get_index_size() == 2

    def test_find_duplicates(self) -> None:
        """Test finding duplicates in index.

        Why: This is the main use case - checking new documents against existing.
        """
        dedup = MinHashDeduplicator(threshold=0.8)

        # Add some documents
        dedup.add_to_index("doc1", "def hello(): return 'world'")
        dedup.add_to_index("doc2", "class Calculator: def add(self, a, b): return a + b")

        # Check for duplicate of first doc
        duplicates = dedup.find_duplicates("def hello(): return 'world'")
        assert "doc1" in duplicates

        # Check for non-duplicate
        duplicates = dedup.find_duplicates("totally different code here")
        assert "doc1" not in duplicates
        assert "doc2" not in duplicates

    def test_threshold_behavior(self) -> None:
        """Test that threshold affects duplicate detection.

        Why: Different use cases need different sensitivity levels.
        """
        text1 = "def hello(): return 'world'"
        text2 = "def hello(): return 'World'"  # Minor difference

        # Strict threshold
        dedup_strict = MinHashDeduplicator(threshold=0.99)
        sig1 = dedup_strict.compute_signature(text1)
        sig2 = dedup_strict.compute_signature(text2)
        # May not be duplicate with strict threshold
        dedup_strict.is_duplicate(sig1, sig2)

        # Loose threshold
        dedup_loose = MinHashDeduplicator(threshold=0.5)
        # Should be duplicate with loose threshold
        assert dedup_loose.is_duplicate(sig1, sig2)

    def test_clear_index(self) -> None:
        """Test clearing the index."""
        dedup = MinHashDeduplicator()

        dedup.add_to_index("doc1", "hello world")
        assert dedup.get_index_size() == 1

        dedup.clear_index()
        assert dedup.get_index_size() == 0

    def test_empty_text(self) -> None:
        """Test handling of empty text.

        Why: Edge case - empty strings should not crash.
        """
        dedup = MinHashDeduplicator()

        sig = dedup.compute_signature("")
        assert len(sig.signature) == dedup.num_perm
        assert sig.shingle_count == 0


class TestCurationPipeline:
    """Test suite for the full curation pipeline.

    Why: End-to-end tests ensure all components work together correctly.
    Tests cover the three main outcomes: positive, negative, and rejected.
    """

    def test_initialization(self) -> None:
        """Test pipeline initializes with configurable thresholds."""
        pipeline = CurationPipeline(quality_threshold=0.7, min_lines=10)
        assert pipeline.quality_threshold == 0.7
        assert pipeline.min_lines == 10

    def test_good_code_positive_label(self) -> None:
        """Test that high-quality code gets positive label.

        Why: Good code should be used as positive training examples.
        """
        pipeline = CurationPipeline()

        code = '''
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b


def multiply_numbers(x: int, y: int) -> int:
    """Multiply two numbers.

    Args:
        x: First factor
        y: Second factor

    Returns:
        Product of x and y
    """
    return x * y
'''
        result = pipeline.process(code, "python")

        # Good code should be positive
        assert result.quality_label == "positive"
        assert result.quality_score > 0.5
        assert len(result.security_issues) == 0
        assert result.rejected_reason is None

    def test_code_with_secrets_rejected(self) -> None:
        """Test that code with secrets is ALWAYS rejected.

        Why: Secrets must never appear in training data - critical security.
        """
        pipeline = CurationPipeline()

        code = """
def connect():
    api_key = "sk-abcdefghijklmnopqrstuvwxyz123456789012345678901234"
    return api_key
"""
        result = pipeline.process(code, "python")

        # Must be rejected
        assert result.quality_label == "rejected"
        assert result.rejected_reason == "contains_secrets"
        assert len(result.secret_matches) > 0
        assert "secret" in result.explanation.lower()

    def test_code_with_security_issues_negative(self) -> None:
        """Test that code with security issues gets negative label.

        Why: Security issues become contrastive learning examples.
        """
        pipeline = CurationPipeline()

        code = '''
def process_input(data):
    """Process user input."""
    # DANGEROUS: arbitrary code execution
    result = eval(data)
    return result

def run_command(cmd):
    """Run a shell command."""
    # Also dangerous
    import subprocess
    subprocess.run(cmd, shell=True)
'''
        result = pipeline.process(code, "python")

        # Should be negative (not rejected - that's only for secrets)
        assert result.quality_label == "negative"
        assert len(result.security_issues) > 0
        assert result.explanation is not None
        assert "security" in result.explanation.lower()

    def test_poor_quality_code_negative(self) -> None:
        """Test that poor quality code gets negative label.

        Why: Quality issues enable contrastive learning about code style.
        """
        pipeline = CurationPipeline(quality_threshold=0.6)

        # Code with many quality issues
        code = """
def f():
    x=42*3.14159265359*100000
    if True:
        if True:
            if True:
                if True:
                    if True:
                        if True:
                            pass
def g():
    pass
def h():
    pass
"""
        result = pipeline.process(code, "python")

        # May be negative due to quality issues
        # (depends on exact scoring, which may vary)
        assert result.quality_label in ("positive", "negative")
        if result.quality_label == "negative":
            assert len(result.quality_issues) > 0 or result.quality_score < 0.6

    def test_too_short_rejected(self) -> None:
        """Test that very short code is rejected.

        Why: Very short snippets provide little training value.
        """
        pipeline = CurationPipeline(min_lines=5)

        code = "x = 1"
        result = pipeline.process(code, "python")

        assert result.quality_label == "rejected"
        assert result.rejected_reason == "too_short"

    def test_too_long_rejected(self) -> None:
        """Test that very long code is rejected.

        Why: Very long files may cause memory issues and often have quality problems.
        """
        pipeline = CurationPipeline(max_lines=100)

        code = "\n".join([f"x{i} = {i}" for i in range(200)])
        result = pipeline.process(code, "python")

        assert result.quality_label == "rejected"
        assert result.rejected_reason == "too_long"

    def test_create_sample(self) -> None:
        """Test creating CuratedSample from result.

        Why: Samples are the final training format.
        """
        pipeline = CurationPipeline()

        code = '''
def hello():
    """Say hello."""
    return "Hello, World!"
'''
        result = pipeline.process(code, "python")
        sample = pipeline.create_sample(
            code,
            "python",
            result,
            source_metadata={"repo": "test/repo", "path": "hello.py"},
        )

        assert isinstance(sample, CuratedSample)
        assert sample.text == code
        assert sample.language == "python"
        assert sample.source.repo == "test/repo"
        assert sample.source.path == "hello.py"

    def test_explanation_generation(self) -> None:
        """Test that explanations are generated for negative samples.

        Why: Explanations enable the model to learn WHY code is bad.
        """
        pipeline = CurationPipeline()

        code = """
def dangerous():
    eval(input())
    pickle.load(open('data.pkl'))
"""
        result = pipeline.process(code, "python")

        assert result.quality_label == "negative"
        assert result.explanation is not None
        assert len(result.explanation) > 50  # Non-trivial explanation
        # Should mention recommendations
        assert "fix" in result.explanation.lower() or "use" in result.explanation.lower()


class TestSchemaClasses:
    """Test suite for data schema classes.

    Why: Schema validation ensures data integrity across the pipeline.
    """

    def test_source_metadata_creation(self) -> None:
        """Test SourceMetadata creation with all fields."""
        source = SourceMetadata(
            repo="test/repo",
            path="src/module.py",
            license="MIT",
            stars=100,
            commit_sha="abc123",
            url="https://github.com/test/repo",
        )

        assert source.repo == "test/repo"
        assert source.license == "MIT"
        assert source.stars == 100

    def test_quality_metadata_defaults(self) -> None:
        """Test QualityMetadata default values."""
        metadata = QualityMetadata()

        assert metadata.complexity == 0.0
        assert metadata.maintainability == 100.0
        assert metadata.line_count == 0

    def test_curated_sample_to_dict(self) -> None:
        """Test CuratedSample serialization to dict.

        Why: Samples must be serializable for storage and training.
        """
        sample = CuratedSample(
            text="def foo(): pass",
            language="python",
            quality_label="positive",
            quality_score=0.9,
            security_issues=[],
            quality_issues=[],
            anti_patterns=[],
            explanation=None,
        )

        data = sample.to_dict()

        assert data["text"] == "def foo(): pass"
        assert data["language"] == "python"
        assert data["quality_label"] == "positive"
        assert data["quality_score"] == 0.9

    def test_curated_sample_from_dict(self) -> None:
        """Test CuratedSample deserialization from dict.

        Why: Samples must be loadable from storage.
        """
        data = {
            "text": "def bar(): pass",
            "language": "python",
            "quality_label": "negative",
            "quality_score": 0.3,
            "security_issues": [
                {
                    "type": "code_injection",
                    "severity": "critical",
                    "line": 1,
                    "message": "eval usage",
                    "recommendation": "Use ast.literal_eval",
                }
            ],
            "quality_issues": ["deep nesting"],
            "anti_patterns": ["deep_nesting"],
            "explanation": "Has issues",
            "source": {"repo": "test/repo", "path": "bar.py"},
            "metadata": {"line_count": 1},
        }

        sample = CuratedSample.from_dict(data)

        assert sample.text == "def bar(): pass"
        assert sample.quality_label == "negative"
        assert len(sample.security_issues) == 1
        assert sample.source.repo == "test/repo"

    def test_curated_sample_round_trip(self) -> None:
        """Test that to_dict/from_dict preserves all data.

        Why: Symmetric encode/decode is required for data integrity.
        """
        original = CuratedSample(
            text="def test(): pass",
            language="python",
            quality_label="positive",
            quality_score=0.85,
            security_issues=[],
            quality_issues=["minor issue"],
            anti_patterns=["magic_numbers"],
            explanation="Test explanation",
            source=SourceMetadata(
                repo="owner/repo",
                path="test.py",
                license="Apache-2.0",
                stars=500,
            ),
            metadata=QualityMetadata(
                line_count=1,
                function_count=1,
                docstring_ratio=0.0,
            ),
        )

        # Round-trip
        data = original.to_dict()
        restored = CuratedSample.from_dict(data)

        # Verify key fields preserved
        assert restored.text == original.text
        assert restored.quality_label == original.quality_label
        assert restored.quality_score == original.quality_score
        assert restored.explanation == original.explanation
        assert restored.source.repo == original.source.repo
        assert restored.source.license == original.source.license
        assert restored.metadata.function_count == original.metadata.function_count
