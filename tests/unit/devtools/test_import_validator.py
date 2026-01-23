"""Unit tests for devtools.import_validator.

Why:
    The import validator ensures __all__ exports match actual imports.
    These tests verify the validation logic catches common issues without
    false positives.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from devtools.import_validator import ImportValidator, validate_imports


class TestImportValidator:
    """Tests for ImportValidator class."""

    def test_valid_module_passes(self) -> None:
        """Test that a properly structured module passes validation.

        Why: Validates the happy path where __all__ matches imports exactly.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = Path(tmpdir) / "__init__.py"
            init_file.write_text('''
"""Test module."""
from .submodule import foo, bar

__all__ = ["foo", "bar"]
''')

            validator = ImportValidator()
            issues = validator.validate_file(init_file)

            # Should have no critical issues (undefined_export)
            critical = [i for i in issues if i.issue_type == "undefined_export"]
            assert len(critical) == 0, f"Unexpected critical issues: {critical}"

    def test_undefined_export_detected(self) -> None:
        """Test that exports without imports are caught.

        Why: This is the most common error - declaring something in __all__
        but forgetting to import it.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = Path(tmpdir) / "__init__.py"
            init_file.write_text('''
"""Test module."""
from .submodule import foo

__all__ = ["foo", "bar"]  # bar is not imported!
''')

            validator = ImportValidator()
            issues = validator.validate_file(init_file)

            undefined = [i for i in issues if i.issue_type == "undefined_export"]
            assert len(undefined) == 1
            assert undefined[0].name == "bar"

    def test_missing_all_detected(self) -> None:
        """Test that missing __all__ is flagged.

        Why: Per DEVELOPMENT_STANDARDS.md, every __init__.py should have __all__.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = Path(tmpdir) / "__init__.py"
            init_file.write_text('''
"""Test module without __all__."""
from .submodule import foo
''')

            validator = ImportValidator()
            issues = validator.validate_file(init_file)

            missing_all = [i for i in issues if i.issue_type == "missing_all"]
            assert len(missing_all) == 1

    def test_syntax_error_handled(self) -> None:
        """Test that syntax errors don't crash the validator.

        Why: Project analysis should gracefully handle broken files.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = Path(tmpdir) / "__init__.py"
            init_file.write_text("def broken(:\n    pass")  # Invalid syntax

            validator = ImportValidator()
            issues = validator.validate_file(init_file)

            syntax_errors = [i for i in issues if i.issue_type == "syntax_error"]
            assert len(syntax_errors) == 1

    def test_definitions_detected(self) -> None:
        """Test that local definitions can be exported.

        Why: __all__ can export locally defined classes/functions, not just imports.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = Path(tmpdir) / "__init__.py"
            init_file.write_text('''
"""Test module with local definitions."""

class MyClass:
    pass

def my_function():
    pass

__all__ = ["MyClass", "my_function"]
''')

            validator = ImportValidator()
            issues = validator.validate_file(init_file)

            critical = [i for i in issues if i.issue_type == "undefined_export"]
            assert len(critical) == 0


class TestValidateImports:
    """Tests for the validate_imports convenience function."""

    def test_path_not_found(self) -> None:
        """Test handling of non-existent paths.

        Why: Should handle gracefully when path doesn't exist.
        """
        result = validate_imports(Path("/nonexistent/path"))
        assert result.files_checked == 0

    def test_strict_mode_filters_warnings(self) -> None:
        """Test that strict mode filters out informational warnings.

        Why: Strict mode is for CI where only critical errors should fail.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = Path(tmpdir) / "__init__.py"
            # No __all__ - normally a warning
            init_file.write_text('"""Module."""\nfrom os import path\n')

            # Non-strict: includes warning
            result = validate_imports(Path(tmpdir), strict=False)
            has_warning = any(i.issue_type == "missing_all" for i in result.issues)
            assert has_warning

            # Strict: filters out warning
            result_strict = validate_imports(Path(tmpdir), strict=True)
            has_warning_strict = any(i.issue_type == "missing_all" for i in result_strict.issues)
            assert not has_warning_strict
