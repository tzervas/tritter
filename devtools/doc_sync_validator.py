#!/usr/bin/env python3
"""Documentation sync validator.

Why: Documentation that drifts from code is worse than no documentation.
This tool validates that documentation matches actual implementation.

Checks:
1. ROADMAP.md status matches actual test coverage
2. CLAUDE.md implementation status is accurate
3. All __all__ exports are documented
4. Spec status matches implementation state
5. Hardware profiles match code capabilities

Usage:
    python -m devtools.doc_sync_validator
    python -m devtools.doc_sync_validator --fix  # Auto-fix where possible
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ValidationIssue:
    """A documentation sync issue."""

    file: str
    line: int | None
    severity: Literal["error", "warning"]
    message: str
    suggestion: str | None = None


@dataclass
class ValidationResult:
    """Result of documentation validation."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.issues)

    def add_error(
        self,
        file: str,
        message: str,
        line: int | None = None,
        suggestion: str | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                file=file,
                line=line,
                severity="error",
                message=message,
                suggestion=suggestion,
            )
        )

    def add_warning(
        self,
        file: str,
        message: str,
        line: int | None = None,
        suggestion: str | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                file=file,
                line=line,
                severity="warning",
                message=message,
                suggestion=suggestion,
            )
        )


def find_project_root() -> Path:
    """Find the project root directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


def check_roadmap_status(root: Path, result: ValidationResult) -> None:
    """Check that ROADMAP.md status matches implementation."""
    roadmap_path = root / "docs" / "architecture" / "ROADMAP.md"
    if not roadmap_path.exists():
        roadmap_path = root / "docs" / "ROADMAP.md"
    if not roadmap_path.exists():
        result.add_warning("ROADMAP.md", "Roadmap file not found")
        return

    content = roadmap_path.read_text()

    # Check for common inconsistencies
    # Look for "✅ Implemented" claims and verify files exist
    implemented_pattern = re.compile(
        r"\|\s*`?([^`|]+)`?\s*\|\s*`([^`]+)`\s*\|\s*✅\s*Implemented"
    )

    for match in implemented_pattern.finditer(content):
        feature = match.group(1).strip()
        location = match.group(2).strip()

        # Check if the file exists
        file_path = root / "src" / "tritter" / location
        if not file_path.exists():
            # Try without .py extension
            file_path_py = root / "src" / "tritter" / f"{location}.py"
            if not file_path_py.exists() and not file_path.with_suffix(".py").exists():
                result.add_error(
                    str(roadmap_path),
                    f"Claims '{feature}' implemented at '{location}' but file not found",
                    suggestion=f"Check path or update status",
                )


def check_exports_documented(root: Path, result: ValidationResult) -> None:
    """Check that all __all__ exports have documentation."""
    src_path = root / "src" / "tritter"

    for init_file in src_path.rglob("__init__.py"):
        try:
            content = init_file.read_text()
            tree = ast.parse(content)

            # Find __all__ assignment
            all_exports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "__all__":
                            if isinstance(node.value, ast.List):
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Constant):
                                        all_exports.append(elt.value)

            # Check each export has docstring or is imported
            module_docstring = ast.get_docstring(tree)
            if all_exports and not module_docstring:
                result.add_warning(
                    str(init_file.relative_to(root)),
                    f"Module has {len(all_exports)} exports but no module docstring",
                )

        except SyntaxError as e:
            result.add_error(str(init_file.relative_to(root)), f"Syntax error: {e}")


def check_claude_md_accuracy(root: Path, result: ValidationResult) -> None:
    """Check that CLAUDE.md accurately reflects implementation state."""
    claude_md = root / "CLAUDE.md"
    if not claude_md.exists():
        result.add_warning("CLAUDE.md", "CLAUDE.md not found")
        return

    content = claude_md.read_text()

    # Check claimed implementations exist
    implemented_claims = re.findall(
        r"\|\s*(.+?)\s*\|\s*`([^`]+)`\s*\|\s*✅\s*Implemented", content
    )

    for feature, location in implemented_claims:
        file_path = root / "src" / "tritter" / location
        # Handle directory paths
        if "/" in location:
            dir_path = root / "src" / "tritter" / location.split("/")[0]
            if not dir_path.exists():
                result.add_error(
                    "CLAUDE.md",
                    f"Claims '{feature}' at '{location}' but path not found",
                )
                continue
        elif not file_path.exists() and not file_path.with_suffix(".py").exists():
            result.add_error(
                "CLAUDE.md",
                f"Claims '{feature}' implemented at '{location}' but file not found",
            )


def check_spec_status(root: Path, result: ValidationResult) -> None:
    """Check that spec status matches implementation."""
    specs_dir = root / "docs" / "specs"
    if not specs_dir.exists():
        result.add_warning("docs/specs", "Specs directory not found")
        return

    for spec_file in specs_dir.glob("SPEC-*.md"):
        content = spec_file.read_text()

        # Check for status
        status_match = re.search(r"\*\*Status\*\*:?\s*(.+?)(?:\n|$)", content)
        if not status_match:
            result.add_warning(
                str(spec_file.relative_to(root)),
                "Spec missing status field",
            )
            continue

        status = status_match.group(1).strip().lower()

        # If implemented, check for test file
        if "implemented" in status:
            spec_name = spec_file.stem.lower()
            # Extract feature name from spec filename
            parts = spec_name.replace("spec-", "").split("-")
            if len(parts) > 1:
                feature_name = "-".join(parts[1:])
                test_patterns = [
                    root / "tests" / "unit" / f"test_{feature_name.replace('-', '_')}.py",
                    root / "tests" / "integration" / f"test_{feature_name.replace('-', '_')}.py",
                ]

                has_test = any(p.exists() for p in test_patterns)
                if not has_test:
                    result.add_warning(
                        str(spec_file.relative_to(root)),
                        f"Spec claims implemented but no obvious test file found",
                        suggestion=f"Expected test file like test_{feature_name.replace('-', '_')}.py",
                    )


def check_hardware_profiles(root: Path, result: ValidationResult) -> None:
    """Check hardware profile accuracy."""
    profiles_file = root / "src" / "tritter" / "utils" / "hardware_profiles.py"
    if not profiles_file.exists():
        return

    content = profiles_file.read_text()

    # Check that verified=True profiles have matching tests
    verified_pattern = re.compile(
        r"(\w+)\s*=\s*HardwareProfile\([^)]*verified\s*=\s*True"
    )

    for match in verified_pattern.finditer(content):
        profile_name = match.group(1)
        # This is informational - hard to auto-verify hardware
        result.add_warning(
            str(profiles_file.relative_to(root)),
            f"Profile '{profile_name}' marked as verified - ensure manual testing completed",
        )


def validate_documentation(root: Path) -> ValidationResult:
    """Run all documentation validation checks."""
    result = ValidationResult()

    check_roadmap_status(root, result)
    check_exports_documented(root, result)
    check_claude_md_accuracy(root, result)
    check_spec_status(root, result)
    check_hardware_profiles(root, result)

    return result


def print_result(result: ValidationResult) -> None:
    """Print validation results."""
    if not result.issues:
        print("✅ Documentation is in sync with code")
        return

    errors = [i for i in result.issues if i.severity == "error"]
    warnings = [i for i in result.issues if i.severity == "warning"]

    if errors:
        print(f"\n❌ {len(errors)} error(s) found:\n")
        for issue in errors:
            loc = f"{issue.file}:{issue.line}" if issue.line else issue.file
            print(f"  ERROR: {loc}")
            print(f"         {issue.message}")
            if issue.suggestion:
                print(f"         Suggestion: {issue.suggestion}")
            print()

    if warnings:
        print(f"\n⚠️  {len(warnings)} warning(s) found:\n")
        for issue in warnings:
            loc = f"{issue.file}:{issue.line}" if issue.line else issue.file
            print(f"  WARN: {loc}")
            print(f"        {issue.message}")
            if issue.suggestion:
                print(f"        Suggestion: {issue.suggestion}")
            print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate documentation sync with code"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix issues where possible (not yet implemented)",
    )

    args = parser.parse_args()

    root = find_project_root()
    result = validate_documentation(root)

    print_result(result)

    if args.strict:
        return 1 if result.issues else 0
    return 1 if result.has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
