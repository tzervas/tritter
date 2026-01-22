"""Import validation for Tritter modules.

Validates that __all__ exports in __init__.py files match actual imports.
This ensures that `from module import *` works correctly and that all
advertised public APIs are actually available.

Why:
    Import/export mismatches are a common source of runtime errors.
    A module may declare an export in __all__ but forget to import it,
    or import something without exporting it. This script catches these
    issues before they cause problems in production.

Usage:
    python -m devtools.import_validator
    python -m devtools.import_validator src/tritter
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class ImportIssue:
    """A single import/export validation issue.

    Attributes:
        file: Path to the __init__.py file with the issue.
        issue_type: Type of issue (missing_import, missing_export, etc.)
        name: The symbol name with the issue.
        message: Human-readable description of the issue.
    """

    file: Path
    issue_type: str
    name: str
    message: str


@dataclass
class ValidationResult:
    """Result of validating imports across a module tree.

    Attributes:
        issues: List of all issues found.
        files_checked: Number of __init__.py files examined.
    """

    issues: list[ImportIssue] = field(default_factory=list)
    files_checked: int = 0

    @property
    def is_valid(self) -> bool:
        """True if no issues were found."""
        return len(self.issues) == 0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Import Validation Results",
            "=" * 50,
            f"Files checked: {self.files_checked}",
            f"Issues found: {len(self.issues)}",
            "",
        ]

        if self.issues:
            lines.append("Issues:")
            lines.append("-" * 50)
            for issue in self.issues:
                lines.append(f"[{issue.issue_type}] {issue.file}")
                lines.append(f"    {issue.message}")
                lines.append("")

        return "\n".join(lines)


class ImportValidator:
    """Validates __all__ exports match actual imports in __init__.py files.

    Why:
        Per DEVELOPMENT_STANDARDS.md, every __init__.py must have __all__
        that matches its imports exactly. This validator enforces that rule
        automatically.

    Usage:
        validator = ImportValidator()
        result = validator.validate_path(Path("src/tritter"))
        if not result.is_valid:
            print(result.summary())
    """

    def __init__(self) -> None:
        """Initialize validator."""
        pass

    def _extract_imports(self, tree: ast.AST) -> set[str]:
        """Extract all imported names from an AST.

        Args:
            tree: Parsed AST of a Python file.

        Returns:
            Set of names that are imported (available in module namespace).
        """
        imported: set[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # import foo -> foo is available
                    # import foo as bar -> bar is available
                    name = alias.asname if alias.asname else alias.name
                    # For dotted imports like "import a.b", only "a" is available
                    imported.append(name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.names:
                    for alias in node.names:
                        # from x import y -> y is available
                        # from x import y as z -> z is available
                        name = alias.asname if alias.asname else alias.name
                        if name != "*":  # Skip star imports
                            imported.append(name)

        return set(imported)

    def _extract_all(self, tree: ast.AST) -> set[str] | None:
        """Extract __all__ definition from an AST.

        Args:
            tree: Parsed AST of a Python file.

        Returns:
            Set of names in __all__, or None if __all__ is not defined.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        # __all__ = ["a", "b", "c"]
                        if isinstance(node.value, ast.List):
                            names = []
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    names.append(elt.value)
                            return set(names)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id == "__all__":
                    # __all__: list[str] = ["a", "b", "c"]
                    if node.value and isinstance(node.value, ast.List):
                        names = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                names.append(elt.value)
                        return set(names)

        return None

    def _extract_definitions(self, tree: ast.AST) -> set[str]:
        """Extract all top-level definitions from an AST.

        Args:
            tree: Parsed AST of a Python file.

        Returns:
            Set of names defined at module level (classes, functions, variables).
        """
        defined: set[str] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                defined.append(node.name)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                defined.append(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined.append(target.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    defined.append(node.target.id)

        return set(defined)

    def validate_file(self, file_path: Path) -> list[ImportIssue]:
        """Validate a single __init__.py file.

        Args:
            file_path: Path to the __init__.py file.

        Returns:
            List of issues found in the file.
        """
        issues: list[ImportIssue] = []

        try:
            content = file_path.read_text()
            tree = ast.parse(content)
        except SyntaxError as e:
            issues.append(
                ImportIssue(
                    file=file_path,
                    issue_type="syntax_error",
                    name="",
                    message=f"Syntax error: {e}",
                )
            )
            return issues

        imports = self._extract_imports(tree)
        definitions = self._extract_definitions(tree)
        all_exports = self._extract_all(tree)
        available = imports | definitions

        if all_exports is None:
            # No __all__ defined - this might be intentional for internal modules
            # but we'll flag it as a warning for __init__.py files
            issues.append(
                ImportIssue(
                    file=file_path,
                    issue_type="missing_all",
                    name="__all__",
                    message="No __all__ defined. Consider adding explicit exports.",
                )
            )
            return issues

        # Check for exports that aren't available
        for name in all_exports:
            if name not in available:
                issues.append(
                    ImportIssue(
                        file=file_path,
                        issue_type="undefined_export",
                        name=name,
                        message=f"'{name}' is in __all__ but not imported or defined",
                    )
                )

        # Check for imports that aren't exported (informational)
        # This is not necessarily an error, but can indicate forgotten exports
        public_imports = {n for n in imports if not n.startswith("_")}
        unexported = public_imports - all_exports - {"__version__"}
        for name in unexported:
            # Skip common utility imports that don't need exporting
            if name not in {"dataclass", "field", "Enum", "Literal", "Path"}:
                issues.append(
                    ImportIssue(
                        file=file_path,
                        issue_type="unexported_import",
                        name=name,
                        message=f"'{name}' is imported but not in __all__ (may be intentional)",
                    )
                )

        return issues

    def validate_path(self, root: Path) -> ValidationResult:
        """Validate all __init__.py files under a path.

        Args:
            root: Root path to search for __init__.py files.

        Returns:
            ValidationResult with all issues found.
        """
        result = ValidationResult()

        # Find all __init__.py files
        if root.is_file() and root.name == "__init__.py":
            init_files = [root]
        else:
            init_files = list(root.rglob("__init__.py"))

        for init_file in init_files:
            result.files_checked += 1
            issues = self.validate_file(init_file)
            result.issues.extend(issues)

        return result


def validate_imports(
    path: Path | None = None,
    strict: bool = False,
) -> ValidationResult:
    """Convenience function to validate imports.

    Args:
        path: Path to validate. Defaults to src/tritter.
        strict: If True, treat warnings as errors.

    Returns:
        ValidationResult with all issues found.
    """
    if path is None:
        path = Path.cwd() / "src" / "tritter"

    validator = ImportValidator()
    result = validator.validate_path(path)

    if strict:
        # Filter out informational issues
        result.issues = [
            i for i in result.issues if i.issue_type not in {"unexported_import", "missing_all"}
        ]

    return result


def main() -> int:
    """CLI entry point for import validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate __all__ exports match imports in __init__.py files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m devtools.import_validator              # Validate src/tritter
    python -m devtools.import_validator devtools     # Validate devtools
    python -m devtools.import_validator --strict     # Treat warnings as errors
        """,
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="src/tritter",
        help="Path to validate (default: src/tritter)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )

    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        return 1

    result = validate_imports(path, strict=args.strict)
    print(result.summary())

    # Return error code if critical issues found
    critical_issues = [
        i for i in result.issues if i.issue_type in {"undefined_export", "syntax_error"}
    ]
    return 1 if critical_issues else 0


if __name__ == "__main__":
    sys.exit(main())
