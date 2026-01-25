"""Project analysis utilities for Tritter development.

Provides tools to analyze project structure, implementation status,
and codebase metrics. Useful for development status tracking and
documentation generation.

Why:
    Understanding project state (what's implemented vs. stubbed, code metrics,
    dependency status) helps prioritize development work. This module centralizes
    that analysis in one place.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class ModuleInfo:
    """Information about a Python module.

    Attributes:
        path: Path to the module file.
        name: Module name (dotted notation).
        lines: Total lines of code (excluding blank/comments).
        classes: List of class names defined.
        functions: List of function names defined.
        is_stub: Whether this appears to be a stub module.
        docstring: Module-level docstring if present.
    """

    path: Path
    name: str
    lines: int
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    is_stub: bool = False
    docstring: str | None = None


@dataclass
class ProjectStatus:
    """Overall project implementation status.

    Attributes:
        modules: Information about each module.
        total_lines: Total lines of Python code.
        implemented_modules: Modules with real implementation.
        stub_modules: Modules that are stubs/placeholders.
    """

    modules: list[ModuleInfo] = field(default_factory=list)

    @property
    def total_lines(self) -> int:
        """Total lines of code across all modules."""
        return sum(m.lines for m in self.modules)

    @property
    def implemented_modules(self) -> list[ModuleInfo]:
        """Modules with real implementation (not stubs)."""
        return [m for m in self.modules if not m.is_stub]

    @property
    def stub_modules(self) -> list[ModuleInfo]:
        """Modules that are stubs/placeholders."""
        return [m for m in self.modules if m.is_stub]

    def summary(self) -> str:
        """Generate human-readable project status summary."""
        lines = [
            "Tritter Project Status",
            "=" * 50,
            "",
            f"Total modules: {len(self.modules)}",
            f"Implemented: {len(self.implemented_modules)}",
            f"Stubs: {len(self.stub_modules)}",
            f"Total lines: {self.total_lines:,}",
            "",
            "Implemented Modules:",
            "-" * 30,
        ]

        for mod in self.implemented_modules:
            lines.append(f"  {mod.name}: {mod.lines} lines")
            if mod.classes:
                lines.append(f"    Classes: {', '.join(mod.classes)}")

        if self.stub_modules:
            lines.append("")
            lines.append("Stub Modules (TODO):")
            lines.append("-" * 30)
            for mod in self.stub_modules:
                lines.append(f"  {mod.name}")

        return "\n".join(lines)


class ProjectAnalyzer:
    """Analyzes Tritter project structure and implementation status.

    Why:
        Provides programmatic access to project metrics and status,
        enabling automated status reporting, documentation generation,
        and development planning.

    Attributes:
        project_root: Path to project root directory.
        src_root: Path to src/tritter source directory.
    """

    # Patterns that indicate a module is a stub
    STUB_PATTERNS = [
        r"raise NotImplementedError",
        r"TODO.*implement",
        r"pass\s*#.*stub",
        r"\.\.\..*#.*placeholder",
    ]

    def __init__(self, project_root: Path | None = None) -> None:
        """Initialize project analyzer.

        Args:
            project_root: Path to project root. Defaults to auto-detection.
        """
        self.project_root = project_root or self._detect_project_root()
        self.src_root = self.project_root / "src" / "tritter"

    @staticmethod
    def _detect_project_root() -> Path:
        """Find project root by looking for pyproject.toml."""
        current = Path.cwd()
        for parent in [current, *current.parents]:
            if (parent / "pyproject.toml").exists():
                return parent
        return current

    def _count_code_lines(self, content: str) -> int:
        """Count non-blank, non-comment lines of code.

        Args:
            content: File content to analyze.

        Returns:
            Number of meaningful lines of code.
        """
        count = 0
        in_multiline_string = False

        for line in content.split("\n"):
            stripped = line.strip()

            # Skip blank lines
            if not stripped:
                continue

            # Track multiline strings (rough heuristic)
            if '"""' in stripped or "'''" in stripped:
                quote_count = stripped.count('"""') + stripped.count("'''")
                if quote_count == 1:
                    in_multiline_string = not in_multiline_string
                count += 1
                continue

            if in_multiline_string:
                count += 1
                continue

            # Skip single-line comments
            if stripped.startswith("#"):
                continue

            count += 1

        return count

    def _is_stub_module(self, content: str) -> bool:
        """Check if module content indicates it's a stub.

        Args:
            content: File content to analyze.

        Returns:
            True if module appears to be a stub/placeholder.
        """
        for pattern in self.STUB_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False

    def _extract_definitions(self, content: str) -> tuple[list[str], list[str], str | None]:
        """Extract class and function definitions from Python code.

        Args:
            content: Python source code.

        Returns:
            Tuple of (class_names, function_names, module_docstring).
        """
        classes: list[str] = []
        functions: list[str] = []
        docstring: str | None = None

        try:
            tree = ast.parse(content)

            # Get module docstring
            if (
                tree.body
                and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)
            ):
                docstring = tree.body[0].value.value

            # Iterate only over top-level nodes (O(n) instead of O(n²))
            # This avoids nested ast.walk() which would re-traverse the tree
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    # Top-level functions only (methods are inside ClassDef.body)
                    functions.append(node.name)
        except SyntaxError:
            # Intentionally silent: gracefully handle modules with syntax errors
            # during project analysis. These are reported elsewhere (e.g., linting).
            # If the module contains syntax errors (e.g., is partially written),
            # treat it as having no discoverable structure rather than failing analysis.
            return classes, functions, docstring

        return classes, functions, docstring

    def analyze_module(self, path: Path) -> ModuleInfo:
        """Analyze a single Python module.

        Args:
            path: Path to the Python file.

        Returns:
            ModuleInfo with analysis results.
        """
        content = path.read_text()

        # Calculate module name relative to src
        try:
            relative = path.relative_to(self.src_root)
            parts = list(relative.parts)
            if parts[-1] == "__init__.py":
                parts = parts[:-1]
            else:
                parts[-1] = parts[-1].replace(".py", "")
            name = "tritter." + ".".join(parts) if parts else "tritter"
        except ValueError:
            name = path.stem

        classes, functions, docstring = self._extract_definitions(content)

        return ModuleInfo(
            path=path,
            name=name,
            lines=self._count_code_lines(content),
            classes=classes,
            functions=functions,
            is_stub=self._is_stub_module(content),
            docstring=docstring,
        )

    def analyze_project(self) -> ProjectStatus:
        """Analyze entire project and return status.

        Returns:
            ProjectStatus with all module information.
        """
        status = ProjectStatus()

        if not self.src_root.exists():
            return status

        for py_file in self.src_root.rglob("*.py"):
            module_info = self.analyze_module(py_file)
            status.modules.append(module_info)

        # Sort by module name for consistent output
        status.modules.sort(key=lambda m: m.name)

        return status

    def get_implementation_roadmap(self) -> dict[str, list[str]]:
        """Get roadmap of what's implemented vs. pending.

        Returns:
            Dict with 'implemented' and 'pending' lists.
        """
        status = self.analyze_project()

        return {
            "implemented": [m.name for m in status.implemented_modules],
            "pending": [m.name for m in status.stub_modules],
        }


def get_project_status(verbose: bool = True) -> ProjectStatus:
    """Convenience function to get project status.

    Args:
        verbose: If True, print summary to stdout.

    Returns:
        ProjectStatus with analysis results.
    """
    analyzer = ProjectAnalyzer()
    status = analyzer.analyze_project()

    if verbose:
        print(status.summary())

    return status


def main() -> int:
    """CLI entry point for project analysis."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Analyze Tritter project status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--roadmap",
        action="store_true",
        help="Show implementation roadmap",
    )

    args = parser.parse_args()

    analyzer = ProjectAnalyzer()

    if args.roadmap:
        roadmap = analyzer.get_implementation_roadmap()
        if args.json:
            print(json.dumps(roadmap, indent=2))
        else:
            print("Implementation Roadmap")
            print("=" * 40)
            print("\nImplemented:")
            for mod in roadmap["implemented"]:
                print(f"  [x] {mod}")
            print("\nPending:")
            for mod in roadmap["pending"]:
                print(f"  [ ] {mod}")
        return 0

    status = analyzer.analyze_project()

    if args.json:
        data = {
            "total_modules": len(status.modules),
            "total_lines": status.total_lines,
            "implemented_count": len(status.implemented_modules),
            "stub_count": len(status.stub_modules),
            "modules": [
                {
                    "name": m.name,
                    "lines": m.lines,
                    "is_stub": m.is_stub,
                    "classes": m.classes,
                }
                for m in status.modules
            ],
        }
        print(json.dumps(data, indent=2))
    else:
        print(status.summary())

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
