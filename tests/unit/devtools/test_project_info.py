"""Unit tests for devtools.project_info.

Why:
    Project analysis utilities help track implementation status.
    These tests verify metrics are calculated correctly and edge cases
    are handled gracefully.
"""

from __future__ import annotations

from pathlib import Path

from devtools.project_info import ModuleInfo, ProjectAnalyzer, ProjectStatus


class TestProjectAnalyzer:
    """Tests for ProjectAnalyzer class."""

    def test_count_code_lines_excludes_blanks(self) -> None:
        """Test that blank lines are not counted.

        Why: Code metrics should reflect actual code, not whitespace.
        """
        analyzer = ProjectAnalyzer()
        content = """
def foo():
    pass

def bar():
    return 42
"""
        count = analyzer._count_code_lines(content)
        # Should count: def foo():, pass, def bar():, return 42 = 4 lines
        assert count == 4, f"Expected 4 code lines, got {count}"

    def test_count_code_lines_excludes_comments(self) -> None:
        """Test that comment-only lines are not counted.

        Why: Comments don't represent executable code.
        """
        analyzer = ProjectAnalyzer()
        content = """# This is a comment
def foo():
    # Another comment
    pass  # inline comment counts
"""
        count = analyzer._count_code_lines(content)
        # Should count: def foo():, pass = 2 lines (comments excluded)
        assert count == 2, f"Expected 2 code lines, got {count}"

    def test_is_stub_module_detects_not_implemented(self) -> None:
        """Test that NotImplementedError indicates a stub.

        Why: raise NotImplementedError is the canonical stub marker.
        """
        analyzer = ProjectAnalyzer()

        stub_content = """
def my_function():
    raise NotImplementedError("Coming soon")
"""
        assert analyzer._is_stub_module(stub_content) is True

        real_content = """
def my_function():
    return 42
"""
        assert analyzer._is_stub_module(real_content) is False

    def test_extract_definitions_finds_classes(self) -> None:
        """Test that class definitions are extracted.

        Why: Project status should track what classes are defined.
        """
        analyzer = ProjectAnalyzer()
        content = '''
"""Module docstring."""

class Foo:
    pass

class Bar:
    def method(self):
        pass
'''
        classes, functions, docstring = analyzer._extract_definitions(content)
        assert "Foo" in classes
        assert "Bar" in classes
        assert docstring == "Module docstring."

    def test_extract_definitions_finds_functions(self) -> None:
        """Test that top-level functions are extracted (not methods).

        Why: Functions inside classes are methods, not module-level functions.
        """
        analyzer = ProjectAnalyzer()
        content = """
def top_level():
    pass

class MyClass:
    def method(self):
        pass
"""
        classes, functions, docstring = analyzer._extract_definitions(content)
        assert "top_level" in functions
        # method should NOT be in functions (it's inside a class)
        assert "method" not in functions

    def test_extract_definitions_handles_syntax_error(self) -> None:
        """Test that syntax errors return empty results.

        Why: Graceful degradation for broken files.
        """
        analyzer = ProjectAnalyzer()
        content = "def broken(:\n    pass"  # Invalid syntax

        classes, functions, docstring = analyzer._extract_definitions(content)
        assert classes == []
        assert functions == []
        assert docstring is None


class TestProjectStatus:
    """Tests for ProjectStatus dataclass."""

    def test_total_lines_sums_modules(self) -> None:
        """Test that total_lines correctly sums all modules.

        Why: Aggregate metrics should be accurate.
        """
        status = ProjectStatus(
            modules=[
                ModuleInfo(path=Path("a.py"), name="a", lines=100),
                ModuleInfo(path=Path("b.py"), name="b", lines=50),
                ModuleInfo(path=Path("c.py"), name="c", lines=25),
            ]
        )
        assert status.total_lines == 175

    def test_implemented_vs_stub_separation(self) -> None:
        """Test that modules are correctly categorized.

        Why: Status reporting depends on accurate categorization.
        """
        status = ProjectStatus(
            modules=[
                ModuleInfo(path=Path("a.py"), name="a", lines=100, is_stub=False),
                ModuleInfo(path=Path("b.py"), name="b", lines=50, is_stub=True),
                ModuleInfo(path=Path("c.py"), name="c", lines=25, is_stub=False),
            ]
        )
        assert len(status.implemented_modules) == 2
        assert len(status.stub_modules) == 1
        assert status.stub_modules[0].name == "b"

    def test_summary_includes_key_info(self) -> None:
        """Test that summary contains essential information.

        Why: Summary is the primary status output for developers.
        """
        status = ProjectStatus(
            modules=[
                ModuleInfo(
                    path=Path("model.py"),
                    name="model",
                    lines=500,
                    classes=["TritterModel"],
                    is_stub=False,
                ),
            ]
        )
        summary = status.summary()
        assert "Total modules: 1" in summary
        assert "TritterModel" in summary


class TestModuleInfo:
    """Tests for ModuleInfo dataclass."""

    def test_default_values(self) -> None:
        """Test that defaults are sensible.

        Why: Ensure dataclass defaults don't cause surprises.
        """
        info = ModuleInfo(path=Path("test.py"), name="test", lines=10)
        assert info.classes == []
        assert info.functions == []
        assert info.is_stub is False
        assert info.docstring is None
