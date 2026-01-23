"""Validation runner for Tritter development workflow.

Provides utilities to run the full validation suite including formatting,
linting, type checking, and testing. Designed to be used both programmatically
and via CLI.

Why:
    Consistent validation before commits catches issues early. This module
    centralizes the validation logic so it can be invoked from multiple contexts
    (CI, pre-commit hooks, IDE, CLI) with identical behavior.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class ValidationStep(Enum):
    """Individual validation steps in the suite."""

    FORMAT = "format"
    LINT = "lint"
    TYPECHECK = "typecheck"
    IMPORTS = "imports"
    TESTS = "tests"


@dataclass
class StepResult:
    """Result of a single validation step.

    Attributes:
        step: Which validation step was run.
        success: Whether the step passed.
        output: Captured stdout/stderr from the command.
        return_code: Process return code.
    """

    step: ValidationStep
    success: bool
    output: str
    return_code: int


@dataclass
class ValidationResult:
    """Aggregate result of running validation suite.

    Attributes:
        steps: Results for each validation step.
        all_passed: True if every step succeeded.
    """

    steps: list[StepResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """Check if all validation steps passed."""
        return all(step.success for step in self.steps)

    def summary(self) -> str:
        """Generate human-readable summary of validation results."""
        lines = ["Validation Results", "=" * 40]
        for step in self.steps:
            status = "[PASS]" if step.success else "[FAIL]"
            lines.append(f"{status} {step.step.value}")
        lines.append("=" * 40)
        lines.append(f"Overall: {'PASSED' if self.all_passed else 'FAILED'}")
        return "\n".join(lines)


class ValidationRunner:
    """Runs the Tritter validation suite.

    Why:
        Encapsulating validation logic in a class allows configuration
        (project root, verbosity, step selection) and makes testing easier.

    Attributes:
        project_root: Path to project root directory.
        verbose: Whether to print step output as it runs.
    """

    def __init__(
        self,
        project_root: Path | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize validation runner.

        Args:
            project_root: Path to project root. Defaults to detecting from cwd.
            verbose: Print output as validation runs.
        """
        self.project_root = project_root or self._detect_project_root()
        self.verbose = verbose

    @staticmethod
    def _detect_project_root() -> Path:
        """Find project root by looking for pyproject.toml."""
        current = Path.cwd()
        for parent in [current, *current.parents]:
            if (parent / "pyproject.toml").exists():
                return parent
        return current

    def _run_command(
        self,
        cmd: Sequence[str],
        step: ValidationStep,
    ) -> StepResult:
        """Run a shell command and capture result.

        Args:
            cmd: Command and arguments to run.
            step: Which validation step this is for.

        Returns:
            StepResult with captured output and status.
        """
        if self.verbose:
            print(f"\n>>> Running: {' '.join(cmd)}")

        try:
            # Security note: Using list form (not shell=True) prevents shell injection.
            # Each element is passed as a separate argv entry to the subprocess.
            # Commands are constructed from static strings; only test_args may vary,
            # but as a local dev tool this is acceptable risk.
            result = subprocess.run(
                list(cmd),  # Ensure it's a list, not potentially shell-parsed string
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            output = result.stdout + result.stderr
            success = result.returncode == 0

            if self.verbose:
                print(output)
                status = "PASSED" if success else "FAILED"
                print(f"<<< {step.value}: {status}")

            return StepResult(
                step=step,
                success=success,
                output=output,
                return_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return StepResult(
                step=step,
                success=False,
                output="Command timed out after 300 seconds",
                return_code=-1,
            )
        except Exception as e:
            return StepResult(
                step=step,
                success=False,
                output=f"Error running command: {e}",
                return_code=-1,
            )

    def run_format_check(self) -> StepResult:
        """Check code formatting with ruff."""
        return self._run_command(
            ["ruff", "format", "--check", "."],
            ValidationStep.FORMAT,
        )

    def run_lint(self) -> StepResult:
        """Run linting with ruff."""
        return self._run_command(
            ["ruff", "check", "."],
            ValidationStep.LINT,
        )

    def run_typecheck(self) -> StepResult:
        """Run type checking with mypy."""
        return self._run_command(
            ["mypy", "src/tritter"],
            ValidationStep.TYPECHECK,
        )

    def run_import_check(self) -> StepResult:
        """Verify imports work correctly."""
        return self._run_command(
            ["python", "-c", "from tritter import *; print('Imports OK')"],
            ValidationStep.IMPORTS,
        )

    def run_tests(self, test_args: Sequence[str] | None = None) -> StepResult:
        """Run pytest test suite.

        Args:
            test_args: Additional arguments to pass to pytest.
        """
        cmd = ["pytest"]
        if test_args:
            cmd.extend(test_args)
        return self._run_command(cmd, ValidationStep.TESTS)

    def run_all(
        self,
        steps: Sequence[ValidationStep] | None = None,
        test_args: Sequence[str] | None = None,
    ) -> ValidationResult:
        """Run all validation steps.

        Args:
            steps: Specific steps to run. If None, runs all steps.
            test_args: Additional arguments for pytest.

        Returns:
            ValidationResult with all step outcomes.
        """
        if steps is None:
            steps = list(ValidationStep)

        result = ValidationResult()

        step_runners = {
            ValidationStep.FORMAT: self.run_format_check,
            ValidationStep.LINT: self.run_lint,
            ValidationStep.TYPECHECK: self.run_typecheck,
            ValidationStep.IMPORTS: self.run_import_check,
            ValidationStep.TESTS: lambda: self.run_tests(test_args),
        }

        for step in steps:
            runner = step_runners.get(step)
            if runner:
                step_result = runner()
                result.steps.append(step_result)

        return result


def run_validation(
    verbose: bool = True,
    test_args: Sequence[str] | None = None,
) -> ValidationResult:
    """Convenience function to run full validation suite.

    Args:
        verbose: Print output as validation runs.
        test_args: Additional arguments for pytest.

    Returns:
        ValidationResult with all outcomes.
    """
    runner = ValidationRunner(verbose=verbose)
    return runner.run_all(test_args=test_args)


def main() -> int:
    """CLI entry point for validation runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Tritter validation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m devtools.validate              # Run all checks (verbose by default)
    python -m devtools.validate --quick      # Skip tests
    python -m devtools.validate -q           # Quiet output
        """,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip tests (format, lint, typecheck only)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Minimal output (suppress step-by-step details)",
    )
    parser.add_argument(
        "test_args",
        nargs="*",
        help="Additional arguments to pass to pytest",
    )

    args = parser.parse_args()

    verbose = not args.quiet

    if args.quick:
        steps = [
            ValidationStep.FORMAT,
            ValidationStep.LINT,
            ValidationStep.TYPECHECK,
            ValidationStep.IMPORTS,
        ]
    else:
        steps = None  # All steps

    runner = ValidationRunner(verbose=verbose)
    result = runner.run_all(steps=steps, test_args=args.test_args or None)

    print("\n" + result.summary())

    return 0 if result.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
