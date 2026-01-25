"""Unit tests for devtools.validate.

Why:
    Validation runner is critical for development workflow. These tests
    verify result aggregation logic without actually running external commands.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from devtools.validate import StepResult, ValidationResult, ValidationRunner, ValidationStep


class TestValidationResult:
    """Tests for ValidationResult aggregation."""

    def test_all_passed_when_empty(self) -> None:
        """Test that empty results are considered passed.

        Why: No failures = success by default.
        """
        result = ValidationResult()
        assert result.all_passed is True

    def test_all_passed_when_all_succeed(self) -> None:
        """Test that all_passed is True when every step succeeds.

        Why: Validates the happy path.
        """
        result = ValidationResult(
            steps=[
                StepResult(step=ValidationStep.FORMAT, success=True, output="", return_code=0),
                StepResult(step=ValidationStep.LINT, success=True, output="", return_code=0),
            ]
        )
        assert result.all_passed is True

    def test_all_passed_false_on_any_failure(self) -> None:
        """Test that any failure makes all_passed False.

        Why: Validation should fail fast on any error.
        """
        result = ValidationResult(
            steps=[
                StepResult(step=ValidationStep.FORMAT, success=True, output="", return_code=0),
                StepResult(step=ValidationStep.LINT, success=False, output="error", return_code=1),
            ]
        )
        assert result.all_passed is False

    def test_summary_contains_pass_fail_status(self) -> None:
        """Test that summary clearly shows pass/fail for each step.

        Why: Developers need to quickly see what failed.
        """
        result = ValidationResult(
            steps=[
                StepResult(step=ValidationStep.FORMAT, success=True, output="", return_code=0),
                StepResult(step=ValidationStep.LINT, success=False, output="", return_code=1),
            ]
        )
        summary = result.summary()
        assert "[PASS] format" in summary
        assert "[FAIL] lint" in summary
        assert "FAILED" in summary  # Overall status


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_step_result_creation(self) -> None:
        """Test that StepResult can be created with all fields.

        Why: Basic dataclass functionality check.
        """
        result = StepResult(
            step=ValidationStep.TESTS,
            success=True,
            output="3 passed",
            return_code=0,
        )
        assert result.step == ValidationStep.TESTS
        assert result.success is True
        assert "3 passed" in result.output
        assert result.return_code == 0


class TestValidationRunner:
    """Tests for ValidationRunner class."""

    def test_detect_project_root_finds_pyproject(self) -> None:
        """Test that project root detection works.

        Why: All commands run relative to project root.
        """
        # This test runs in the actual project, so it should find the root
        root = ValidationRunner._detect_project_root()
        assert (root / "pyproject.toml").exists()

    @patch("subprocess.run")
    def test_run_command_handles_timeout(self, mock_run: MagicMock) -> None:
        """Test that command timeouts are handled gracefully.

        Why: Long-running commands shouldn't hang forever.
        """
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["test"], timeout=300)

        runner = ValidationRunner(verbose=False)
        result = runner._run_command(["test"], ValidationStep.TESTS)

        assert result.success is False
        assert "timed out" in result.output.lower()
        assert result.return_code == -1

    @patch("subprocess.run")
    def test_run_command_handles_exception(self, mock_run: MagicMock) -> None:
        """Test that unexpected exceptions are caught.

        Why: Validation shouldn't crash on unexpected errors.
        """
        mock_run.side_effect = OSError("Command not found")

        runner = ValidationRunner(verbose=False)
        result = runner._run_command(["nonexistent"], ValidationStep.TESTS)

        assert result.success is False
        assert "error" in result.output.lower()

    @patch("subprocess.run")
    def test_run_command_captures_output(self, mock_run: MagicMock) -> None:
        """Test that command output is captured.

        Why: Output is needed for debugging failures.
        """
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Success!"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        runner = ValidationRunner(verbose=False)
        result = runner._run_command(["test"], ValidationStep.FORMAT)

        assert result.success is True
        assert "Success!" in result.output

    def test_run_all_filters_steps(self) -> None:
        """Test that run_all respects step filtering.

        Why: Quick mode should skip tests.
        """
        runner = ValidationRunner(verbose=False)

        # Mock all run methods to avoid actual execution
        with (
            patch.object(runner, "run_format_check") as mock_format,
            patch.object(runner, "run_lint") as mock_lint,
            patch.object(runner, "run_typecheck") as mock_typecheck,
            patch.object(runner, "run_import_check") as mock_imports,
            patch.object(runner, "run_tests") as mock_tests,
        ):
            # Set up mock returns
            for mock in [mock_format, mock_lint, mock_typecheck, mock_imports, mock_tests]:
                mock.return_value = StepResult(
                    step=ValidationStep.FORMAT, success=True, output="", return_code=0
                )

            # Run only format and lint
            steps = [ValidationStep.FORMAT, ValidationStep.LINT]
            runner.run_all(steps=steps)

            # Verify only requested steps were run
            mock_format.assert_called_once()
            mock_lint.assert_called_once()
            mock_typecheck.assert_not_called()
            mock_tests.assert_not_called()


class TestValidationStep:
    """Tests for ValidationStep enum."""

    def test_all_steps_have_values(self) -> None:
        """Test that all steps have string values.

        Why: Values are used in output and must be meaningful.
        """
        for step in ValidationStep:
            assert isinstance(step.value, str)
            assert len(step.value) > 0

    def test_step_values_are_unique(self) -> None:
        """Test that step values don't collide.

        Why: Values are used as identifiers in output.
        """
        values = [step.value for step in ValidationStep]
        assert len(values) == len(set(values))
