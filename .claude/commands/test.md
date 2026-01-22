# Run Tests

Run the test suite for Tritter. Pass arguments to run specific tests.

## Usage

- `/test` - Run all tests
- `/test test_config` - Run tests matching pattern
- `/test tests/unit/test_model.py` - Run specific test file

## Instructions

Run pytest with the provided arguments (or all tests if none provided). Report any failures with file locations and error messages. If tests pass, summarize what was tested.

```bash
pytest $ARGUMENTS
```

If tests fail, analyze the failures and suggest fixes based on the DEVELOPMENT_STANDARDS.md requirements (use config values, proper assertions, etc.).
