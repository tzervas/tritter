"""CLI entry point for devtools package.

Allows running devtools as a module:
    python -m devtools --help
    python -m devtools validate
    python -m devtools status
"""

from __future__ import annotations

import sys


def main() -> int:
    """Main entry point for devtools CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="devtools",
        description="Tritter development utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
    validate    Run validation suite (format, lint, typecheck, tests)
    status      Show project implementation status

Examples:
    python -m devtools validate              # Full validation
    python -m devtools validate --quick      # Skip tests
    python -m devtools status                # Project status
    python -m devtools status --json         # JSON output
        """,
    )
    parser.add_argument(
        "command",
        choices=["validate", "status"],
        help="Command to run",
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="Additional arguments for command",
    )

    # Parse known args to handle subcommand args
    args, remaining = parser.parse_known_args()

    if args.command == "validate":
        from devtools.validate import main as validate_main

        # Reconstruct sys.argv for subcommand
        sys.argv = ["devtools.validate", *remaining, *args.args]
        return validate_main()

    elif args.command == "status":
        from devtools.project_info import main as status_main

        sys.argv = ["devtools.project_info", *remaining, *args.args]
        return status_main()

    return 1


if __name__ == "__main__":
    sys.exit(main())
