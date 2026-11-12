"""Explicit CLI opt-in for historical scripts, not scientific validation."""
import sys


def require_legacy_opt_in() -> None:
    """Warn, reject implicit execution, and consume the CLI-only opt-in flag."""
    print(
        "WARNING: historical/unaccepted legacy script; canonical pipeline is "
        "src/reproduce.py. --allow-legacy permits historical behavior, including "
        "possible overwrites; it does not validate scientific results.",
        file=sys.stderr,
    )
    args = sys.argv[1:]
    boundary = args.index("--") if "--" in args else len(args)
    options, literal = args[:boundary], args[boundary:]
    if "--allow-legacy" not in options:
        print("Refusing to run without explicit --allow-legacy.", file=sys.stderr)
        raise SystemExit(2)
    sys.argv[1:] = [arg for arg in options if arg != "--allow-legacy"] + literal
