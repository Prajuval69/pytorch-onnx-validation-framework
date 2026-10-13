"""Portable argparse command-line entry point."""

import argparse
import json
import sys
from pathlib import Path

from amlv.errors import ConfigError, ValidationError


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="amlv", description="Synthetic CPU model conformance")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor", help="Inspect installed dependencies and CPU providers"
    ).add_argument(
        "--out", default="artifacts", help="Artifact root for writable probe (default: artifacts)"
    )
    for name in ("demo", "validate"):
        command = commands.add_parser(name, help="Execute synthetic cross-runtime validation")
        command.add_argument("--out", required=True)
        command.add_argument("--inject-mismatch", action="store_true")
        if name == "validate":
            command.add_argument("--config", required=True)
    commands.add_parser("report", help="Regenerate reports from immutable JSON").add_argument(
        "--run", required=True
    )
    command = commands.add_parser("reproduce", help="Replay exact saved inputs and model")
    command.add_argument("--case", required=True)
    command.add_argument("--out", required=True)
    command.add_argument("--without-fault", action="store_true")
    command.add_argument("--atol", type=float)
    command.add_argument("--rtol", type=float)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            from amlv.doctor import doctor

            result = doctor(Path(args.out))
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result["healthy"] else 3
        if args.command == "report":
            from amlv.reporting import build_report

            manifest = build_report(Path(args.run))
        elif args.command == "reproduce":
            from amlv.pipeline import reproduce

            manifest = reproduce(
                Path(args.case), Path(args.out), args.without_fault, args.atol, args.rtol
            )
        else:
            from amlv.config import load_config, validate_config
            from amlv.pipeline import run_suite

            config = (
                load_config(Path(args.config))
                if args.command == "validate"
                else validate_config({})
            )
            manifest = run_suite(config, Path(args.out), args.inject_mismatch)
        print(
            json.dumps(
                {
                    "run_id": manifest["run_id"],
                    "summary": manifest["summary"],
                    "synthetic_data": True,
                    "fault_injection": manifest["fault_injection"],
                    "report": str(
                        (
                            Path(args.run if args.command == "report" else args.out)
                            / manifest["html_report"]
                        ).resolve()
                    )
                    if manifest["config"]["report"]["html"]
                    else None,
                    "exit_code": manifest["exit_code"],
                },
                indent=2,
            )
        )
        return 0 if args.command == "report" else manifest["exit_code"]
    except ConfigError as exc:
        print(f"CONFIG: {exc}", file=sys.stderr)
        return 2
    except (ValidationError, OSError, ImportError) as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
