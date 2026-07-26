#!/usr/bin/env python3
"""CLI entry point for the Patient Readmission Analytics pipeline.

Usage:
    python run.py full
    python run.py generate
    python run.py analyze
    python run.py quality
"""

import argparse
import logging
import sys

from readmission_analytics.src.pipeline import ReadmissionPipeline

MODES = ("full", "generate", "analyze", "quality")


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger for CLI output.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Patient Readmission Analytics Pipeline",
    )
    parser.add_argument(
        "mode",
        choices=MODES,
        help=(
            "Pipeline mode: "
            "'full' runs all steps, "
            "'generate' creates tables and synthetic data, "
            "'analyze' runs analytics on existing data, "
            "'quality' runs data quality checks only."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entry point.

    Args:
        argv: Optional argument list for testing.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    args = parse_args(argv)
    setup_logging(args.log_level)

    try:
        pipeline = ReadmissionPipeline(config_path=args.config)
        results = pipeline.run(mode=args.mode)

        for name, data in results.items():
            if hasattr(data, "shape"):
                print(f"\n{name}: {data.shape[0]} rows, {data.shape[1]} columns")
            elif isinstance(data, dict):
                if data:
                    print(f"\n{name}: {len(data)} check(s) with issues")
                else:
                    print(f"\n{name}: all checks passed")

        return 0
    except FileNotFoundError as exc:
        logging.error("File not found: %s", exc)
        return 1
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
