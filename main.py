"""
MSIS Entry Point – Project Maven-inspired Multi-Source Intelligent System

Usage:
    python main.py [--metadata PATH] [--report-output PATH] [--generate-samples]

Examples:
    # Run with default sample data
    python main.py

    # Specify custom metadata index
    python main.py --metadata data/images/custom_metadata.json

    # Save report to file
    python main.py --report-output data/reports/report_$(date +%Y%m%d).txt

    # Generate synthetic sample images for testing
    python main.py --generate-samples --metadata data/images/metadata.json
"""

import argparse
import logging
import sys
from pathlib import Path

from pipeline import MavenPipeline
from scripts.generate_sample_data import generate_sample_data

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MSIS: Multi-Source Intelligent System (Project Maven pipeline)"
    )
    parser.add_argument(
        "--metadata",
        default="data/images/metadata.json",
        help="Path to the image metadata index JSON file",
    )
    parser.add_argument(
        "--report-output",
        default=None,
        help="Optional file path to save the generated military report",
    )
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate synthetic sample satellite/drone images for testing",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["huggingface", "ollama"],
        default=None,
        help="Override LLM backend (huggingface or ollama)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Override LLM backend if specified
    if args.llm_backend:
        import os
        os.environ["LLM_BACKEND"] = args.llm_backend

    # Generate sample data if requested
    if args.generate_samples:
        logger.info("Generating synthetic sample data...")
        generate_sample_data(args.metadata)

    # Ensure metadata file exists (create empty if absent so pipeline can auto-generate)
    meta_path = Path(args.metadata)
    if not meta_path.exists():
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text("[]", encoding="utf-8")
        logger.info(
            f"[Main] Created empty metadata file at {meta_path}. "
            "Pipeline will auto-generate synthetic sample data."
        )

    # Run pipeline
    pipeline = MavenPipeline()
    report = pipeline.run(
        metadata_json=args.metadata,
        report_output_path=args.report_output,
    )

    # Print report to stdout
    print("\n" + "=" * 72)
    print(report)
    print("=" * 72 + "\n")

    if args.report_output:
        print(f"Report saved to: {args.report_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
