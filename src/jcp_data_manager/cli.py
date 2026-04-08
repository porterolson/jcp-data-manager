"""Command line entry point for jcp-data-manager."""

from __future__ import annotations

import argparse

from .io import write_dataframe
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge session data with LinkedIn member data and optional enrichments."
    )
    parser.add_argument("--sessions", required=True, help="Path to the sessions JSON file.")
    parser.add_argument("--linkedin", required=True, help="Path to the LinkedIn members JSON file.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output path. Supported extensions: .parquet, .csv, .json",
    )
    parser.add_argument(
        "--skip-image-analysis",
        action="store_true",
        help="Skip DeepFace analysis using the picture URL column.",
    )
    parser.add_argument(
        "--skip-name-analysis",
        action="store_true",
        help="Skip gender-guesser and ethnicolr enrichment using given/family names.",
    )
    parser.add_argument(
        "--image-timeout",
        type=int,
        default=20,
        help="Timeout in seconds for downloading profile pictures.",
    )
    parser.add_argument(
        "--skip-face-detection",
        action="store_true",
        help="Pass enforce_detection=False to DeepFace.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    merged = run_pipeline(
        sessions_path=args.sessions,
        linkedin_path=args.linkedin,
        with_image_analysis=not args.skip_image_analysis,
        with_name_analysis=not args.skip_name_analysis,
        image_timeout=args.image_timeout,
        enforce_detection=not args.skip_face_detection,
    )
    write_dataframe(merged, args.output)


if __name__ == "__main__":
    main()
