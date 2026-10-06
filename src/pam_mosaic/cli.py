"""Command-line entry point."""
from __future__ import annotations

import argparse
from pathlib import Path
from .dataset import DEFAULT_FILENAME_REGEX, index_dataset
from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 2D photoacoustic mosaic")
    parser.add_argument("--data", type=Path, required=True,
                        help="directory containing acquisitions in the supported .dat format")
    parser.add_argument("--filename-regex", default=DEFAULT_FILENAME_REGEX,
                        help="full filename regex with named index and optional scan groups")
    parser.add_argument("--logs", type=Path, default=None)
    parser.add_argument("--log", type=Path, action="append", default=[], help="individual scan log; repeatable")
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory for reconstructed components and cache")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--frontend-recovery", action=argparse.BooleanOptionalAction, default=True,
                        help="conservatively retry disconnected adjacent pairs (default: enabled)")
    parser.add_argument("--suppress-speckles", action=argparse.BooleanOptionalAction, default=True,
                        help="remove compact bright outliers before all registration (default: enabled)")
    parser.add_argument("--temporal-window", type=int, default=5)
    parser.add_argument("--appearance-neighbors", type=int, default=6)
    args = parser.parse_args()
    try:
        indexed = index_dataset(args.data, args.logs, args.log, args.filename_regex)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    records = [r for r in indexed
               if r.index >= args.start and (args.stop is None or r.index <= args.stop)]
    if not records:
        parser.error("No matching .dat files in the selected data directory/range")
    if (args.output / "report.json").exists():
        parser.error(f"Output already contains a completed run: {args.output}")
    report = run(records, args.output.resolve(),
                 temporal_window=args.temporal_window,
                 appearance_neighbors=args.appearance_neighbors,
                 frontend_recovery=args.frontend_recovery,
                 suppress_speckles=args.suppress_speckles)
    print(f"Built {len(report['components'])} connected mosaic(s) from {len(records)} files")


if __name__ == "__main__":
    main()
