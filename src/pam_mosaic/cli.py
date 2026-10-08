"""Command-line entry point."""
from __future__ import annotations

import argparse
from pathlib import Path
from .preprocessing.dataset import index_dataset
from .preprocessing.inputs import available_formats, get_format
from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 2D photoacoustic mosaic")
    parser.add_argument("--data", type=Path, required=True,
                        help="directory containing acquisitions for the selected input adapter")
    parser.add_argument("--input-format", choices=tuple(available_formats()),
                        default="bidirectional500", help="acquisition adapter from preprocessing/formats")
    parser.add_argument("--filename-regex", default=None,
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
    parser.add_argument("--background-correction", action=argparse.BooleanOptionalAction,
                        default=True, help="subtract estimated per-frame background for all input formats (default: enabled)")
    parser.add_argument("--pam512-speckle-excess", type=float, default=.02,
                        help="PAM512 bright-point excess threshold (default: 0.02; larger detects fewer candidates)")
    parser.add_argument("--appearance-neighbors", type=int, default=6)
    args = parser.parse_args()
    try:
        adapter = get_format(args.input_format)
        filename_regex = args.filename_regex or adapter.FILENAME_REGEX
        indexed = index_dataset(args.data, args.logs, args.log, filename_regex,
                                file_glob=getattr(adapter, 'FILE_GLOB', '*.dat'))
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    records = [r for r in indexed
               if r.index >= args.start and (args.stop is None or r.index <= args.stop)]
    if not records:
        parser.error("No matching acquisition files in the selected data directory/range")
    if (args.output / "report.json").exists():
        parser.error(f"Output already contains a completed run: {args.output}")
    report = run(records, args.output.resolve(),
                 temporal_window=args.temporal_window,
                 appearance_neighbors=args.appearance_neighbors,
                 frontend_recovery=args.frontend_recovery,
                 suppress_speckles=args.suppress_speckles,
                 input_format=args.input_format,
                 background_correction=args.background_correction,
                 pam512_speckle_excess=args.pam512_speckle_excess)
    print(f"Built {len(report['components'])} connected mosaic(s) from {len(records)} files")


if __name__ == "__main__":
    main()
