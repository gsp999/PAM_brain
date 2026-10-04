"""Command-line entry point."""
from __future__ import annotations

import argparse
from pathlib import Path
from .dataset import index_dataset
from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 2D photoacoustic mosaic")
    parser.add_argument("--data", type=Path, default=Path("PAM_Skull_FYH_20260929"))
    parser.add_argument("--logs", type=Path, default=None)
    parser.add_argument("--log", type=Path, action="append", default=[], help="individual scan log; repeatable")
    parser.add_argument("--output", type=Path, default=Path("outputs/lm_full"))
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--even-direction", choices=["auto", "as-is", "flip"], default="auto")
    parser.add_argument("--temporal-window", type=int, default=5)
    parser.add_argument("--appearance-neighbors", type=int, default=6)
    args = parser.parse_args()
    records = [r for r in index_dataset(args.data, args.logs, args.log)
               if r.index >= args.start and (args.stop is None or r.index <= args.stop)]
    reverse = {"auto": None, "as-is": False, "flip": True}[args.even_direction]
    report = run(records, args.output, reverse_even=reverse,
                 use_backward=not args.forward_only,
                 temporal_window=args.temporal_window,
                 appearance_neighbors=args.appearance_neighbors)
    print(f"Built {len(report['components'])} connected mosaic(s) from {len(records)} files")


if __name__ == "__main__":
    main()
