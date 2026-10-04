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
    parser.add_argument("--output", type=Path, default=Path("outputs/moderate_denoise_full"))
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--forward-only", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validated-roundtrip", dest="validated_roundtrip", action="store_true",
                      help="one pose per file with validated within-file forward/backward fusion (default)")
    mode.add_argument("--baseline", dest="validated_roundtrip", action="store_false",
                      help="run the original direct forward/backward averaging baseline")
    parser.set_defaults(validated_roundtrip=True)
    parser.add_argument("--frontend-recovery", action=argparse.BooleanOptionalAction, default=True,
                        help="conservatively retry disconnected adjacent pairs (default: enabled)")
    parser.add_argument("--suppress-speckles", action=argparse.BooleanOptionalAction, default=True,
                        help="remove compact bright outliers before all registration (default: enabled)")
    parser.add_argument("--even-direction", choices=["auto", "as-is", "flip"], default="auto")
    parser.add_argument("--temporal-window", type=int, default=5)
    parser.add_argument("--appearance-neighbors", type=int, default=6)
    args = parser.parse_args()
    if not args.validated_roundtrip:
        # Current-only processing defaults must not leak into an explicitly
        # requested baseline run.
        args.frontend_recovery = False
        args.suppress_speckles = False
    records = [r for r in index_dataset(args.data, args.logs, args.log)
               if r.index >= args.start and (args.stop is None or r.index <= args.stop)]
    reverse = {"auto": None, "as-is": False, "flip": True}[args.even_direction]
    if args.validated_roundtrip and (args.forward_only or args.even_direction != "auto"):
        parser.error("--validated-roundtrip requires automatic direction selection and both scans")
    if args.frontend_recovery and not args.validated_roundtrip:
        parser.error("--frontend-recovery requires --validated-roundtrip")
    if args.suppress_speckles and not args.validated_roundtrip:
        parser.error("--suppress-speckles requires --validated-roundtrip")
    report = run(records, args.output, reverse_even=reverse,
                 use_backward=not args.forward_only,
                 temporal_window=args.temporal_window,
                 appearance_neighbors=args.appearance_neighbors,
                 validated_roundtrip=args.validated_roundtrip,
                 frontend_recovery=args.frontend_recovery,
                 suppress_speckles=args.suppress_speckles)
    print(f"Built {len(report['components'])} connected mosaic(s) from {len(records)} files")


if __name__ == "__main__":
    main()
