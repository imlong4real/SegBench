#!/usr/bin/env python3
"""Cirro adapter for an explicit Bin2Cell StarDist tile size.

Bin2Cell 0.3.4 defaults to 4096-pixel tiles.  Its tiled StarDist call requires
each input dimension to be at least that large, which is not true for the
deterministic smoke ROI.  This adapter changes only that orchestration value;
the pinned Bin2Cell/StarDist implementation and all biological parameters are
unchanged.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("method_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.block_size <= 384:
        parser.error("--block-size must exceed min_overlap + 2*context (384)")

    method_args = list(args.method_args)
    if method_args and method_args[0] == "--":
        method_args.pop(0)

    import bin2cell as b2c

    upstream_stardist = b2c.stardist

    def stardist_with_explicit_block_size(*positional, **kwargs):
        kwargs.setdefault("block_size", args.block_size)
        return upstream_stardist(*positional, **kwargs)

    b2c.stardist = stardist_with_explicit_block_size

    from segbench.methods.bin2cell import main as bin2cell_main

    return int(bin2cell_main(method_args) or 0)


if __name__ == "__main__":
    sys.exit(main())
