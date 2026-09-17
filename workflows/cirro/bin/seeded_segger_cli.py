#!/usr/bin/env python3
"""Launch Segger v0.2.0 with deterministic settings and bounded workers."""
from __future__ import annotations

import argparse
import os
import random
import sys
from functools import wraps


def force_num_workers(num_workers: int) -> None:
    """Override Segger's hidden ``ISTDataModule.num_workers`` default.

    Segger v0.2.0 does not expose this dataclass field in its CLI.  Its default
    of eight worker processes uses Docker ``/dev/shm`` and fails on Cirro with
    SIGBUS even when hundreds of GiB of ordinary RAM are requested.  Patching
    the constructor is deliberately narrower than modifying the pinned Segger
    container: all algorithm parameters and the upstream implementation remain
    unchanged, while ``num_workers=0`` performs loading in the main process and
    therefore does not depend on container shared memory.
    """
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    from segger.data import ISTDataModule

    original_init = ISTDataModule.__init__

    @wraps(original_init)
    def bounded_init(self, *args, **kwargs):
        kwargs["num_workers"] = num_workers
        return original_init(self, *args, **kwargs)

    ISTDataModule.__init__ = bounded_init


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Segger DataLoader workers; 0 avoids Cirro /dev/shm SIGBUS.")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")

    import numpy as np
    import torch
    from lightning import seed_everything

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    seed_everything(args.seed, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    force_num_workers(args.num_workers)
    print(f"[segbench] Segger DataLoader num_workers={args.num_workers}", flush=True)
    sys.argv = ["segger", *command]
    from segger.cli.main import app
    app()


if __name__ == "__main__":
    main()
