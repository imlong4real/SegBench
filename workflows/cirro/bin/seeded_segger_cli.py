#!/usr/bin/env python3
"""Launch Segger v0.2.0 with an explicit reproducibility seed."""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
from lightning import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    seed_everything(args.seed, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    sys.argv = ["segger", *command]
    from segger.cli.main import app
    app()


if __name__ == "__main__":
    main()
