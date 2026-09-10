#!/usr/bin/env python3
"""Run a command and record host and GPU resources without conflating them."""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # immutable method images need not carry this optional monitor
    psutil = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--requested-cpus", type=int, required=True)
    parser.add_argument("--requested-memory-gb", type=float, required=True)
    parser.add_argument("--requested-gpus", type=int, default=0)
    parser.add_argument("--method", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("missing command after --")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    peak_rss = 0
    gpu_peak = 0.0
    gpu_utils: list[float] = []
    gpu_active_samples = 0
    gpu_model = None
    stop = threading.Event()
    started = time.monotonic()
    with args.log.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=os.environ.copy())
        root = psutil.Process(process.pid) if psutil is not None else None
        last_cpu_seconds = 0.0

        def sample() -> None:
            nonlocal peak_rss, gpu_peak, gpu_active_samples, gpu_model, last_cpu_seconds
            while not stop.wait(1.0):
                if root is not None:
                    try:
                        family = [root] + root.children(recursive=True)
                        peak_rss = max(peak_rss, sum(p.memory_info().rss for p in family
                                                     if p.is_running()))
                        last_cpu_seconds = max(last_cpu_seconds,
                            sum(sum(p.cpu_times()[:2]) for p in family if p.is_running()))
                    except (psutil.Error, ProcessLookupError):
                        pass
                if args.requested_gpus:
                    try:
                        raw = subprocess.check_output([
                            "nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits"], text=True,
                            stderr=subprocess.DEVNULL, timeout=5)
                        for line in raw.splitlines():
                            name, mem, util = [part.strip() for part in line.split(",")]
                            gpu_model = gpu_model or name
                            gpu_peak = max(gpu_peak, float(mem))
                            gpu_utils.append(float(util))
                            if float(mem) > 0 or float(util) > 0:
                                gpu_active_samples += 1
                    except Exception:
                        pass

        worker = threading.Thread(target=sample, daemon=True)
        worker.start()
        rc = process.wait()
        stop.set(); worker.join(timeout=6)
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_seconds = max(last_cpu_seconds, usage.ru_utime + usage.ru_stime)
        # Linux reports ru_maxrss in KiB. It is a reliable fallback when psutil
        # is absent, though it cannot sum simultaneously-live descendants.
        peak_rss = max(peak_rss, int(usage.ru_maxrss * 1024))

    wall = time.monotonic() - started
    result = {
        "schema_version": "1.0", "method": args.method,
        "command": command, "exit_code": rc, "wall_clock_seconds": wall,
        "host": {
            "cpu_requested": args.requested_cpus,
            "memory_requested_gb": args.requested_memory_gb,
            "cpu_time_seconds": cpu_seconds,
            "average_cpu_cores_used": ((cpu_seconds / wall) if cpu_seconds is not None and wall else None),
            "peak_rss_gb": peak_rss / (1024 ** 3),
        },
        "gpu": {
            "gpu_requested": args.requested_gpus,
            "model": gpu_model,
            "peak_vram_mb": gpu_peak if gpu_model else None,
            "utilization_mean_percent": (sum(gpu_utils) / len(gpu_utils) if gpu_utils else None),
            "utilization_max_percent": (max(gpu_utils) if gpu_utils else None),
            "active_runtime_seconds_sampled": gpu_active_samples if gpu_utils else None,
            "sampling_interval_seconds": 1,
        },
        "notes": {
            "peak_rss": "sum of live process-tree RSS sampled each second; excludes GPU VRAM",
            "gpu": "nvidia-smi device samples; GPU VRAM is never added to host RSS",
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if rc:
        print(f"{args.method} child command failed with exit code {rc}; "
              f"last lines from {args.log}:", file=sys.stderr)
        try:
            with args.log.open(errors="replace") as failed_log:
                for line in deque(failed_log, maxlen=200):
                    print(line, end="", file=sys.stderr)
        except OSError as exc:
            print(f"unable to read child log: {exc}", file=sys.stderr)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
