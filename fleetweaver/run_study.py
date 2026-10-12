"""One-command reproduction; existing checkpoints can resume interrupted training."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--steps", type=int, default=1800)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    for name, seed, regime in [
        ("mixed-s42", 42, "mixed"),
        ("mixed-s43", 43, "mixed"),
        ("mixed-s44", 44, "mixed"),
        ("uniform-s42", 42, "uniform"),
    ]:
        cmd = [
            sys.executable,
            "-m",
            "fleetweaver.train",
            "--name",
            name,
            "--seed",
            str(seed),
            "--regime",
            regime,
            "--steps",
            str(args.steps),
            "--threads",
            "2",
        ]
        if args.resume:
            cmd.append("--resume")
        subprocess.run(cmd, cwd=root, check=True)
    benchmark_cmd = [sys.executable, "-m", "fleetweaver.benchmark"]
    if args.resume:
        benchmark_cmd.append("--resume")
    subprocess.run(benchmark_cmd, cwd=root, check=True)
    subprocess.run([sys.executable, "-m", "fleetweaver.report"], cwd=root, check=True)
    subprocess.run([sys.executable, "-m", "fleetweaver.audit"], cwd=root, check=True)


if __name__ == "__main__":
    main()
