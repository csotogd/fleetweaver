"""Recompute every published objective from saved input data and route sequences."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fleetweaver.core import validate_routes


def main():
    root = Path(__file__).resolve().parents[1]
    frame = pd.read_csv(root / "results/benchmark.csv")
    if frame.duplicated(["scenario", "instance", "method"]).any():
        raise AssertionError("Duplicate benchmark records")
    lookup = frame.set_index(["scenario", "instance", "method"])
    covered = set()
    for scenario in frame.scenario.unique():
        data = np.load(root / f"results/instances/{scenario}.npz")
        with gzip.open(root / f"results/solutions/{scenario}.json.gz", "rt") as stream:
            solutions = json.load(stream)
        for solution in solutions:
            i = solution["instance"]
            key = (scenario, i, solution["method"])
            if key in covered:
                raise AssertionError(f"Duplicate saved solution: {key}")
            inst = {
                k: data[k][i] for k in ["locs", "depot", "demand", "capacity", "speed"]
            }
            value, _ = validate_routes(inst, solution["routes"])
            if not np.isclose(value, lookup.loc[key, "makespan"], atol=1e-8, rtol=1e-8):
                raise AssertionError(f"Saved objective mismatch: {key}")
            covered.add(key)
    if covered != set(lookup.index):
        raise AssertionError("Saved routes do not cover the complete benchmark")
    checksums = 0
    if (root / "MANIFEST.json").exists():
        for record in json.loads((root / "MANIFEST.json").read_text()):
            actual = hashlib.sha256((root / record["path"]).read_bytes()).hexdigest()
            if actual != record["sha256"]:
                raise AssertionError(f"Artifact checksum mismatch: {record['path']}")
            checksums += 1
    result = dict(
        saved_solutions_checked=len(covered),
        independently_feasible=True,
        objectives_match=True,
        artifact_checksums_checked=checksums,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
