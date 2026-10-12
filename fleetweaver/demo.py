"""Solve and visualize a new synthetic instance using the published weights."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fleetweaver.baselines import greedy, two_opt
from fleetweaver.core import (
    actions_to_routes,
    generate,
    instance,
    make_policy,
    predict,
    seed_all,
    validate_routes,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=40)
    parser.add_argument("--vehicles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234567891)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--distribution",
        choices=[
            "uniform",
            "clustered",
            "corner",
            "ring",
            "stripe",
            "heavy_demand",
            "heterogeneous",
        ],
        default="uniform",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("demo-output"))
    args = parser.parse_args()
    if min(args.customers, args.vehicles, args.samples) < 1:
        parser.error("Customers, vehicles and samples must be positive")
    root = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint
    if checkpoint is None:
        metadata = root / "results/benchmark-metadata.json"
        name = (
            json.loads(metadata.read_text())["representative"]
            if metadata.exists()
            else "mixed-s42"
        )
        checkpoint = root / f"checkpoints/{name}-best.pt"
    torch.set_num_threads(2)
    seed_all(args.seed)
    policy = make_policy()
    policy.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)["policy"]
    )
    td = generate(args.customers, args.vehicles, 1, args.seed, args.distribution)
    inst = instance(td[0])
    out = predict(policy, td, samples=args.samples)
    routes = actions_to_routes(out["actions"][0].cpu().numpy(), args.vehicles)
    solutions = {
        f"PARCO · {args.samples} sample(s)": routes,
        "Balanced + 2-opt": two_opt(inst, greedy(inst)),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    records = {}
    xy = np.vstack([inst["depot"], inst["locs"]])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for ax, (name, paths) in zip(axes, solutions.items()):
        score, times = validate_routes(inst, paths)
        records[name] = dict(routes=paths, makespan=score, vehicle_times=times)
        for v, path in enumerate(paths):
            ax.plot(
                xy[path, 0], xy[path, 1], alpha=0.75, lw=1.5, label=f"Vehicle {v + 1}"
            )
        ax.scatter(xy[1:, 0], xy[1:, 1], s=18, c="#253a43", zorder=3)
        ax.scatter(
            *xy[0], marker="*", s=160, c="#f09546", edgecolors="#253a43", zorder=4
        )
        ax.set(
            title=f"{name}\nMakespan {score:.3f}",
            xlim=(-0.05, 1.05),
            ylim=(-0.05, 1.05),
            aspect="equal",
        )
    fig.suptitle(
        f"Fleetweaver · {args.customers} customers / {args.vehicles} vehicles · seed {args.seed}"
    )
    fig.savefig(args.output / "routes.png", dpi=170)
    plt.close(fig)
    payload = dict(
        seed=args.seed,
        distribution=args.distribution,
        checkpoint=checkpoint.name,
        instance={k: v.tolist() for k, v in inst.items()},
        solutions=records,
    )
    (args.output / "solution.json").write_text(json.dumps(payload, indent=2))
    print(
        json.dumps({name: row["makespan"] for name, row in records.items()}, indent=2)
    )
    print(f"Saved checked routes and visualization to {args.output.resolve()}")


if __name__ == "__main__":
    main()
