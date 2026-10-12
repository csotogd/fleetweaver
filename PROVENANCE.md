# Provenance and scope

Fleetweaver means weaving routes into a coordinated fleet. It is an independent
reproduction and empirical study built on **PARCO: Parallel AutoRegressive Models
for Multi-Agent Combinatorial Optimization**, NeurIPS 2025, by Federico Berto,
Chuanbo Hua, Laurin Luttmann, Jiwoo Son, Junyoung Park, Kyuree Ahn, Changhyun Kwon,
Lin Xie and Jinkyoo Park.

- Paper: https://proceedings.neurips.cc/paper_files/paper/2025/hash/f404c0259def290306159047b0b52584-Abstract-Conference.html
- Original repository: https://github.com/ai4co/parco
- Vendored commit: `a118c704ff10539560578f886bce909ebfe55f94`
- The `parco/` package is copied from that commit, without algorithm changes.
- The original MIT copyright is retained in `LICENSE-PARCO` and `LICENSE`.
- RL4CO supplies the original state augmentation and symmetry loss implementation.

## What this repository contributes

The `fleetweaver/` package adds controlled synthetic distributions, a small-scale
training runner, a same-objective multi-trip OR-Tools baseline, independent route
feasibility and objective checks, held-out benchmarks, raw experimental logs, and
a visual report. These are experimental infrastructure and analysis, not a claim
to have invented PARCO or a new state-of-the-art routing architecture.

## Differences from the paper

- Focuses only on heterogeneous capacitated vehicle routing (HCVRP).
- Uses 64-dimensional embeddings and two encoder layers for laptop computation,
  rather than the official 128-dimensional experiment configuration.
- Preserves parallel decoding, agent communication, highest-probability conflict
  resolution, and the symmetry REINFORCE objective plus 0.2 invariance loss.
- Uses four augmentations, gradient clipping at 1.0, and Adam at 1e-4.
- Trains on 20–40 customers and 3–5 vehicles. The mixed regime changes the spatial
  data distribution; the uniform regime is an equal-budget ablation.
- All local weights start from random initialization. No pretrained weights are
  presented as locally trained models.
- A reduced-scale reproduction does not reproduce the paper's published tables.

## Evaluation contract

Minimize `max_v(distance_v / speed_v)`. All vehicles share one depot, start and
finish there, can reload capacity on depot visits, and must collectively serve
each customer exactly once. There is no service time and no travel waiting cost.
Reported makespans are recomputed independently from float coordinates.

Validation seeds are fixed and separate from training and test. Checkpoints and
the representative seed are selected using validation only. Test scenarios are
declared in `configs/study.json` before final evaluation; test results are not used
to tune the models. Out-of-distribution tests are ring/stripe geometry, demand and
fleet heterogeneity shifts, and customer/fleet sizes beyond the training range.
Clustered and corner layouts occur in mixed training and are not called unseen.

OR-Tools uses vehicle-specific time and capacity callbacks, optional depot reload
copies, and a global-span objective with zero arc costs. It is warm-started with
balanced greedy plus intra-trip 2-opt. Each vehicle has N-1 reload copies, enough
for N non-empty trips; the search time limit excludes model construction, which
is included in measured end-to-end latency. Rounding scale: 100,000. OR-Tools is
a time-limited reference, not a proven optimum. Large reload models can be slow.
Both feasible fallback and solver failures must be explicitly reported.

Timings use the same host and CPU thread setting. Batch throughput is distinct
from single-instance latency. Training cost is reported separately. Relative gaps
to OR-Tools are not optimality gaps. Confidence intervals over instances do not
replace variation over independent training seeds.
