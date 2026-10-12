"""Build publication-style figures and a self-contained, offline visual report."""

import base64
import hashlib
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
COLORS = ["#166d80", "#f09546", "#7b64ae", "#4d967d", "#c26470", "#7b8796"]


def ci(values, seed=777):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (2000, len(values)))].mean(1)
    return np.percentile(means, [2.5, 97.5]).tolist()


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "results/benchmark.csv")
    meta = json.loads((ROOT / "results/benchmark-metadata.json").read_text())
    representative = meta["representative"]
    labels = {
        representative: "Mixed · greedy",
        "mixed-sample16": "Mixed · sample 16",
        "mixed-sample16-2opt": "Mixed · sample 16 + 2-opt",
        "uniform-s42": "Uniform · greedy",
        "balanced-2opt": "Balanced + 2-opt",
        "ortools-1s": "OR-Tools · 1 s",
        "nearest": "Nearest neighbour",
        "balanced-greedy": "Balanced greedy",
        "untrained": "Untrained PARCO",
    }
    methods = [
        "untrained",
        "uniform-s42",
        representative,
        "mixed-sample16",
        "balanced-2opt",
        "ortools-1s",
    ]
    pivot = df.pivot(
        index=["scenario", "instance"], columns="method", values="makespan"
    )
    gap = (pivot.div(pivot["ortools-1s"], axis=0) - 1) * 100
    summary = []
    for method in pivot.columns:
        lo, hi = ci(gap[method])
        summary.append(
            dict(
                method=method,
                gap_pct=float(gap[method].mean()),
                ci_low=lo,
                ci_high=hi,
                win_rate_pct=float(
                    (pivot[method] < pivot["ortools-1s"] - 1e-6).mean() * 100
                ),
                mean_makespan=float(pivot[method].mean()),
                mean_seconds=float(df[df.method == method].seconds.mean()),
            )
        )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT / "summary.csv", index=False)
    scenarios = json.loads((ROOT / "configs/study.json").read_text())["scenarios"]
    scenario_order = [s[0] for s in scenarios if s[0] in gap.index.get_level_values(0)]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fcfcfa",
            "axes.facecolor": "#fcfcfa",
            "axes.labelcolor": "#253a43",
            "text.color": "#253a43",
            "xtick.color": "#53646b",
            "ytick.color": "#53646b",
            "savefig.facecolor": "#fcfcfa",
        }
    )

    def save(fig, name):
        fig.savefig(OUT / f"{name}.png", dpi=170, bbox_inches="tight")
        fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    overview_methods = methods[:4] + ["mixed-sample16-2opt"] + methods[4:]
    means = [gap[m].mean() for m in overview_methods]
    intervals = np.array([ci(gap[m]) for m in overview_methods])
    positions = np.arange(len(overview_methods))
    ax.barh(
        positions,
        means,
        color=[
            "#bcc5ca",
            COLORS[2],
            COLORS[0],
            COLORS[1],
            COLORS[4],
            COLORS[3],
            "#253a43",
        ],
        height=0.58,
    )
    ax.errorbar(
        means,
        positions,
        xerr=np.maximum(
            0, np.stack([np.array(means) - intervals[:, 0], intervals[:, 1] - means])
        ),
        fmt="none",
        color="#253a43",
        capsize=3,
    )
    ax.set_yticks(positions, [labels.get(m, m) for m in overview_methods])
    ax.invert_yaxis()
    ax.axvline(0, color="#253a43", lw=1)
    ax.set_xlabel("Mean paired makespan gap to OR-Tools (%) · lower is better")
    ax.set_title("What did the policy learn?", loc="left", fontsize=18, pad=18)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    fig.text(
        0.01,
        -0.025,
        "95% bootstrap intervals over held-out instances. OR-Tools is a time-limited reference, not an optimum.",
        fontsize=9,
    )
    save(fig, "overview")
    fig, ax = plt.subplots(figsize=(11, 8))
    heat = gap.groupby(level="scenario").mean().loc[scenario_order, methods[1:5]]
    lim = max(10, float(np.abs(heat.to_numpy()).max()))
    im = ax.imshow(heat, aspect="auto", cmap="RdYlBu_r", vmin=-lim, vmax=lim)
    ax.set_yticks(range(len(heat)), heat.index)
    ax.set_xticks(
        range(len(heat.columns)),
        [labels.get(m, m) for m in heat.columns],
        rotation=15,
        ha="right",
    )
    for i in range(len(heat)):
        for j in range(len(heat.columns)):
            val = heat.iloc[i, j]
            ax.text(
                j,
                i,
                f"{val:+.1f}%",
                ha="center",
                va="center",
                color="white" if abs(val) > lim * 0.65 else "#253a43",
                fontsize=10,
            )
    ax.set_title(
        "Generalization is a scenario-by-scenario question",
        loc="left",
        fontsize=17,
        pad=18,
    )
    fig.colorbar(im, ax=ax, label="Gap to OR-Tools (%)", shrink=0.7)
    save(fig, "generalization")
    ablation = (1 - pivot["mixed-s42"] / pivot["uniform-s42"]) * 100
    ablation_rows = []
    for name in scenario_order:
        values = ablation.loc[name]
        low, high = ci(values)
        ablation_rows.append(
            dict(
                scenario=name,
                improvement_pct=float(values.mean()),
                ci_low=low,
                ci_high=high,
            )
        )
    pd.DataFrame(ablation_rows).to_csv(OUT / "ablation.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    means = np.array([row["improvement_pct"] for row in ablation_rows])
    intervals = np.array([[row["ci_low"], row["ci_high"]] for row in ablation_rows])
    ax.barh(
        range(len(scenario_order)),
        means,
        color=[COLORS[0] if v >= 0 else COLORS[1] for v in means],
        height=0.6,
    )
    ax.errorbar(
        means,
        range(len(scenario_order)),
        xerr=np.maximum(
            0, np.stack([means - intervals[:, 0], intervals[:, 1] - means])
        ),
        fmt="none",
        color="#253a43",
        capsize=3,
    )
    ax.set_yticks(range(len(scenario_order)), scenario_order)
    ax.invert_yaxis()
    ax.axvline(0, color="#253a43", lw=1)
    ax.set_xlabel("Makespan reduction from mixed training (%) · positive favours mixed")
    ax.set_title(
        "Does varied training help? · fixed seed 42", loc="left", fontsize=17, pad=18
    )
    ax.grid(axis="x", alpha=0.15)
    fig.text(
        0.01,
        -0.02,
        "Single-seed ablation; 95% intervals reflect held-out instances, not training-seed uncertainty.",
        fontsize=9,
    )
    save(fig, "ablation")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    training = []
    for color, name in zip(
        COLORS, ["mixed-s42", "mixed-s43", "mixed-s44", "uniform-s42"]
    ):
        rows = [
            json.loads(line)
            for line in (ROOT / f"results/train-{name}.jsonl").read_text().splitlines()
        ]
        vals = [r for r in rows if "validation" in r]
        ax.plot(
            [r["step"] for r in vals],
            [r["validation"] for r in vals],
            label=name,
            color=color,
            lw=2,
        )
        best = min(vals, key=lambda r: r["validation"])
        training.append(
            dict(
                run=name,
                initial=vals[0]["validation"],
                best=best["validation"],
                best_step=best["step"],
                steps=vals[-1]["step"],
                seconds=vals[-1]["seconds"],
            )
        )
    ax.set_xlabel("Optimizer updates")
    ax.set_ylabel("Validation makespan (lower is better)")
    ax.set_title(
        "Training from scratch · fixed validation set", loc="left", fontsize=18, pad=18
    )
    ax.legend(ncol=2, frameon=False)
    ax.grid(alpha=0.15)
    save(fig, "learning")
    pd.DataFrame(training).to_csv(OUT / "training.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    scale_scenarios = [
        s
        for s in scenarios
        if s[0] in ["id40", "size60", "fixedfleet80", "fixedfleet100"]
    ]
    for color, method in zip(
        COLORS, [representative, "mixed-sample16", "uniform-s42", "balanced-2opt"]
    ):
        ax.plot(
            [s[1] for s in scale_scenarios],
            [gap.loc[s[0], method].mean() for s in scale_scenarios],
            "o-",
            label=labels[method],
            color=color,
            lw=2,
        )
    ax.axvspan(20, 40, color=COLORS[0], alpha=0.07, label="Training customer range")
    ax.axhline(0, color="#253a43", lw=1)
    ax.set_xlabel("Customers · fleet fixed at 5 vehicles")
    ax.set_ylabel("Mean gap to OR-Tools (%)")
    ax.set_title("More customers, same fleet", loc="left", fontsize=18, pad=18)
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.15)
    save(fig, "scaling")
    latency = pd.read_csv(ROOT / "results/latency.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    time_methods = ["mixed-greedy", "mixed-sample16", "balanced-2opt", "ortools-1s"]
    time_labels = {**labels, "mixed-greedy": "Mixed · greedy"}
    for color, method in zip(COLORS, time_methods):
        medians, lows, highs = [], [], []
        for n, scenario in [(20, "id20"), (40, "id40"), (100, "size100")]:
            if method.startswith("mixed"):
                values = latency[(latency.n == n) & (latency.method == method)].seconds
            else:
                values = df[(df.scenario == scenario) & (df.method == method)].seconds
            lows.append(values.quantile(0.25))
            medians.append(values.median())
            highs.append(values.quantile(0.75))
        ax.plot(range(3), medians, "o-", color=color, label=time_labels[method], lw=2)
        ax.fill_between(range(3), lows, highs, color=color, alpha=0.10)
    ax.set_xticks(range(3), ["20 / 3", "40 / 5", "100 / 10"])
    ax.set_xlabel("Customers / vehicles")
    ax.set_ylabel("Wall-clock seconds per instance (log scale)")
    ax.set_yscale("log")
    ax.set_title(
        "Single-instance latency · shared laptop", loc="left", fontsize=18, pad=18
    )
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.15)
    fig.text(
        0.01,
        -0.03,
        "Medians and interquartile ranges. Neural: 5 instances × 3 repetitions; classical: 32 test instances.\nIncludes solver setup; excludes model loading, training and route validation. No amortized batch timings here.",
        fontsize=9,
    )
    save(fig, "latency")
    total_instances = len(pivot)
    valid_count = int(df.valid.sum())
    model_gap = float(gap[representative].mean())
    sample_gap = float(gap["mixed-sample16"].mean())
    hybrid_gap = float(gap["mixed-sample16-2opt"].mean())
    improvement = float(
        ((pivot.untrained - pivot[representative]) / pivot.untrained * 100).mean()
    )
    wins = int((gap.groupby(level="scenario")["mixed-sample16"].mean() < 0).sum())
    seed_gaps = {
        m: float(gap[m].mean()) for m in ["mixed-s42", "mixed-s43", "mixed-s44"]
    }
    final = dict(
        instances=total_instances,
        scenarios=len(scenario_order),
        validated_solutions=valid_count,
        representative=representative,
        greedy_gap_pct=model_gap,
        sampling_gap_pct=sample_gap,
        hybrid_gap_pct=hybrid_gap,
        improvement_over_untrained_pct=improvement,
        scenarios_beating_reference=wins,
        mixed_seed_gaps_pct=seed_gaps,
        solver_fallbacks=int(df.solver_fallback.sum()),
        training=training,
    )
    (OUT / "summary.json").write_text(json.dumps(final, indent=2))
    table = gap.groupby(level="scenario").mean().loc[scenario_order, methods[1:5]]
    markdown_table = "| Scenario | Uniform | Mixed greedy | Mixed sample 16 | Balanced + 2-opt |\n|---|---:|---:|---:|---:|\n"
    for name, row in table.iterrows():
        markdown_table += (
            f"| {name} | " + " | ".join(f"{v:+.2f}%" for v in row) + " |\n"
        )
    hours = sum(t["seconds"] for t in training) / 3600
    narrative = (
        f"The validation-selected mixed policy improves makespan by {improvement:.1f}% on average relative to the untrained policy. "
        f"Its mean paired gap to the 1-second OR-Tools reference is {model_gap:+.1f}% with greedy decoding and {sample_gap:+.1f}% with 16 sampled solutions. "
        f"Adding intra-trip 2-opt to sampled routes changes that gap to {hybrid_gap:+.1f}%; this is a separate hybrid variant, not the neural policy alone. "
        f"Sampling beats the reference on average in {wins} of {len(scenario_order)} scenarios. These are measured comparisons, not optimality guarantees."
    )
    md = f"""# Fleetweaver — measured reproduction report

{narrative}

![Overview](overview.png)

## Experimental coverage

- {total_instances} held-out instances across {len(scenario_order)} declared scenarios.
- {valid_count} evaluated solutions independently checked for unique service, capacity, refills and depot return.
- Three mixed-training seeds plus one equal-budget uniform-training ablation.
- Model selected on validation: `{representative}`. No test-based checkpoint selection.
- Recorded training wall time: {hours:.2f} hours. Runs may overlap; startup, pilots and rolled-back updates are excluded. This is not total elapsed project time.
- OR-Tools fallback count: {int(df.solver_fallback.sum())}.

## Generalization

All entries below are mean paired gaps to a time-limited OR-Tools reference; lower is better.
Clustered and corner geometry are seen by mixed training. Ring, stripe, high demand,
strong heterogeneity and sizes outside 20–40 customers / 3–5 vehicles are shifts.

{markdown_table}

![Generalization](generalization.png)
![Scaling](scaling.png)

## Learning and uncertainty

![Learning curves](learning.png)

![Fixed-seed data-mixture ablation](ablation.png)

Mixed-seed aggregate greedy gaps: {", ".join(f"{k}: {v:+.2f}%" for k, v in seed_gaps.items())}.
The bootstrap intervals in `summary.csv` resample instances (2,000 replicates).
They are conditional on these trained weights and scenario suite, not confidence
intervals over all possible training seeds. The uniform ablation has one seed and
supports only a preliminary comparison of data mixtures.

## Timing and limitations

![Single-instance latency](latency.png)

`benchmark.csv` distinguishes amortized batched neural inference from per-instance
classical timings. `latency.csv` measures neural batch-one latency after warmup,
with five instances and three repetitions per size and decoding method. Loading,
training, plotting and the independent route checker are excluded from inference
timings. The shared laptop was busy and under memory pressure; these timings are
exploratory host measurements, not isolated-hardware speed claims.
OR-Tools time includes Python model construction and warm start, in addition
to its one-second search budget; extra reload nodes make large cases expensive.

This is a reduced-size PARCO reproduction, trained on synthetic Euclidean data.
It does not establish performance on road networks, time windows, real logistics,
or the full original benchmark. OR-Tools is not an exact optimum. Better results
on a finite test suite do not guarantee generalization to arbitrary instances.

See [provenance](../PROVENANCE.md), [protocol](../configs/study.json),
[raw benchmark](../results/benchmark.csv), [weights](../checkpoints/), and
[offline interactive report](report.html).
"""
    (OUT / "REPORT.md").write_text(md)
    readme = (ROOT / "README.md").read_text()
    start_marker, end_marker = (
        "<!-- measured-results-start -->",
        "<!-- measured-results-end -->",
    )
    measured = f"""## Measured results

{narrative}

**{total_instances} test problems · {len(scenario_order)} scenarios · {valid_count} checked solutions · 211,137 policy parameters.**

![Measured performance](reports/overview.png)
![Example routes](reports/routes.png)

- [Full research report](reports/REPORT.md)
- [Interactive route explorer](https://csotogd.github.io/fleetweaver/)
- [Offline HTML report](reports/report.html)
- [Per-instance benchmark](results/benchmark.csv) and [training logs](results/)
- [Trained checkpoints](checkpoints/) and [artifact checksums](MANIFEST.json)

The policy is smaller than the paper configuration. Reported gaps use a
time-limited classical reference, not exact optima. See the report for seed
variation, distribution shifts, timing definitions, and limitations.
"""
    before, rest = readme.split(start_marker, 1)
    _, after = rest.split(end_marker, 1)
    (ROOT / "README.md").write_text(
        before + start_marker + "\n" + measured + end_marker + after
    )
    # A route figure uses instance zero by policy, never the most flattering example.
    routes = json.loads((ROOT / "results/routes.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, method in zip(axes, [representative, "mixed-sample16", "ortools-1s"]):
        item = next(
            r for r in routes if r["scenario"] == "id30" and r["method"] == method
        )
        xy = np.vstack([item["instance"]["depot"], item["instance"]["locs"]])
        for v, path in enumerate(item["routes"]):
            ax.plot(
                xy[path, 0],
                xy[path, 1],
                "-",
                color=COLORS[v % len(COLORS)],
                lw=1.5,
                alpha=0.8,
            )
        ax.scatter(xy[1:, 0], xy[1:, 1], s=17, c="#253a43", zorder=3)
        ax.scatter(
            *xy[0], s=130, marker="*", c="#e99640", edgecolors="#253a43", zorder=4
        )
        ax.set_title(f"{labels[method]}\nMakespan {item['makespan']:.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(
        "Same problem, three solutions · first test instance, 30 customers / 4 vehicles",
        fontsize=15,
        y=1.06,
    )
    save(fig, "routes")
    render_html(final, narrative, table, routes, labels)
    manifest = []
    for folder in ["checkpoints", "results", "reports"]:
        for path in sorted((ROOT / folder).rglob("*")):
            if (
                path.is_file()
                and "pilot" not in path.name
                and "last.pt" not in path.name
                and "initial.pt" not in path.name
            ):
                manifest.append(
                    dict(
                        path=str(path.relative_to(ROOT)),
                        bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
    (ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(final, indent=2))


def render_html(summary, narrative, table, routes, labels):
    def img(name):
        return (
            "data:image/png;base64,"
            + base64.b64encode((OUT / f"{name}.png").read_bytes()).decode()
        )

    data = json.dumps(routes).replace("</", "<\\/")
    template = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fleetweaver — a measured PARCO reproduction</title>
<style>
:root{font-family:Inter,system-ui,sans-serif;color:#253a43;background:#fcfcfa;line-height:1.6}*{box-sizing:border-box}body{margin:0}main{max-width:1120px;margin:auto;padding:40px 24px 80px}header{padding:30px 0 40px;border-bottom:1px solid #dce3e5}.eyebrow{font-size:12px;letter-spacing:.15em;text-transform:uppercase;color:#166d80;font-weight:700}h1{font-size:clamp(42px,8vw,76px);line-height:1.03;margin:15px 0;font-weight:650;letter-spacing:-.055em}h2{font-size:27px;line-height:1.2;margin:44px 0 16px}p{max-width:850px}.sub{font-size:20px;color:#53646b}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;margin:30px 0}.stat strong{font-size:36px;display:block;color:#166d80}.stat span{color:#53646b}.chart{width:100%;height:auto;display:block;margin:18px 0 30px}.note{border-left:3px solid #f09546;padding:6px 18px;color:#53646b;font-size:14px}.controls{display:flex;gap:18px;flex-wrap:wrap;margin:20px 0}label{font-size:13px;color:#53646b;flex:1;min-width:190px}select{display:block;width:100%;padding:10px;border:1px solid #c5d0d4;border-radius:6px;background:#fff;font:inherit;color:#253a43}.routes{display:grid;grid-template-columns:1fr 1fr;gap:24px}.route h3{font-size:16px;margin-bottom:0}.route p{margin-top:3px;font-size:14px;color:#53646b}.route svg{width:100%;height:auto;background:#f3f6f5;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;text-align:right;border-bottom:1px solid #e1e7e7}th:first-child,td:first-child{text-align:left}.table-wrap{overflow:auto}a{color:#166d80}footer{margin-top:50px;font-size:13px;color:#53646b}.legend{font-size:13px;color:#53646b} @media(max-width:650px){main{padding:20px 15px}.routes{grid-template-columns:1fr}.stats{gap:10px}.stat strong{font-size:26px}.stat span{font-size:12px}}
</style><main><header><div class="eyebrow">NeurIPS 2025 · independent reproduction · Apple M4 / CPU</div><h1>Fleetweaver</h1><p class="sub">Can a small routing policy learn to coordinate a bigger fleet?</p><p>Original PARCO architecture. Local training from scratch. Held-out experiments with explicit checks and classical references.</p></header>
<div class="stats"><div class="stat"><strong>__INSTANCES__</strong><span>Held-out problems</span></div><div class="stat"><strong>__SCENARIOS__</strong><span>Declared test scenarios</span></div><div class="stat"><strong>__SOLUTIONS__</strong><span>Independently checked solutions</span></div></div>
<h2>The measured result</h2><p>__NARRATIVE__</p><img class="chart" alt="Mean paired gaps to the OR-Tools reference with bootstrap confidence intervals" src="__OVERVIEW__">
<p class="note">Lower makespan is better. A positive gap means the neural policy is worse than the reference. OR-Tools has a one-second search budget plus construction overhead; it is not a proven optimum. Sample 16 uses extra inference compute.</p>
<h2>Inspect the routes</h2><p>Compare solutions to the same problem. Every example is instance zero of its scenario, selected before looking at solution quality. Colours identify vehicles; the orange star is the shared depot. Lines show Euclidean travel.</p>
<div class="controls"><label>Problem scenario<select id="scenario"></select></label><label>Left solution<select id="left"></select></label><label>Right solution<select id="right"></select></label></div>
<div class="routes"><div class="route"><h3 id="left-title"></h3><p id="left-value" aria-live="polite"></p><svg id="left-svg" viewBox="0 0 500 500" role="img" aria-label="Left route comparison"></svg></div><div class="route"><h3 id="right-title"></h3><p id="right-value" aria-live="polite"></p><svg id="right-svg" viewBox="0 0 500 500" role="img" aria-label="Right route comparison"></svg></div></div>
<p class="legend" id="fleet"></p>
<h2>Generalization under pressure</h2><p>Training covers 20–40 customers and 3–5 vehicles. Mixed training sees uniform, clustered and corner-depot geometry. Ring and stripe geometry, high demand, stronger fleet heterogeneity and sizes outside the training range are distribution shifts.</p><img class="chart" alt="Per-scenario relative performance heatmap" src="__GENERALIZATION__"><img class="chart" alt="Scaling to more customers" src="__SCALING__">
<h2>Learning, not just decoding</h2><p>Three independent mixed-training seeds and one uniform-only ablation use the same update budget. Checkpoints and the representative mixed seed are selected on fixed validation problems, never on the test suite.</p><img class="chart" alt="Validation curves for all four training runs" src="__LEARNING__">
<h2>Varied training and runtime</h2><p>The data-mixture comparison fixes seed 42 for both runs. Latency plots use one problem at a time, with medians and interquartile ranges; they do not mix in amortized batch throughput.</p><img class="chart" alt="Fixed seed ablation of varied training" src="__ABLATION__"><img class="chart" alt="Single-instance solver latency" src="__LATENCY__"><h2>Numbers behind the figures</h2><p>Mean paired makespan gaps to OR-Tools, in percent. Lower is better.</p><div class="table-wrap">__TABLE__</div>
<h2>What this study cannot establish</h2><p>This smaller architecture and synthetic problem suite do not reproduce all tables in the original paper. Road networks, time windows, real delivery costs and arbitrary distributions are outside scope. The single-seed uniform ablation is preliminary. Bootstrap intervals describe test-instance variation conditional on trained weights; training-seed variation is reported separately in the repository.</p><p>Batch throughput and batch-one latency are stored separately. All inference benchmarks run after training has finished. The shared laptop was busy and under memory pressure, so timings are exploratory rather than isolated-hardware speed measurements. Raw results, checkpoints, exact dependencies, protocol and SHA-256 artifact hashes are included in the repository.</p>
<footer>Based on <a href="https://github.com/ai4co/parco">PARCO by Berto et al.</a>, NeurIPS 2025. Original MIT attribution retained. <a href="https://github.com/csotogd/fleetweaver">Code, data and full methodology</a>. This report runs offline and makes no external requests.</footer></main>
<script>
const data=__DATA__;const labels=__LABELS__;const colors=['#166d80','#f09546','#7b64ae','#4d967d','#c26470','#667aaf','#997144','#bd66a3','#408fba','#7b8796'];
const select=document.getElementById('scenario'),left=document.getElementById('left'),right=document.getElementById('right');
for(const name of [...new Set(data.map(d=>d.scenario))])select.add(new Option(name,name));
for(const method of [...new Set(data.map(d=>d.method))]){left.add(new Option(labels[method]||method,method));right.add(new Option(labels[method]||method,method));}
select.value='id30';left.value='__REPRESENTATIVE__';right.value='ortools-1s';
function render(side,method){const item=data.find(d=>d.scenario===select.value&&d.method===method);if(!item)return;const xy=[item.instance.depot,...item.instance.locs];const x=p=>30+440*p[0],y=p=>470-440*p[1];let s='';item.routes.forEach((route,v)=>{s+='<polyline points="'+route.map(i=>x(xy[i])+','+y(xy[i])).join(' ')+'" fill="none" stroke="'+colors[v%colors.length]+'" stroke-width="2.4" opacity=".8"/>';});xy.slice(1).forEach((p,i)=>{s+='<circle cx="'+x(p)+'" cy="'+y(p)+'" r="4" fill="#253a43"><title>Customer '+(i+1)+' · demand '+item.instance.demand[i]+'</title></circle>';});const dx=x(xy[0]),dy=y(xy[0]);let points=[];for(let k=0;k<10;k++){const a=-Math.PI/2+k*Math.PI/5,r=k%2?6:14;points.push((dx+Math.cos(a)*r)+','+(dy+Math.sin(a)*r));}s+='<polygon points="'+points.join(' ')+'" fill="#f09546" stroke="#253a43"><title>Shared depot</title></polygon>';document.getElementById(side+'-svg').innerHTML=s;document.getElementById(side+'-title').textContent=labels[method]||method;document.getElementById(side+'-value').textContent='Makespan '+item.makespan.toFixed(3)+' · '+item.routes.length+' vehicles';return item;}
function update(){const item=render('left',left.value);render('right',right.value);document.getElementById('fleet').textContent=item.instance.capacity.map((c,i)=>'Vehicle '+(i+1)+': capacity '+c+', speed '+item.instance.speed[i].toFixed(2)).join('  ·  ');}
select.addEventListener('change',update);left.addEventListener('change',update);right.addEventListener('change',update);update();
</script></html>"""
    replacements = {
        "INSTANCES": str(summary["instances"]),
        "SCENARIOS": str(summary["scenarios"]),
        "SOLUTIONS": str(summary["validated_solutions"]),
        "NARRATIVE": html.escape(narrative),
        "REPRESENTATIVE": summary["representative"],
        "DATA": data,
        "LABELS": json.dumps(labels),
        "TABLE": table.rename(columns=labels).round(2).to_html(border=0),
    }
    for name in [
        "overview",
        "generalization",
        "scaling",
        "learning",
        "ablation",
        "latency",
    ]:
        replacements[name.upper()] = img(name)
    for key, value in replacements.items():
        template = template.replace("__" + key + "__", value)
    (OUT / "report.html").write_text(template)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/index.html").write_text(template)
    (ROOT / "docs/.nojekyll").write_text("")


if __name__ == "__main__":
    main()
