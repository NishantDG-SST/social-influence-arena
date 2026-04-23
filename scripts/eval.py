"""Baseline-vs-trained evaluation.

Runs ``inference.py`` twice (baseline model, trained model) on the same
episode seeds, aggregates per-task metrics, and emits matplotlib plots
comparing the two on the same axes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

TASKS = ["resist_pressure", "consistency_memory", "evidence_update"]


def run_inference(base_url: str, model_id: str, episodes: int, seed: int, out_dir: Path) -> Path:
    """Invoke the inference CLI and return the JSONL path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "social_influence_env.inference",
        "--base-url", base_url,
        "--model-id", model_id,
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--out-dir", str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    return out_dir / "inference.jsonl"


def aggregate(jsonl: Path) -> Dict[str, List[float]]:
    scores: Dict[str, List[float]] = {t: [] for t in TASKS}
    ends: Dict[str, dict] = {}
    with jsonl.open() as fh:
        for raw in fh:
            rec = json.loads(raw)
            if rec["type"] == "end":
                ends[rec["episode_id"]] = rec
    for rec in ends.values():
        scores.setdefault(rec["task_id"], []).append(rec["total_reward"])
    return scores


def plot_bars(baseline: Dict[str, List[float]], trained: Dict[str, List[float]], out: Path):
    import numpy as np
    tasks = TASKS
    b_means = [sum(baseline[t]) / len(baseline[t]) if baseline[t] else 0.0 for t in tasks]
    t_means = [sum(trained[t]) / len(trained[t]) if trained[t] else 0.0 for t in tasks]
    x = np.arange(len(tasks))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, b_means, w, label="Baseline (untrained)")
    ax.bar(x + w / 2, t_means, w, label="DPO-trained")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=15)
    ax.set_ylabel("Mean total reward")
    ax.set_title("Social Influence Arena — Baseline vs DPO-trained")
    ax.axhline(0.6, color="gray", linestyle="--", label="pass threshold (0.6)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_component_breakdown(jsonl: Path, label: str, out: Path):
    # Average reward component across end records.
    import numpy as np
    ends = []
    with jsonl.open() as fh:
        for raw in fh:
            rec = json.loads(raw)
            if rec["type"] == "end":
                ends.append(rec)
    if not ends:
        return
    components = {}
    for rec in ends:
        bd = rec["task_score"]["breakdown"]
        for k, v in bd.items():
            components.setdefault(k, []).append(v)
    names = list(components.keys())
    means = [float(np.mean(components[k])) for k in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, means, color="steelblue")
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("Mean score")
    ax.set_title(f"Reward breakdown — {label}")
    ax.axhline(0, color="black", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--baseline-model", default="scripted://always_agree")
    p.add_argument("--trained-model", default="scripted://always_truthful")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("assets/plots"))
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    print("Running baseline inference...")
    b_jsonl = run_inference(args.base_url, args.baseline_model, args.episodes, args.seed, args.out / "runs" / "baseline")
    print("Running trained inference...")
    t_jsonl = run_inference(args.base_url, args.trained_model, args.episodes, args.seed, args.out / "runs" / "trained")

    baseline = aggregate(b_jsonl)
    trained = aggregate(t_jsonl)

    plot_bars(baseline, trained, args.out / "reward_by_task.png")
    plot_component_breakdown(b_jsonl, "Baseline", args.out / "baseline_breakdown.png")
    plot_component_breakdown(t_jsonl, "DPO-trained", args.out / "trained_breakdown.png")

    print("\nBaseline means:")
    for t in TASKS:
        vals = baseline[t]
        if vals:
            print(f"  {t}: {sum(vals)/len(vals):.3f}  (n={len(vals)})")
    print("Trained means:")
    for t in TASKS:
        vals = trained[t]
        if vals:
            print(f"  {t}: {sum(vals)/len(vals):.3f}  (n={len(vals)})")
    print(f"\nPlots written to {args.out}")


if __name__ == "__main__":
    main()
