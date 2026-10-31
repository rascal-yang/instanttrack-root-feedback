#!/usr/bin/env python3
"""Recompute the published three-seed table and optionally plot its metrics/losses."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import math

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/three_seed"
ARMS = ("baseline", "root_position", "torso_position")


def recompute():
    audit = json.loads((RESULTS / "endpoint_independent_audit.json").read_text())
    summary = json.loads((RESULTS / "summary.json").read_text())
    if not audit["passed"] or not audit["source_hashes_unchanged"] or summary["checkpoint_reports"] != 36:
        raise ValueError("Published evidence is incomplete")
    result = {}
    for arm in ARMS:
        result[arm] = {}
        for name in ("mean_horizontal_error_cm", "mean_yaw_error_deg", "joint_rmse_deg"):
            values = [audit["seeds"][str(s)]["identical_prefix_audit"][arm][name] for s in (1, 2, 3)]
            result[arm][name] = dict(mean=statistics.mean(values), sample_std=statistics.stdev(values), values=values)
        result[arm]["completed_clips_per_seed"] = [
            next(r["completed"] for r in summary["results"] if r["seed"] == s and r["arm"] == arm and r["iteration"] == 6000)
            for s in (1, 2, 3)]
    expected = json.loads((RESULTS / "identical_prefix_summary.json").read_text())["results"]
    def same(a, b):
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        if isinstance(a, float):
            return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
        return a == b
    if not same(result, expected):
        raise ValueError("Published table does not match independent per-seed audit")
    if result["root_position"]["completed_clips_per_seed"] != [40, 40, 40]:
        raise ValueError("Root feedback did not complete all clips")
    return result


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    names = ["Baseline", "Root feedback", "Torso ablation"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric, label in zip(axes, ("mean_horizontal_error_cm", "joint_rmse_deg"),
                                 ("Horizontal position error (cm)", "Joint RMSE (degrees)")):
        ax.bar(names, [result[a][metric]["mean"] for a in ARMS],
               yerr=[result[a][metric]["sample_std"] for a in ARMS], capsize=4,
               color=["#8b929a", "#2875b6", "#e3983b"])
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Three fresh seeds; identical frame IDs within each seed")
    fig.tight_layout()
    fig.savefig(RESULTS / "endpoints.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    for ax, metric, title in zip(axes, ("wm_loss", "pi_loss", "v_loss"),
                                 ("World-model loss", "Policy loss", "Critic loss")):
        for arm, label in zip(ARMS, names):
            logs = []
            for seed in (1, 2, 3):
                with (RESULTS / "training_logs" / f"seed_{seed:03d}" / arm / "train_log.csv").open() as f:
                    logs.append({int(r["iteration"]): float(r[metric]) for r in csv.DictReader(f)})
            points = sorted(set.intersection(*(set(x) for x in logs)))
            if metric != "wm_loss":
                points = [x for x in points if x > 500]
            values = np.array([[row[x] for x in points] for row in logs])
            mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
            line, = ax.plot(points, mean, label=label, linewidth=1.3)
            ax.fill_between(points, mean - std, mean + std, color=line.get_color(), alpha=.12)
        ax.set(title=title, xlabel="Training iteration")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Mean ± sample SD over seeds 1/2/3; actor/critic start after WM warmup")
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_curves.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    p.add_argument("--plots", action="store_true")
    args = p.parse_args()
    result = recompute()
    if args.plots:
        plot(result)
    print(json.dumps(dict(verified=True, seeds=[1, 2, 3], checkpoint_reports=36, results=result), indent=2))


if __name__ == "__main__":
    main()
