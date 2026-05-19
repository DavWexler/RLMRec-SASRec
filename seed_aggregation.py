"""Multi-seed training-result aggregation helpers.

Used by `run_all.py` and `run_all_yelp.py` to:
  - read per-seed `test_metrics.csv` files written under
    `<model_out_dir>/seed_<N>/test_metrics.csv`
  - compute mean and std across seeds for each metric
  - write per-seed and aggregate CSVs (per model, and across models)
  - produce comparison bar plots with error bars and validation-curve bands

The on-disk layout this module expects/produces:

    <model_out_dir>/
        seed_42/test_metrics.csv          (written by the trainer)
        seed_42/train_history.csv         (written by the trainer)
        seed_43/...
        ...
        test_metrics.csv                  (mean per metric — kept for
                                          backward compat with make_overview)
        test_metrics_per_seed.csv         (rows: seed; cols: metrics)
        test_metrics_aggregate.csv        (rows: metric; cols: mean,std,n)

    <compare_dir>/
        comparison.csv                    (rows: metric; cols: per-model mean)
        comparison_per_seed.csv           (long format: model,seed,metric,value)
        comparison_aggregate.csv          (rows: metric; cols: per-model mean+std+n)
        comparison_bars.png               (grouped bars with error bars)
        val_NDCG@10_curves.png            (mean line + ±1σ band per model)
        summary.txt                       (mean ± std table + per-seed tables)
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def seed_subdir(model_out_dir: Path, seed: int) -> Path:
    return model_out_dir / f"seed_{seed}"


def _read_test_metrics(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            try:
                out[row[0]] = float(row[1])
            except ValueError:
                continue
    return out


def _read_val_ndcg10_curve(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    if not path.exists():
        return out
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            v = row.get("val_NDCG@10", "")
            if v == "" or v is None:
                continue
            try:
                out[int(row["epoch"])] = float(v)
            except (ValueError, KeyError):
                continue
    return out


def read_seed_test_metrics(model_out_dir: Path, seeds: list[int]
                           ) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for s in seeds:
        m = _read_test_metrics(seed_subdir(model_out_dir, s) / "test_metrics.csv")
        if m:
            out[s] = m
    return out


def read_seed_val_curves(model_out_dir: Path, seeds: list[int]
                         ) -> dict[int, dict[int, float]]:
    out: dict[int, dict[int, float]] = {}
    for s in seeds:
        c = _read_val_ndcg10_curve(seed_subdir(model_out_dir, s) / "train_history.csv")
        if c:
            out[s] = c
    return out


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return statistics.fmean(values), statistics.stdev(values)


def aggregate_metrics(per_seed: dict[int, dict[str, float]]
                      ) -> dict[str, dict[str, float]]:
    if not per_seed:
        return {}
    metric_keys = sorted({m for d in per_seed.values() for m in d})
    out: dict[str, dict[str, float]] = {}
    for mk in metric_keys:
        vals = [d[mk] for d in per_seed.values() if mk in d]
        mean, std = _mean_std(vals)
        out[mk] = {"mean": mean, "std": std, "n": float(len(vals))}
    return out


def write_per_seed_metrics_csv(per_seed: dict[int, dict[str, float]],
                               path: Path) -> None:
    if not per_seed:
        return
    seeds = sorted(per_seed.keys())
    metric_keys = sorted({m for d in per_seed.values() for m in d})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed"] + metric_keys)
        for s in seeds:
            row = [str(s)] + [f"{per_seed[s].get(mk, float('nan')):.6f}"
                              for mk in metric_keys]
            w.writerow(row)


def write_aggregate_metrics_csv(aggregate: dict[str, dict[str, float]],
                                path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std", "n"])
        for mk in sorted(aggregate.keys()):
            a = aggregate[mk]
            w.writerow([mk, f"{a['mean']:.6f}", f"{a['std']:.6f}",
                        f"{int(a['n'])}"])


def write_mean_metrics_csv(aggregate: dict[str, dict[str, float]],
                           path: Path) -> None:
    """Write mean-only `test_metrics.csv` (metric,value). This keeps
    `make_overview.py` and other consumers that expect the legacy schema
    working unchanged when seeds > 1."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for mk in sorted(aggregate.keys()):
            w.writerow([mk, f"{aggregate[mk]['mean']:.6f}"])


def write_comparison_per_seed_csv(
        by_model: dict[str, dict[int, dict[str, float]]],
        path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed", "metric", "value"])
        for model_label, per_seed in by_model.items():
            for seed in sorted(per_seed.keys()):
                for mk in sorted(per_seed[seed].keys()):
                    w.writerow([model_label, seed, mk,
                                f"{per_seed[seed][mk]:.6f}"])


def write_comparison_mean_csv(
        by_model_agg: dict[str, dict[str, dict[str, float]]],
        path: Path) -> list[str]:
    """Backward-compat: same shape the old single-seed `comparison.csv` had —
    rows = metric, cols = one per model with the mean value."""
    models = list(by_model_agg.keys())
    metric_keys = sorted({mk for agg in by_model_agg.values() for mk in agg})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + models)
        for mk in metric_keys:
            row = [mk] + [
                f"{by_model_agg[m].get(mk, {}).get('mean', float('nan')):.6f}"
                for m in models
            ]
            w.writerow(row)
    return metric_keys


def write_comparison_aggregate_csv(
        by_model_agg: dict[str, dict[str, dict[str, float]]],
        path: Path) -> list[str]:
    models = list(by_model_agg.keys())
    metric_keys = sorted({mk for agg in by_model_agg.values() for mk in agg})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["metric"]
        for m in models:
            header += [f"{m} mean", f"{m} std", f"{m} n"]
        w.writerow(header)
        for mk in metric_keys:
            row: list[str] = [mk]
            for m in models:
                a = by_model_agg[m].get(mk)
                if a:
                    row += [f"{a['mean']:.6f}", f"{a['std']:.6f}",
                            f"{int(a['n'])}"]
                else:
                    row += ["", "", ""]
            w.writerow(row)
    return metric_keys


def plot_comparison_bars_with_err(
        by_model_agg: dict[str, dict[str, dict[str, float]]],
        metric_keys: list[str], path: Path, title: str) -> None:
    models = list(by_model_agg.keys())
    n_metrics = len(metric_keys)
    n_models = len(models)
    if n_metrics == 0 or n_models == 0:
        return
    x = list(range(n_metrics))
    width = 0.8 / n_models

    plt.figure(figsize=(max(12, n_metrics * 1.1), 6))
    for mi, m in enumerate(models):
        means = [by_model_agg[m].get(mk, {}).get("mean", 0.0) for mk in metric_keys]
        stds = [by_model_agg[m].get(mk, {}).get("std", 0.0) for mk in metric_keys]
        offset = (mi - (n_models - 1) / 2) * width
        positions = [xi + offset for xi in x]
        bars = plt.bar(positions, means, width=width, yerr=stds,
                       label=m, capsize=3)
        for bar, mean, std in zip(bars, means, stds):
            plt.text(bar.get_x() + bar.get_width() / 2,
                     mean + std,
                     f"{mean:.3f}±{std:.3f}",
                     ha="center", va="bottom", fontsize=6, rotation=90)

    plt.xticks(x, metric_keys, rotation=30, ha="right")
    plt.ylabel("Score (mean ± std across seeds)")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_val_curves_with_band(
        by_model_curves: dict[str, dict[int, dict[int, float]]],
        path: Path, title: str) -> None:
    """Per model: mean line + ±1σ band over seeds, evaluated at the epochs
    common to all seeds of that model."""
    if not any(seed_curves for seed_curves in by_model_curves.values()):
        return
    plt.figure(figsize=(9, 5))
    plotted = 0
    for label, seed_curves in by_model_curves.items():
        if not seed_curves:
            continue
        epochs_per_seed = [set(c.keys()) for c in seed_curves.values()]
        common = set.intersection(*epochs_per_seed) if epochs_per_seed else set()
        eps_sorted = sorted(common)
        if not eps_sorted:
            continue
        means: list[float] = []
        stds: list[float] = []
        for ep in eps_sorted:
            vals = [c[ep] for c in seed_curves.values() if ep in c]
            mean, std = _mean_std(vals)
            means.append(mean)
            stds.append(std)
        line, = plt.plot(eps_sorted, means, marker="o", linewidth=1.5, label=label)
        upper = [m + s for m, s in zip(means, stds)]
        lower = [m - s for m, s in zip(means, stds)]
        plt.fill_between(eps_sorted, lower, upper, alpha=0.18,
                         color=line.get_color())
        plotted += 1
    if plotted == 0:
        plt.close()
        return
    plt.xlabel("Epoch")
    plt.ylabel("Validation NDCG@10")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def format_aggregate_summary(
        by_model_agg: dict[str, dict[str, dict[str, float]]],
        metric_keys: list[str]) -> str:
    models = list(by_model_agg.keys())
    if not models:
        return ""
    col_w = max(20, max(len(m) for m in models) + 2)
    lines = [f"{'metric':>12} " + " ".join(f"{m:>{col_w}}" for m in models)]
    lines.append("-" * len(lines[0]))
    for mk in metric_keys:
        row = f"{mk:>12}"
        for m in models:
            a = by_model_agg[m].get(mk)
            cell = f"{a['mean']:.4f} ± {a['std']:.4f}" if a else "—"
            row += f" {cell:>{col_w}}"
        lines.append(row)
    n_runs = max(
        (int(a["n"]) for d in by_model_agg.values() for a in d.values()),
        default=0,
    )
    lines.append(f"\n(aggregated over {n_runs} seed run(s) per cell)")
    return "\n".join(lines)


def format_per_seed_table(
        by_model: dict[str, dict[int, dict[str, float]]],
        metric: str) -> str:
    seeds = sorted({s for per_seed in by_model.values() for s in per_seed.keys()})
    models = list(by_model.keys())
    if not seeds or not models:
        return ""
    col_w = max(15, max(len(m) for m in models) + 2)
    header = f"{'seed':>6} " + " ".join(f"{m:>{col_w}}" for m in models)
    lines = [f"\nPer-seed {metric}:", header, "-" * len(header)]
    for s in seeds:
        row = f"{s:>6}"
        for m in models:
            v = by_model[m].get(s, {}).get(metric)
            row += f" {(f'{v:.4f}' if v is not None else '—'):>{col_w}}"
        lines.append(row)
    return "\n".join(lines)
