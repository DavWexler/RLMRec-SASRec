"""Train/test all three recommender models across multiple seeds on MovieLens-1M.

Models (runs in this order):
  1. sasrec       — pure SASRec (paper-faithful)
  2. rlmrec_lightgcn — RLMRec-Con with a LightGCN backbone (paper-faithful RLMRec)
  3. rlmrec_sasrec   — RLMRec-Con with a SASRec backbone (hypothesis test)

For each model and each seed, this orchestrator:
  - invokes the training script as a subprocess (streaming its stdout) into
    `<out_dir>/seed_<N>/`,
  - reads each seed's `test_metrics.csv`,
  - aggregates across seeds (mean ± std), and across all three models, into
    `artifacts/comparison/`:
      * comparison.csv             — rows: metric, cols: per-model mean (legacy)
      * comparison_per_seed.csv    — long format: model, seed, metric, value
      * comparison_aggregate.csv   — rows: metric, cols: per-model mean+std+n
      * comparison_bars.png        — grouped bars with ±std error bars
      * val_NDCG@10_curves.png     — per-model mean curve + ±1σ band
      * summary.txt                — aggregate table + per-seed NDCG@10/HR@10
  - and at each model's `out_dir` root:
      * test_metrics.csv           — mean per metric (legacy schema)
      * test_metrics_per_seed.csv  — rows: seed, cols: metrics
      * test_metrics_aggregate.csv — rows: metric, cols: mean,std,n
      * run_metadata.json          — copied from the first seed run
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import seed_aggregation as agg


MODELS = [
    {
        "name": "sasrec",
        "label": "SASRec",
        "script": "sasrec_train.py",
        "out_dir": "artifacts/sasrec",
    },
    {
        "name": "rlmrec_lightgcn",
        "label": "RLMRec+LightGCN",
        "script": "rlmrec_lightgcn_train.py",
        "out_dir": "artifacts/rlmrec_lightgcn",
    },
    {
        "name": "rlmrec_sasrec",
        "label": "RLMRec+SASRec",
        "script": "rlmrec_sasrec_train.py",
        "out_dir": "artifacts/rlmrec_sasrec",
    },
]

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def run_training(script: str, seed_out_dir: str, seed: int,
                 extra_args: list[str]) -> None:
    cmd = [sys.executable, script, "--out_dir", seed_out_dir,
           "--seed", str(seed), *extra_args]
    print(f"\n{'=' * 72}\n>>> {' '.join(cmd)}\n{'=' * 72}", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"{script} (seed {seed}) exited with code "
                           f"{proc.returncode}")


def consolidate_model_outputs(model_out_dir: Path, seeds: list[int]
                              ) -> tuple[dict[int, dict[str, float]],
                                         dict[str, dict[str, float]]]:
    """Read per-seed test metrics and write per-model aggregate artifacts.

    Returns (per_seed_metrics, aggregate). `aggregate` may be empty if no
    seed runs produced a `test_metrics.csv`.
    """
    per_seed = agg.read_seed_test_metrics(model_out_dir, seeds)
    if not per_seed:
        return per_seed, {}
    aggregate = agg.aggregate_metrics(per_seed)
    agg.write_per_seed_metrics_csv(
        per_seed, model_out_dir / "test_metrics_per_seed.csv")
    agg.write_aggregate_metrics_csv(
        aggregate, model_out_dir / "test_metrics_aggregate.csv")
    # Mean-only file at the model root, keeping legacy consumers (e.g.
    # make_overview.py) working.
    agg.write_mean_metrics_csv(aggregate, model_out_dir / "test_metrics.csv")
    # Copy run_metadata.json from the first available seed dir as the
    # representative config (training args are identical across seeds).
    for s in seeds:
        meta_src = agg.seed_subdir(model_out_dir, s) / "run_metadata.json"
        if meta_src.exists():
            shutil.copy2(meta_src, model_out_dir / "run_metadata.json")
            break
    return per_seed, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/test SASRec, RLMRec+LightGCN, RLMRec+SASRec across "
                    "multiple seeds.")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=[m["name"] for m in MODELS],
                        help="Model names to skip (e.g. --skip sasrec).")
    parser.add_argument("--compare_only", action="store_true",
                        help="Skip training entirely, just aggregate existing "
                             "per-seed results.")
    parser.add_argument("--compare_dir", type=str, default="artifacts/comparison")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override --epochs for every training script.")
    parser.add_argument("--eval_every", type=int, default=None,
                        help="Override --eval_every for every training script.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help=f"Seeds to run each model with. "
                             f"Default: {DEFAULT_SEEDS}.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single-seed legacy alias for --seeds <N>.")
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="Extra args forwarded to each training script "
                             "(e.g. `-- --data_dir data`).")
    args = parser.parse_args()

    if args.seeds is not None and args.seed is not None:
        raise SystemExit("Pass either --seeds or --seed, not both.")
    if args.seeds:
        seeds = list(dict.fromkeys(args.seeds))  # de-dup, preserve order
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(DEFAULT_SEEDS)

    forward: list[str] = []
    if args.epochs is not None:
        forward += ["--epochs", str(args.epochs)]
    if args.eval_every is not None:
        forward += ["--eval_every", str(args.eval_every)]
    extra = [a for a in (args.extra or []) if a != "--"]
    forward += extra

    print(f"Seeds: {seeds}")

    if not args.compare_only:
        for model in MODELS:
            if model["name"] in args.skip:
                print(f"Skipping {model['name']} (--skip)")
                continue
            for s in seeds:
                seed_dir = str(agg.seed_subdir(Path(model["out_dir"]), s))
                run_training(model["script"], seed_dir, s, forward)

    by_model_per_seed: dict[str, dict[int, dict[str, float]]] = {}
    by_model_agg: dict[str, dict[str, dict[str, float]]] = {}
    by_model_curves: dict[str, dict[int, dict[int, float]]] = {}
    for model in MODELS:
        out_dir = Path(model["out_dir"])
        per_seed, aggregate = consolidate_model_outputs(out_dir, seeds)
        if not aggregate:
            print(f"[warn] no per-seed test_metrics under {out_dir}, "
                  f"skipping in comparison")
            continue
        by_model_per_seed[model["label"]] = per_seed
        by_model_agg[model["label"]] = aggregate
        by_model_curves[model["label"]] = agg.read_seed_val_curves(out_dir, seeds)

    if not by_model_agg:
        print("No per-seed test_metrics.csv files found; nothing to compare.")
        return

    compare_dir = Path(args.compare_dir)
    compare_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = agg.write_comparison_aggregate_csv(
        by_model_agg, compare_dir / "comparison_aggregate.csv")
    agg.write_comparison_per_seed_csv(
        by_model_per_seed, compare_dir / "comparison_per_seed.csv")
    agg.write_comparison_mean_csv(by_model_agg, compare_dir / "comparison.csv")

    agg.plot_comparison_bars_with_err(
        by_model_agg, metric_keys, compare_dir / "comparison_bars.png",
        title="Test metrics — mean ± std across seeds, MovieLens-1M (full-rank)")
    agg.plot_val_curves_with_band(
        by_model_curves, compare_dir / "val_NDCG@10_curves.png",
        title="Validation NDCG@10 — mean ± 1σ across seeds, MovieLens-1M")

    summary = agg.format_aggregate_summary(by_model_agg, metric_keys)
    summary += "\n" + agg.format_per_seed_table(by_model_per_seed, "NDCG@10")
    summary += "\n" + agg.format_per_seed_table(by_model_per_seed, "HR@10")
    print("\n" + "=" * 72)
    print("Test metrics comparison — mean ± std across seeds (MovieLens-1M)")
    print("=" * 72)
    print(summary)
    with open(compare_dir / "summary.txt", "w") as f:
        f.write(summary + "\n")

    print(f"\nComparison artifacts written to: {compare_dir}")


if __name__ == "__main__":
    main()
