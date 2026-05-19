"""Train/test all three recommender models across multiple seeds on the Yelp dataset.

Mirrors `run_all.py` but writes everything under `artifacts_yelp/` so the
existing MovieLens-1M results are not overwritten. Each training subprocess
is invoked with `--dataset yelp` and a Yelp-appropriate `--out_dir`.

Prerequisite: download the Yelp Open Dataset and place these files at
`data/yelp/`:
    yelp_academic_dataset_review.json
    yelp_academic_dataset_business.json
    yelp_academic_dataset_user.json

Per-seed runs land in `<out_dir>/seed_<N>/`; aggregate (mean ± std)
artifacts land in `artifacts_yelp/comparison/`. See `run_all.py` for the
full output schema, which is identical here.
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
        "out_dir": "artifacts_yelp/sasrec",
    },
    {
        "name": "rlmrec_lightgcn",
        "label": "RLMRec+LightGCN",
        "script": "rlmrec_lightgcn_train.py",
        "out_dir": "artifacts_yelp/rlmrec_lightgcn",
    },
    {
        "name": "rlmrec_sasrec",
        "label": "RLMRec+SASRec",
        "script": "rlmrec_sasrec_train.py",
        "out_dir": "artifacts_yelp/rlmrec_sasrec",
    },
]

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def run_training(script: str, seed_out_dir: str, seed: int,
                 extra_args: list[str]) -> None:
    cmd = [sys.executable, script, "--out_dir", seed_out_dir,
           "--dataset", "yelp", "--seed", str(seed), *extra_args]
    print(f"\n{'=' * 72}\n>>> {' '.join(cmd)}\n{'=' * 72}", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"{script} (seed {seed}) exited with code "
                           f"{proc.returncode}")


def consolidate_model_outputs(model_out_dir: Path, seeds: list[int]
                              ) -> tuple[dict[int, dict[str, float]],
                                         dict[str, dict[str, float]]]:
    per_seed = agg.read_seed_test_metrics(model_out_dir, seeds)
    if not per_seed:
        return per_seed, {}
    aggregate = agg.aggregate_metrics(per_seed)
    agg.write_per_seed_metrics_csv(
        per_seed, model_out_dir / "test_metrics_per_seed.csv")
    agg.write_aggregate_metrics_csv(
        aggregate, model_out_dir / "test_metrics_aggregate.csv")
    agg.write_mean_metrics_csv(aggregate, model_out_dir / "test_metrics.csv")
    for s in seeds:
        meta_src = agg.seed_subdir(model_out_dir, s) / "run_metadata.json"
        if meta_src.exists():
            shutil.copy2(meta_src, model_out_dir / "run_metadata.json")
            break
    return per_seed, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/test SASRec, RLMRec+LightGCN, RLMRec+SASRec on Yelp "
                    "across multiple seeds.")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=[m["name"] for m in MODELS],
                        help="Model names to skip (e.g. --skip sasrec).")
    parser.add_argument("--compare_only", action="store_true",
                        help="Skip training entirely, just aggregate existing "
                             "per-seed results.")
    parser.add_argument("--compare_dir", type=str,
                        default="artifacts_yelp/comparison")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help=f"Seeds to run each model with. "
                             f"Default: {DEFAULT_SEEDS}.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single-seed legacy alias for --seeds <N>.")
    parser.add_argument("--yelp_start_year", type=int, default=None)
    parser.add_argument("--yelp_min_inter", type=int, default=None)
    parser.add_argument("--also_make_overview", action="store_true",
                        help="After comparison, run make_overview.py --dataset yelp.")
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="Extra args forwarded to each training script.")
    args = parser.parse_args()

    if args.seeds is not None and args.seed is not None:
        raise SystemExit("Pass either --seeds or --seed, not both.")
    if args.seeds:
        seeds = list(dict.fromkeys(args.seeds))
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(DEFAULT_SEEDS)

    forward: list[str] = []
    if args.epochs is not None:
        forward += ["--epochs", str(args.epochs)]
    if args.eval_every is not None:
        forward += ["--eval_every", str(args.eval_every)]
    if args.yelp_start_year is not None:
        forward += ["--yelp_start_year", str(args.yelp_start_year)]
    if args.yelp_min_inter is not None:
        forward += ["--yelp_min_inter", str(args.yelp_min_inter)]
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
        title="Test metrics — mean ± std across seeds, Yelp (full-rank)")
    agg.plot_val_curves_with_band(
        by_model_curves, compare_dir / "val_NDCG@10_curves.png",
        title="Validation NDCG@10 — mean ± 1σ across seeds, Yelp")

    summary = agg.format_aggregate_summary(by_model_agg, metric_keys)
    summary += "\n" + agg.format_per_seed_table(by_model_per_seed, "NDCG@10")
    summary += "\n" + agg.format_per_seed_table(by_model_per_seed, "HR@10")
    print("\n" + "=" * 72)
    print("Test metrics comparison — mean ± std across seeds (Yelp)")
    print("=" * 72)
    print(summary)
    with open(compare_dir / "summary.txt", "w") as f:
        f.write(summary + "\n")

    print(f"\nComparison artifacts written to: {compare_dir}")

    if args.also_make_overview:
        print("\nGenerating results_overview.md ...")
        proc = subprocess.run(
            [sys.executable, "make_overview.py", "--dataset", "yelp",
             "--artifacts_root", "artifacts_yelp"])
        if proc.returncode != 0:
            print(f"[warn] make_overview.py exited with code {proc.returncode}")


if __name__ == "__main__":
    main()
