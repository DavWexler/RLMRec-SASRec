"""Single entrypoint: train all three models on ML-1M, Yelp, and
Amazon-Books, then generate the markdown overview for each.

What this script does, in order (per dataset):
  1. Invoke `run_all.py` (ML-1M), `run_all_yelp.py` (Yelp), or
     `run_all_amazon.py` (Amazon-Books), which itself:
       - trains SASRec, RLMRec+LightGCN, RLMRec+SASRec sequentially
       - aggregates test metrics into <root>/comparison/{comparison.csv,
         comparison_bars.png, val_NDCG@10_curves.png, summary.txt}
  2. Invoke `make_overview.py --dataset {ml1m,yelp,amazon}` to write
     <root>/comparison/results_overview.md.

Common flags (--epochs, --eval_every, --seed) are forwarded to every
training subprocess. Yelp- and Amazon-specific filter flags only get
forwarded to the matching dataset's run.

Examples:
    # Full pipeline on all datasets (warning: many hours on a Mac;
    # Amazon-Books also has a multi-GiB first-time download)
    python run_everything.py

    # Only ML-1M
    python run_everything.py --datasets ml1m

    # All, but skip retraining — just regenerate comparison + overviews
    python run_everything.py --compare_only

    # Yelp + Amazon, smaller Yelp filter, skip RLMRec on ML-1M to save time
    python run_everything.py --datasets yelp amazon \\
        --yelp_start_year 2019 --yelp_min_inter 20 \\
        --skip_models_ml rlmrec_lightgcn rlmrec_sasrec

Artifacts land under:
    artifacts/         (ML-1M)         ← run_all.py + make_overview.py --dataset ml1m
    artifacts_yelp/    (Yelp)          ← run_all_yelp.py + make_overview.py --dataset yelp
    artifacts_amazon/  (Amazon-Books)  ← run_all_amazon.py + make_overview.py --dataset amazon
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ALL_MODELS = ["sasrec", "rlmrec_lightgcn", "rlmrec_sasrec"]
ALL_DATASETS = ["ml1m", "yelp", "amazon"]

RUNNER_BY_DATASET = {
    "ml1m": "run_all.py",
    "yelp": "run_all_yelp.py",
    "amazon": "run_all_amazon.py",
}

ARTIFACTS_ROOT_BY_DATASET = {
    "ml1m": "artifacts",
    "yelp": "artifacts_yelp",
    "amazon": "artifacts_amazon",
}


def _run(cmd: list[str], *, label: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n[{label}] >>> {' '.join(cmd)}\n{bar}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    dt = time.time() - t0
    if rc != 0:
        raise SystemExit(
            f"[{label}] FAILED (exit {rc}) after {dt:.1f}s — see stderr above.")
    print(f"[{label}] OK in {dt:.1f}s", flush=True)


def _runner_cmd(dataset: str, args) -> list[str]:
    """Build the run_all*.py invocation for one dataset.

    Flags that the runner itself parses (--epochs, --eval_every, --seed, plus
    Yelp-only --yelp_start_year etc., Amazon-only --amazon_start_year etc.)
    go directly. Flags only the underlying trainers know about (--model_name)
    get appended via the `-- ...` REMAINDER tail so the runner forwards them
    verbatim.
    """
    runner = RUNNER_BY_DATASET[dataset]
    cmd = [sys.executable, runner]

    skip = {"yelp": args.skip_models_yelp, "ml1m": args.skip_models_ml,
            "amazon": args.skip_models_amazon}.get(dataset)
    if skip is None:
        skip = args.skip_models  # global fallback
    if skip:
        cmd += ["--skip", *skip]

    if args.compare_only:
        cmd += ["--compare_only"]

    # Runner-known training knobs (all run_all*.py share these).
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.eval_every is not None:
        cmd += ["--eval_every", str(args.eval_every)]
    if args.seeds is not None:
        cmd += ["--seeds", *[str(s) for s in args.seeds]]
    elif args.seed is not None:
        cmd += ["--seed", str(args.seed)]

    # Yelp-only filter knobs.
    if dataset == "yelp":
        if args.yelp_start_year is not None:
            cmd += ["--yelp_start_year", str(args.yelp_start_year)]
        if args.yelp_min_inter is not None:
            cmd += ["--yelp_min_inter", str(args.yelp_min_inter)]

    # Amazon-only filter knobs.
    if dataset == "amazon":
        if args.amazon_start_year is not None:
            cmd += ["--amazon_start_year", str(args.amazon_start_year)]
        if args.amazon_min_inter is not None:
            cmd += ["--amazon_min_inter", str(args.amazon_min_inter)]
        if args.amazon_category is not None:
            cmd += ["--amazon_category", args.amazon_category]

    # Trainer-only knobs (unknown to the runner) — forward via REMAINDER.
    tail: list[str] = []
    if args.model_name is not None:
        tail += ["--model_name", args.model_name]
    if tail:
        cmd += ["--", *tail]
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", choices=ALL_DATASETS,
                   default=ALL_DATASETS,
                   help="Which datasets to run (default: both).")
    p.add_argument("--skip_models", nargs="*", choices=ALL_MODELS, default=[],
                   help="Models to skip on EVERY dataset.")
    p.add_argument("--skip_models_ml", nargs="*", choices=ALL_MODELS, default=None,
                   help="Models to skip on ML-1M only (overrides --skip_models).")
    p.add_argument("--skip_models_yelp", nargs="*", choices=ALL_MODELS, default=None,
                   help="Models to skip on Yelp only (overrides --skip_models).")
    p.add_argument("--skip_models_amazon", nargs="*", choices=ALL_MODELS, default=None,
                   help="Models to skip on Amazon only (overrides --skip_models).")
    p.add_argument("--compare_only", action="store_true",
                   help="Skip training; just rebuild comparison artifacts and overviews.")
    p.add_argument("--no_overview", action="store_true",
                   help="Skip the markdown overview generation step.")

    # Forwarded to every training subprocess.
    p.add_argument("--epochs", type=int, default=None,
                   help="Cap epochs for every training run.")
    p.add_argument("--eval_every", type=int, default=None,
                   help="Validation cadence for every training run.")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Seeds for every training run. "
                        "Default of the runners is [42, 43, 44, 45, 46].")
    p.add_argument("--seed", type=int, default=None,
                   help="Single-seed legacy alias for --seeds <N>.")
    p.add_argument("--model_name", type=str, default=None,
                   help="Sentence-transformers / HF encoder model name "
                        "(only affects RLMRec runs).")

    # Yelp-only filters.
    p.add_argument("--yelp_start_year", type=int, default=None,
                   help="Drop Yelp reviews before YYYY-01-01 (default: trainer default).")
    p.add_argument("--yelp_min_inter", type=int, default=None,
                   help="K for k-core filter on Yelp (default: trainer default).")

    # Amazon-only filters.
    p.add_argument("--amazon_start_year", type=int, default=None,
                   help="Drop Amazon reviews before YYYY-01-01 (default: trainer default).")
    p.add_argument("--amazon_min_inter", type=int, default=None,
                   help="K for k-core filter on Amazon (default: trainer default).")
    p.add_argument("--amazon_category", type=str, default=None,
                   help="Amazon Reviews 2023 category (default: Books).")

    args = p.parse_args()

    failures: list[str] = []

    for ds in args.datasets:
        # 1. Train + aggregate.
        try:
            _run(_runner_cmd(ds, args), label=f"{ds}/train+aggregate")
        except SystemExit as e:
            failures.append(f"{ds}/train+aggregate: {e}")
            print(f"[{ds}] training/aggregation failed; skipping overview for "
                  f"this dataset.", flush=True)
            continue

        # 2. Overview.
        if args.no_overview:
            continue
        try:
            _run([sys.executable, "make_overview.py", "--dataset", ds],
                 label=f"{ds}/overview")
        except SystemExit as e:
            failures.append(f"{ds}/overview: {e}")

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    for ds in args.datasets:
        root = Path(ARTIFACTS_ROOT_BY_DATASET[ds])
        cmp_csv = root / "comparison" / "comparison.csv"
        overview = root / "comparison" / "results_overview.md"
        print(f"  [{ds}]")
        print(f"    comparison: {cmp_csv} "
              f"({'present' if cmp_csv.exists() else 'MISSING'})")
        print(f"    overview:   {overview} "
              f"({'present' if overview.exists() else 'MISSING'})")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
