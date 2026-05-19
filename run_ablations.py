"""Ablation study orchestrator for RLMRec+SASRec.

For a paper-submission ablation grid: trains one variant at a time, each one
flipping a single knob from the paper-faithful RLMRec+SASRec config (full
losses on at lambda_u=lambda_i=0.1, tau=0.1, MLP projection heads, BGE-large
semantic encoder). Variants run on every dataset (ML-1M, Yelp, Amazon-Books)
across 3 seeds for 50 epochs by default.

Variants
--------
RLMRec+SASRec (trainer: rlmrec_sasrec_train.py):
  sasrec_full              paper config; in-ablation reference for paired t-tests
  sasrec_no_user_align     lambda_user = 0
  sasrec_no_item_align     lambda_item = 0
  sasrec_no_align          lambda_user = lambda_item = 0  (~ pure SASRec at 50 ep)
  sasrec_lambda_{0.01,0.05,0.5,1.0}  symmetric lambda sweep
  sasrec_temp_{0.05,0.2,0.5}         InfoNCE temperature sweep
  sasrec_proj_linear       single nn.Linear projection heads (vs default MLP)
  sasrec_encoder_minilm    sentence-transformers/all-MiniLM-L6-v2 (384-d, 22M)

RLMRec+LightGCN (trainer: rlmrec_lightgcn_train.py):
  lightgcn_full            paper RLMRec config at 50ep (in-ablation reference)
  lightgcn_encoder_minilm  MiniLM-L6 encoder

Optional (--include_e5_mistral): adds E5-Mistral-7B (4096-d, 7B params)
encoder variants for both backbones. Heavy first-time encode.

Per-variant outputs land in:
  artifacts_ablation/<dataset>/<variant>/seed_<N>/...   (per-seed, written by trainer)
  artifacts_ablation/<dataset>/<variant>/test_metrics_per_seed.csv
  artifacts_ablation/<dataset>/<variant>/test_metrics_aggregate.csv
  artifacts_ablation/<dataset>/<variant>/test_metrics.csv  (mean only, legacy)

Per-dataset cross-variant outputs land in artifacts_ablation/<dataset>/comparison/:
  ablation.csv                metric x variant grid (mean only)
  ablation_aggregate.csv      metric x variant (mean + std + n)
  ablation_per_seed.csv       long format
  ablation_significance.csv   paired t-tests vs in-ablation reference + markers
  ablation_bars.png           horizontal bars w/ error bars, HR@10 + NDCG@10
  ablation_overview.md        paper-style markdown table with significance

The pre-trained baselines under artifacts*/ (the 75-epoch, 5-seed runs from
run_everything.py) are also read in and reported as reference rows in the
markdown, but they are NOT used in paired t-tests because the epoch budget
differs from the ablation runs. The fair statistical comparison is each
ablation variant vs its matched in-ablation reference (`sasrec_full` for
SASRec variants; `lightgcn_full` for LightGCN variants), which uses the same
seeds and epoch budget.

Examples
--------
Full ablation grid on every dataset:
  python run_ablations.py

Smoke test (one variant, one seed, few epochs):
  python run_ablations.py --datasets ml1m --variants sasrec_no_user_align \\
      --seeds 42 --epochs 5

Skip training; rebuild aggregates/markdown from existing per-seed CSVs:
  python run_ablations.py --aggregate_only

Add E5-Mistral-7B encoder variants (heavy):
  python run_ablations.py --include_e5_mistral
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seed_aggregation as agg

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABLATION_ROOT = Path("artifacts_ablation")
SASREC_TRAINER = "rlmrec_sasrec_train.py"
LIGHTGCN_TRAINER = "rlmrec_lightgcn_train.py"

DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_EPOCHS = 50
DEFAULT_EVAL_EVERY = 10  # finer cadence so val curves are usable at 50 ep

DATASETS = ["ml1m", "yelp", "amazon"]
DATASET_LABELS = {"ml1m": "MovieLens-1M", "yelp": "Yelp",
                  "amazon": "Amazon-Books"}

# Pre-trained baselines (5 seeds x 75 epochs; produced by run_everything.py).
# Reported as reference rows; NOT used in paired t-tests due to epoch mismatch.
EXISTING_BASELINE_ROOT = {
    "ml1m": Path("artifacts"),
    "yelp": Path("artifacts_yelp"),
    "amazon": Path("artifacts_amazon"),
}
EXISTING_BASELINE_SEEDS = [42, 43, 44, 45, 46]
EXISTING_BASELINES = [
    ("sasrec",          "SASRec (75ep, paper-faithful)"),
    ("rlmrec_lightgcn", "RLMRec+LightGCN (75ep, paper RLMRec)"),
    ("rlmrec_sasrec",   "RLMRec+SASRec (75ep, our model)"),
]

# Two-sided critical t-values, hard-coded so we can compute significance
# markers without scipy. Used only when scipy is unavailable.
T_CRIT_TWO_SIDED = {
    # df: {alpha: t_crit}
    1: {0.05: 12.706, 0.01: 63.657, 0.001: 636.619},
    2: {0.05: 4.303,  0.01: 9.925,  0.001: 31.598},
    3: {0.05: 3.182,  0.01: 5.841,  0.001: 12.924},
    4: {0.05: 2.776,  0.01: 4.604,  0.001: 8.610},
    5: {0.05: 2.571,  0.01: 4.032,  0.001: 6.869},
}

# Headline metrics for the paper-style markdown table. Other metrics still
# get included in the CSVs.
HEADLINE_METRICS = ["HR@5", "HR@10", "NDCG@5", "NDCG@10", "MRR"]


# ---------------------------------------------------------------------------
# Variant grid
# ---------------------------------------------------------------------------

def _variant(name: str, trainer: str, label: str, args: list[str],
             reference: str | None) -> dict:
    return {"name": name, "trainer": trainer, "label": label,
            "args": list(args), "reference": reference}


VARIANTS_CORE: list[dict] = [
    # SASRec ablations -----------------------------------------------------
    _variant("sasrec_full", SASREC_TRAINER,
             "RLMRec+SASRec (full)", [], reference=None),
    _variant("sasrec_no_user_align", SASREC_TRAINER,
             "no user-align (lambda_u=0)",
             ["--lambda_user", "0"], reference="sasrec_full"),
    _variant("sasrec_no_item_align", SASREC_TRAINER,
             "no item-align (lambda_i=0)",
             ["--lambda_item", "0"], reference="sasrec_full"),
    _variant("sasrec_no_align", SASREC_TRAINER,
             "no align (~ pure SASRec)",
             ["--lambda_user", "0", "--lambda_item", "0"],
             reference="sasrec_full"),
    _variant("sasrec_lambda_0.01", SASREC_TRAINER,
             "lambda = 0.01",
             ["--lambda_user", "0.01", "--lambda_item", "0.01"],
             reference="sasrec_full"),
    _variant("sasrec_lambda_0.05", SASREC_TRAINER,
             "lambda = 0.05",
             ["--lambda_user", "0.05", "--lambda_item", "0.05"],
             reference="sasrec_full"),
    _variant("sasrec_lambda_0.5", SASREC_TRAINER,
             "lambda = 0.5",
             ["--lambda_user", "0.5", "--lambda_item", "0.5"],
             reference="sasrec_full"),
    _variant("sasrec_lambda_1.0", SASREC_TRAINER,
             "lambda = 1.0",
             ["--lambda_user", "1.0", "--lambda_item", "1.0"],
             reference="sasrec_full"),
    _variant("sasrec_temp_0.05", SASREC_TRAINER,
             "tau = 0.05", ["--temperature", "0.05"],
             reference="sasrec_full"),
    _variant("sasrec_temp_0.2", SASREC_TRAINER,
             "tau = 0.2", ["--temperature", "0.2"],
             reference="sasrec_full"),
    _variant("sasrec_temp_0.5", SASREC_TRAINER,
             "tau = 0.5", ["--temperature", "0.5"],
             reference="sasrec_full"),
    _variant("sasrec_proj_linear", SASREC_TRAINER,
             "linear proj-head", ["--proj_head", "linear"],
             reference="sasrec_full"),
    _variant("sasrec_encoder_minilm", SASREC_TRAINER,
             "MiniLM-L6 encoder",
             ["--model_name", "sentence-transformers/all-MiniLM-L6-v2"],
             reference="sasrec_full"),
    # LightGCN ablations ---------------------------------------------------
    _variant("lightgcn_full", LIGHTGCN_TRAINER,
             "RLMRec+LightGCN (full)", [], reference=None),
    _variant("lightgcn_encoder_minilm", LIGHTGCN_TRAINER,
             "RLMRec+LightGCN MiniLM-L6",
             ["--model_name", "sentence-transformers/all-MiniLM-L6-v2"],
             reference="lightgcn_full"),
]

VARIANTS_E5 = [
    _variant("sasrec_encoder_e5", SASREC_TRAINER,
             "E5-Mistral-7B encoder",
             ["--model_name", "intfloat/e5-mistral-7b-instruct"],
             reference="sasrec_full"),
    _variant("lightgcn_encoder_e5", LIGHTGCN_TRAINER,
             "RLMRec+LightGCN E5-Mistral-7B",
             ["--model_name", "intfloat/e5-mistral-7b-instruct"],
             reference="lightgcn_full"),
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def variant_dir(dataset: str, variant_name: str) -> Path:
    return ABLATION_ROOT / dataset / variant_name


def variant_seed_dir(dataset: str, variant_name: str, seed: int) -> Path:
    return variant_dir(dataset, variant_name) / f"seed_{seed}"


def comparison_dir(dataset: str) -> Path:
    return ABLATION_ROOT / dataset / "comparison"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def paired_ttest(treatment: list[float], reference: list[float]
                 ) -> tuple[float, int, float]:
    """Two-sided paired t-test on (treatment - reference).

    Returns (t-statistic, df, p-value). p-value is NaN if scipy is not
    installed (use significance_marker instead).
    """
    if len(treatment) != len(reference):
        raise ValueError("paired t-test requires same-length samples")
    n = len(treatment)
    if n < 2:
        return float("nan"), 0, float("nan")
    diffs = [t - r for t, r in zip(treatment, reference)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    sem = math.sqrt(var_d / n)
    df = n - 1
    if sem == 0:
        # Degenerate: all diffs identical. Either no effect or perfectly
        # constant separation. Report inf t when there is a non-zero mean
        # diff (perfect separation), else 0.
        t = 0.0 if mean_d == 0 else math.copysign(float("inf"), mean_d)
    else:
        t = mean_d / sem
    if HAS_SCIPY and not math.isinf(t):
        # scipy_stats.ttest_rel accepts the raw arrays.
        p = float(scipy_stats.ttest_rel(treatment, reference).pvalue)
    elif math.isinf(t):
        p = 0.0
    else:
        p = float("nan")
    return t, df, p


def significance_marker(t: float, df: int, p: float) -> str:
    """Return '*'/'**'/'***'/'' based on two-sided significance.

    Uses exact p-value when scipy supplied one; otherwise consults a hard-
    coded critical-value table for df in {1..5}.
    """
    if math.isnan(t):
        return ""
    if not math.isnan(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""
    if df not in T_CRIT_TWO_SIDED:
        return ""
    crit = T_CRIT_TWO_SIDED[df]
    abs_t = abs(t)
    if abs_t >= crit[0.001]:
        return "***"
    if abs_t >= crit[0.01]:
        return "**"
    if abs_t >= crit[0.05]:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_one_training(variant: dict, dataset: str, seed: int,
                     epochs: int, eval_every: int,
                     extra_args: list[str]) -> None:
    seed_dir = variant_seed_dir(dataset, variant["name"], seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, variant["trainer"],
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--eval_every", str(eval_every),
        "--seed", str(seed),
        "--out_dir", str(seed_dir),
        *variant["args"],
        *extra_args,
    ]
    bar = "=" * 78
    print(f"\n{bar}\n>>> [{dataset}/{variant['name']}/seed_{seed}] "
          f"{' '.join(cmd)}\n{bar}", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{variant['trainer']} ({dataset}/{variant['name']}/seed_{seed}) "
            f"exited with code {proc.returncode}")


def consolidate_variant(dataset: str, variant: dict, seeds: list[int]
                        ) -> tuple[dict[int, dict[str, float]],
                                   dict[str, dict[str, float]]]:
    vdir = variant_dir(dataset, variant["name"])
    per_seed = agg.read_seed_test_metrics(vdir, seeds)
    if not per_seed:
        return per_seed, {}
    aggregate = agg.aggregate_metrics(per_seed)
    agg.write_per_seed_metrics_csv(per_seed, vdir / "test_metrics_per_seed.csv")
    agg.write_aggregate_metrics_csv(aggregate, vdir / "test_metrics_aggregate.csv")
    agg.write_mean_metrics_csv(aggregate, vdir / "test_metrics.csv")
    for s in seeds:
        meta_src = agg.seed_subdir(vdir, s) / "run_metadata.json"
        if meta_src.exists():
            shutil.copy2(meta_src, vdir / "run_metadata.json")
            break
    return per_seed, aggregate


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def read_existing_baselines(dataset: str
                            ) -> dict[str, dict[str, dict[str, float]]]:
    """Returns {label: {"per_seed": {seed: {metric: val}}, "agg": {...}}}.

    Empty dict if the artifacts directory does not exist.
    """
    root = EXISTING_BASELINE_ROOT[dataset]
    out: dict[str, dict] = {}
    if not root.exists():
        return out
    for model_name, label in EXISTING_BASELINES:
        per_seed = agg.read_seed_test_metrics(root / model_name,
                                              EXISTING_BASELINE_SEEDS)
        if not per_seed:
            continue
        out[label] = {
            "per_seed": per_seed,
            "agg": agg.aggregate_metrics(per_seed),
        }
    return out


def write_ablation_csvs(dataset: str,
                        per_variant: dict[str, dict[int, dict[str, float]]],
                        agg_variant: dict[str, dict[str, dict[str, float]]],
                        ) -> list[str]:
    cdir = comparison_dir(dataset)
    cdir.mkdir(parents=True, exist_ok=True)

    metric_keys = sorted({mk for d in agg_variant.values() for mk in d})
    variant_names = list(agg_variant.keys())

    # 1. Long-format per-seed CSV
    with open(cdir / "ablation_per_seed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "seed", "metric", "value"])
        for vn in variant_names:
            for s in sorted(per_variant.get(vn, {}).keys()):
                for mk in sorted(per_variant[vn][s].keys()):
                    w.writerow([vn, s, mk, f"{per_variant[vn][s][mk]:.6f}"])

    # 2. Mean-only CSV (rows = metric, cols = variants)
    with open(cdir / "ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + variant_names)
        for mk in metric_keys:
            row = [mk] + [
                f"{agg_variant[vn].get(mk, {}).get('mean', float('nan')):.6f}"
                for vn in variant_names
            ]
            w.writerow(row)

    # 3. Aggregate CSV (rows = metric, cols = mean/std/n per variant)
    with open(cdir / "ablation_aggregate.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["metric"]
        for vn in variant_names:
            header += [f"{vn} mean", f"{vn} std", f"{vn} n"]
        w.writerow(header)
        for mk in metric_keys:
            row: list[str] = [mk]
            for vn in variant_names:
                a = agg_variant[vn].get(mk)
                if a:
                    row += [f"{a['mean']:.6f}", f"{a['std']:.6f}",
                            f"{int(a['n'])}"]
                else:
                    row += ["", "", ""]
            w.writerow(row)

    return metric_keys


def write_significance_csv(dataset: str, variants_in_run: list[dict],
                           per_variant: dict[str, dict[int, dict[str, float]]]
                           ) -> dict[tuple[str, str], dict]:
    """Compute paired t-tests of each variant vs its in-ablation reference.

    Returns {(variant_name, metric): {"t":, "df":, "p":, "marker":,
                                       "delta":, "delta_pct":}}.
    Also writes ablation_significance.csv.
    """
    sig: dict[tuple[str, str], dict] = {}
    rows: list[dict] = []
    for v in variants_in_run:
        ref_name = v.get("reference")
        if not ref_name:
            continue
        if ref_name not in per_variant or v["name"] not in per_variant:
            continue
        treat = per_variant[v["name"]]
        ref = per_variant[ref_name]
        common_seeds = sorted(set(treat.keys()) & set(ref.keys()))
        if len(common_seeds) < 2:
            continue
        common_metrics = sorted(
            set.intersection(*[set(treat[s].keys()) for s in common_seeds],
                             *[set(ref[s].keys())   for s in common_seeds]))
        for mk in common_metrics:
            t_vals = [treat[s][mk] for s in common_seeds]
            r_vals = [ref[s][mk]   for s in common_seeds]
            t, df, p = paired_ttest(t_vals, r_vals)
            marker = significance_marker(t, df, p)
            mean_t = statistics.fmean(t_vals)
            mean_r = statistics.fmean(r_vals)
            delta = mean_t - mean_r
            delta_pct = (delta / mean_r * 100.0) if mean_r != 0 else float("nan")
            sig[(v["name"], mk)] = {
                "t": t, "df": df, "p": p, "marker": marker,
                "delta": delta, "delta_pct": delta_pct,
                "n_pairs": len(common_seeds),
            }
            rows.append({
                "variant": v["name"], "reference": ref_name, "metric": mk,
                "n_pairs": len(common_seeds),
                "treatment_mean": f"{mean_t:.6f}",
                "reference_mean": f"{mean_r:.6f}",
                "delta": f"{delta:+.6f}",
                "delta_pct": f"{delta_pct:+.3f}",
                "t": f"{t:.4f}",
                "df": df,
                "p": "" if math.isnan(p) else f"{p:.6f}",
                "marker": marker,
            })
    if not rows:
        return sig
    cdir = comparison_dir(dataset)
    cdir.mkdir(parents=True, exist_ok=True)
    with open(cdir / "ablation_significance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return sig


def plot_ablation_bars(dataset: str, variants_in_run: list[dict],
                       agg_variant: dict[str, dict[str, dict[str, float]]],
                       ) -> None:
    """Side-by-side horizontal bar charts for HR@10 and NDCG@10."""
    metrics = ["HR@10", "NDCG@10"]
    names = [v["name"] for v in variants_in_run if v["name"] in agg_variant]
    if not names:
        return
    labels = [v["label"] for v in variants_in_run if v["name"] in agg_variant]
    references = {v["name"] for v in variants_in_run if v.get("reference") is None}

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(13, max(5, 0.35 * len(names) + 1.5)),
                             sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    y = list(range(len(names)))
    for ax, mk in zip(axes, metrics):
        means = [agg_variant[n].get(mk, {}).get("mean", 0.0) for n in names]
        stds = [agg_variant[n].get(mk, {}).get("std", 0.0) for n in names]
        colors = ["tab:orange" if n in references else "tab:blue" for n in names]
        ax.barh(y, means, xerr=stds, color=colors, capsize=3, alpha=0.85)
        for yi, (m, s) in enumerate(zip(means, stds)):
            ax.text(m + s, yi, f"  {m:.4f}", va="center", fontsize=7)
        ax.set_xlabel(mk)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    fig.suptitle(f"Ablation grid - {DATASET_LABELS.get(dataset, dataset)} "
                 f"(mean +/- std across seeds; orange = in-ablation reference)")
    fig.tight_layout()
    fig.savefig(comparison_dir(dataset) / "ablation_bars.png", dpi=140)
    plt.close(fig)


def _format_cell(agg_dict: dict[str, dict[str, float]] | None, mk: str,
                 marker: str | None = None) -> str:
    if not agg_dict or mk not in agg_dict:
        return "-"
    a = agg_dict[mk]
    cell = f"{a['mean']:.4f} +/- {a['std']:.4f}"
    if marker:
        cell += f" {marker}"
    return cell


def write_markdown(dataset: str, variants_in_run: list[dict],
                   agg_variant: dict[str, dict[str, dict[str, float]]],
                   sig: dict[tuple[str, str], dict],
                   existing: dict[str, dict[str, dict[str, float]]],
                   epochs: int, seeds: list[int]) -> None:
    cdir = comparison_dir(dataset)
    cdir.mkdir(parents=True, exist_ok=True)
    label = DATASET_LABELS.get(dataset, dataset)

    sasrec_variants = [v for v in variants_in_run
                       if v["trainer"] == SASREC_TRAINER and v["name"] in agg_variant]
    lightgcn_variants = [v for v in variants_in_run
                         if v["trainer"] == LIGHTGCN_TRAINER and v["name"] in agg_variant]

    metrics = [m for m in HEADLINE_METRICS
               if any(m in agg_variant.get(v["name"], {}) for v in variants_in_run)]

    n_seeds = len(seeds)
    df = max(n_seeds - 1, 0)
    sig_method = ("scipy.stats.ttest_rel exact p-values"
                  if HAS_SCIPY else
                  f"hard-coded two-sided critical t-table for df={df}")

    lines: list[str] = []
    lines.append(f"# Ablation study - RLMRec+SASRec on {label}")
    lines.append("")
    lines.append(f"- {n_seeds} seeds (`{', '.join(str(s) for s in seeds)}`) "
                 f"x {epochs} epochs each")
    lines.append("- Full-rank leave-one-out evaluation "
                 "(`eval_full_rank.evaluate_full_rank`)")
    lines.append(f"- Significance markers: `*` p<0.05, `**` p<0.01, "
                 f"`***` p<0.001 (paired t-test vs in-ablation reference, "
                 f"{sig_method})")
    lines.append("- Reference variants are highlighted in **bold**; the "
                 "delta column shows mean(variant) - mean(reference).")
    lines.append("")

    def _section(title: str, vlist: list[dict], ref_name: str | None) -> None:
        if not vlist:
            return
        lines.append(f"## {title}")
        lines.append("")
        header = ["variant"] + metrics
        if ref_name:
            header += [f"delta NDCG@10 vs {ref_name}"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for v in vlist:
            cells: list[str] = []
            name = v["name"]
            display = f"**{v['label']}** (ref)" if ref_name and name == ref_name \
                else v["label"]
            cells.append(display)
            for mk in metrics:
                marker = sig.get((name, mk), {}).get("marker", "") if name != ref_name else ""
                cells.append(_format_cell(agg_variant.get(name), mk, marker))
            if ref_name:
                if name == ref_name:
                    cells.append("(reference)")
                else:
                    s = sig.get((name, "NDCG@10"))
                    if s:
                        cells.append(f"{s['delta']:+.4f} ({s['delta_pct']:+.2f}%)")
                    else:
                        cells.append("-")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Group SASRec variants by ablation theme so the table reads like a paper
    def _by_prefix(prefix_filter) -> list[dict]:
        return [v for v in sasrec_variants if prefix_filter(v["name"])]

    loss_block = _by_prefix(lambda n: n in {
        "sasrec_full", "sasrec_no_user_align", "sasrec_no_item_align",
        "sasrec_no_align"})
    lambda_block = _by_prefix(lambda n: n == "sasrec_full" or n.startswith("sasrec_lambda_"))
    temp_block = _by_prefix(lambda n: n == "sasrec_full" or n.startswith("sasrec_temp_"))
    arch_block = _by_prefix(lambda n: n in {
        "sasrec_full", "sasrec_proj_linear",
        "sasrec_encoder_minilm", "sasrec_encoder_e5"})

    _section("Loss-component ablation (RLMRec+SASRec)",
             loss_block, ref_name="sasrec_full" if loss_block else None)
    _section("Alignment-loss weight (lambda) sweep (RLMRec+SASRec)",
             lambda_block, ref_name="sasrec_full" if lambda_block else None)
    _section("InfoNCE temperature (tau) sweep (RLMRec+SASRec)",
             temp_block, ref_name="sasrec_full" if temp_block else None)
    _section("Encoder + projection-head ablation (RLMRec+SASRec)",
             arch_block, ref_name="sasrec_full" if arch_block else None)
    _section("RLMRec+LightGCN ablations",
             lightgcn_variants,
             ref_name="lightgcn_full" if lightgcn_variants else None)

    # Pre-trained baseline reference table -------------------------------
    if existing:
        lines.append("## Pre-trained baselines (75ep, 5 seeds)")
        lines.append("")
        lines.append("Reference numbers from `run_everything.py`'s 75-epoch / 5-seed runs. "
                     "These are NOT used in paired t-tests because the epoch budget differs "
                     "from the 50-epoch ablation runs above; for fair statistical "
                     "comparisons, use the in-ablation `sasrec_full` / `lightgcn_full` "
                     "references.")
        lines.append("")
        header = ["baseline"] + metrics
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for blabel, bdata in existing.items():
            cells = [blabel]
            for mk in metrics:
                cells.append(_format_cell(bdata["agg"], mk))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Methodological note -------------------------------------------------
    lines.append("## Notes")
    lines.append("")
    lines.append("- Each variant flips ONE knob from the paper config "
                 "(`lambda_user = lambda_item = 0.1`, `tau = 0.1`, MLP "
                 "projection heads, BGE-large-en-v1.5 encoder).")
    lines.append("- All trained on the same dataset preprocessing, with "
                 "identical max sequence length and SASRec hyperparameters.")
    lines.append("- The `no align` variant zeros both alignment losses, "
                 "reducing RLMRec+SASRec to the SASRec backbone alone "
                 "(at this epoch budget) - the gap between `no align` and "
                 "`full` isolates the contribution of RLMRec's contrastive "
                 "alignment on a sequential backbone.")
    lines.append("")

    with open(cdir / "ablation_overview.md", "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    all_variant_names = [v["name"] for v in VARIANTS_CORE + VARIANTS_E5]
    p.add_argument("--datasets", nargs="+", choices=DATASETS,
                   default=DATASETS,
                   help=f"Datasets to run (default: {DATASETS}).")
    p.add_argument("--variants", nargs="+", default=None,
                   choices=all_variant_names,
                   help="Restrict to a subset of variants (default: all). "
                        "Note: filtering out a reference variant means its "
                        "dependent variants get no paired t-test markers.")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=all_variant_names,
                   help="Variants to skip.")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                   help=f"Seeds per variant (default: {DEFAULT_SEEDS}).")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                   help=f"Epochs per training run (default: {DEFAULT_EPOCHS}).")
    p.add_argument("--eval_every", type=int, default=DEFAULT_EVAL_EVERY,
                   help=f"Validation cadence in epochs "
                        f"(default: {DEFAULT_EVAL_EVERY}).")
    p.add_argument("--include_e5_mistral", action="store_true",
                   help="Add E5-Mistral-7B encoder variants for both "
                        "RLMRec backbones (heavy first-time encode).")
    p.add_argument("--aggregate_only", action="store_true",
                   help="Skip training; rebuild aggregates and overview "
                        "from existing per-seed CSVs.")
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help="Extra args forwarded to every trainer subprocess "
                        "(e.g. `-- --batch_size 256`).")
    args = p.parse_args()

    seeds = list(dict.fromkeys(args.seeds))
    extra = [a for a in (args.extra or []) if a != "--"]

    catalog = list(VARIANTS_CORE)
    if args.include_e5_mistral:
        catalog += list(VARIANTS_E5)

    if args.variants:
        wanted = set(args.variants)
        # Always keep referenced reference variants in the run so paired
        # t-tests have something to compare against.
        for v in catalog:
            if v["name"] in wanted and v.get("reference"):
                wanted.add(v["reference"])
        variants_in_run = [v for v in catalog if v["name"] in wanted]
    else:
        variants_in_run = list(catalog)

    if args.skip:
        skip = set(args.skip)
        variants_in_run = [v for v in variants_in_run if v["name"] not in skip]

    if not variants_in_run:
        raise SystemExit("No variants selected; refusing to run.")

    print(f"Datasets:  {args.datasets}")
    print(f"Seeds:     {seeds}  ({len(seeds)} per variant)")
    print(f"Epochs:    {args.epochs} (eval every {args.eval_every})")
    print(f"Variants:  {[v['name'] for v in variants_in_run]} "
          f"({len(variants_in_run)} total)")
    print(f"Statistical tests: "
          f"{'scipy paired t-test' if HAS_SCIPY else 'paired t + critical-value table'}")
    print(f"Aggregate-only:    {args.aggregate_only}")

    failures: list[str] = []

    for ds in args.datasets:
        # 1. Train each (variant, seed)
        if not args.aggregate_only:
            for v in variants_in_run:
                for s in seeds:
                    try:
                        run_one_training(v, ds, s, args.epochs,
                                         args.eval_every, extra)
                    except RuntimeError as e:
                        failures.append(str(e))
                        print(f"[warn] training failed: {e}", flush=True)

        # 2. Per-variant aggregation
        per_variant: dict[str, dict[int, dict[str, float]]] = {}
        agg_variant: dict[str, dict[str, dict[str, float]]] = {}
        for v in variants_in_run:
            ps, agv = consolidate_variant(ds, v, seeds)
            if not agv:
                print(f"[warn] no per-seed test_metrics under "
                      f"{variant_dir(ds, v['name'])}, dropping from comparison")
                continue
            per_variant[v["name"]] = ps
            agg_variant[v["name"]] = agv

        if not agg_variant:
            print(f"[{ds}] no variants produced metrics; skipping comparison.")
            continue

        # 3. Cross-variant outputs
        write_ablation_csvs(ds, per_variant, agg_variant)
        sig = write_significance_csv(ds, variants_in_run, per_variant)
        plot_ablation_bars(ds, [v for v in variants_in_run
                                if v["name"] in agg_variant], agg_variant)
        existing = read_existing_baselines(ds)
        write_markdown(ds, variants_in_run, agg_variant, sig, existing,
                       args.epochs, seeds)

        cdir = comparison_dir(ds)
        print(f"\n[{ds}] wrote ablation artifacts to {cdir}/")
        for fname in ["ablation.csv", "ablation_aggregate.csv",
                      "ablation_per_seed.csv", "ablation_significance.csv",
                      "ablation_bars.png", "ablation_overview.md"]:
            present = (cdir / fname).exists()
            print(f"    {fname:<32} {'present' if present else 'MISSING'}")

    print("\n" + "=" * 78)
    print("Ablation run complete.")
    print("=" * 78)
    if failures:
        print(f"\n{len(failures)} training failure(s):")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
