"""Generate `<artifacts_root>/comparison/results_overview.md` for ML-1M or Yelp.

Single overview generator that handles both datasets. Dataset-specific
narrative (paper baselines, dataset stats, profile-text source) is selected
by the --dataset flag; the table-rendering pipeline is shared.

Usage:
    python make_overview.py --dataset ml1m            # writes artifacts/comparison/results_overview.md
    python make_overview.py --dataset yelp            # writes artifacts_yelp/comparison/results_overview.md
    python make_overview.py --dataset yelp --artifacts_root artifacts_yelp_v2

Run after `run_all.py` / `run_all_yelp.py` finish (or via run_everything.py).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MODELS = [
    {"name": "sasrec", "label": "SASRec",
     "ckpt": "sasrec_best.pt"},
    {"name": "rlmrec_sasrec", "label": "RLMRec+SASRec",
     "ckpt": "rlmrec_sasrec_best.pt"},
    {"name": "rlmrec_lightgcn", "label": "RLMRec+LightGCN",
     "ckpt": "rlmrec_lightgcn_best.pt"},
]

METRIC_ORDER = ["HR@1", "HR@5", "HR@10", "HR@20",
                "NDCG@1", "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]

# Paper baselines for the Yelp split — RLMRec paper Table 2 (LightGCN backbone).
RLMREC_PAPER_YELP = {
    "lightgcn_baseline":    {"Recall@20": 0.0653, "NDCG@20": 0.0540,
                             "Recall@40": 0.1078, "NDCG@40": 0.0666},
    "rlmrec_con_lightgcn":  {"Recall@20": 0.0721, "NDCG@20": 0.0594,
                             "Recall@40": 0.1184, "NDCG@40": 0.0731},
    "rlmrec_gen_lightgcn":  {"Recall@20": 0.0708, "NDCG@20": 0.0586,
                             "Recall@40": 0.1166, "NDCG@40": 0.0721},
}

# Paper baselines for the Amazon-Books split — RLMRec paper Table 2 (LightGCN
# backbone). Their preprocessing is heavier than ours (different start year
# and k-core), so the absolute numbers aren't directly comparable; the lift
# (RLMRec-Con over LightGCN) is the apples-to-apples signal.
RLMREC_PAPER_AMAZON = {
    "lightgcn_baseline":    {"Recall@20": 0.0807, "NDCG@20": 0.0612,
                             "Recall@40": 0.1264, "NDCG@40": 0.0741},
    "rlmrec_con_lightgcn":  {"Recall@20": 0.0867, "NDCG@20": 0.0670,
                             "Recall@40": 0.1351, "NDCG@40": 0.0807},
    "rlmrec_gen_lightgcn":  {"Recall@20": 0.0857, "NDCG@20": 0.0658,
                             "Recall@40": 0.1334, "NDCG@40": 0.0791},
}

# SASRec paper Table 3 (Kang & McAuley, ICDM 2018) on ML-1M — these are
# *sampled-100-negative* HR@10 / NDCG@10, NOT full-rank, so direct comparison
# to our full-rank LOO numbers is not apples-to-apples (the paper's task is
# strictly easier).
SASREC_PAPER_ML1M_SAMPLED = {"HR@10": 0.7022, "NDCG@10": 0.4534}

# Default config per dataset.
DATASET_CONFIG = {
    "ml1m": {
        "display_name": "MovieLens-1M",
        "artifacts_root": "artifacts",
        "item_word": "movies",
        "interaction_word": "ratings",
        "profile_source": (
            "movie titles + genres for items, and a templated history-summary "
            "(genre histogram, demographics) for users — see `rlmrec_data.py`"
        ),
    },
    "yelp": {
        "display_name": "Yelp",
        "artifacts_root": "artifacts_yelp",
        "item_word": "businesses",
        "interaction_word": "reviews",
        "profile_source": (
            "business metadata (name, categories, city, stars, review_count) "
            "for items, and a templated category-summary built from the user's "
            "review history for users — see `yelp_data.py`"
        ),
    },
    "amazon": {
        "display_name": "Amazon-Books",
        "artifacts_root": "artifacts_amazon",
        "item_word": "products",
        "interaction_word": "reviews",
        "profile_source": (
            "product metadata (title, store/author, multi-level categories, "
            "description, average rating) for items, and a templated "
            "category-summary built from the user's review history for "
            "users — see `amazon_data.py`"
        ),
    },
}


def _load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _load_metrics(path: Path) -> dict[str, float]:
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


def _load_aggregate(path: Path) -> dict[str, dict[str, float]]:
    """Read `test_metrics_aggregate.csv` (cols: metric, mean, std, n).
    Returns {} if the file is absent (single-seed runs)."""
    out: dict[str, dict[str, float]] = {}
    if not path.exists():
        return out
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            mk = row.get("metric")
            if not mk:
                continue
            try:
                out[mk] = {
                    "mean": float(row["mean"]),
                    "std": float(row["std"]),
                    "n": float(row["n"]),
                }
            except (ValueError, KeyError):
                continue
    return out


def _fmt(v: float | None, nd: int = 4) -> str:
    if v is None:
        return "—"
    return f"{v:.{nd}f}"


def _render_metric_table(
        per_model: dict[str, dict[str, float]],
        per_model_agg: dict[str, dict[str, dict[str, float]]] | None = None,
        ) -> str:
    """Render the per-model metric table.

    If `per_model_agg` is provided (multi-seed runs), each cell shows
    `mean ± std`; the best-per-row highlight uses the mean. Falls back to
    plain values when only single-seed numbers are available.
    """
    labels = list(per_model.keys())
    cell_w = 22 if per_model_agg else 14
    header = "| Metric    | " + " | ".join(f"{l:<{cell_w}}" for l in labels) + " |"
    sep = "|-----------|" + "|".join(["-" * (cell_w + 2)] * len(labels)) + "|"
    lines = [header, sep]
    for m in METRIC_ORDER:
        means = [per_model[l].get(m) for l in labels]
        best_v = max((v for v in means if v is not None), default=None)
        cells = []
        for label, mean in zip(labels, means):
            if mean is None:
                cells.append(f"{'—':<{cell_w}}")
                continue
            agg_cell = (per_model_agg or {}).get(label, {}).get(m)
            if agg_cell is not None:
                s = f"{agg_cell['mean']:.4f} ± {agg_cell['std']:.4f}"
            else:
                s = _fmt(mean)
            if best_v is not None and mean == best_v:
                s = f"**{s}**"
            cells.append(f"{s:<{cell_w}}")
        lines.append(f"| {m:<9} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_config_table(metas: dict[str, dict]) -> str:
    rows = [
        ("Hidden / dim",     ["hidden", "hidden", "dim"]),
        ("Blocks / layers",  ["blocks", "blocks", "num_layers"]),
        ("Heads",            ["heads", "heads", None]),
        ("Max seq len",      ["max_len", "max_len", None]),
        ("Batch size",       ["batch_size", "batch_size", "batch_size"]),
        ("Dropout",          ["dropout", "dropout", None]),
        ("L2",               ["l2", "l2", "l2"]),
        ("LR",               ["lr", "lr", "lr"]),
        ("Epochs (cap)",     ["epochs", "epochs", "epochs"]),
    ]
    labels = ["SASRec", "RLMRec+SASRec", "RLMRec+LightGCN"]
    header = "| Field            | " + " | ".join(f"{l:>15}" for l in labels) + " |"
    sep = "|------------------|" + "|".join(["-" * 17] * len(labels)) + "|"
    out_lines = [header, sep]
    for label, keys in rows:
        cells = []
        for ml_label, key in zip(labels, keys):
            meta = metas.get(ml_label, {})
            args = meta.get("args", {}) if meta else {}
            v = args.get(key) if key else None
            cells.append(f"{(str(v) if v is not None else '—'):>15}")
        out_lines.append(f"| {label:<16} | " + " | ".join(cells) + " |")

    extra_rows: list[tuple[str, list[str]]] = []
    for fname in ("best_val_epoch", "best_val_NDCG@10", "model_params"):
        cells = []
        for ml_label in labels:
            meta = metas.get(ml_label, {})
            v = meta.get(fname) if meta else None
            if isinstance(v, float):
                cells.append(f"{v:>15.4f}")
            elif v is None:
                cells.append(f"{'—':>15}")
            else:
                cells.append(f"{v:>15,}" if isinstance(v, int)
                             and fname == "model_params" else f"{v:>15}")
        nice = {"best_val_epoch": "Best val epoch",
                "best_val_NDCG@10": "Best val NDCG@10",
                "model_params": "# params"}[fname]
        extra_rows.append((nice, cells))
    for label, cells in extra_rows:
        out_lines.append(f"| {label:<16} | " + " | ".join(cells) + " |")

    for label, key in [("λ_user / λ_item", "lambda_user"),
                       ("τ (InfoNCE)", "temperature")]:
        cells = [f"{'—':>15}"]
        for ml_label in ["RLMRec+SASRec", "RLMRec+LightGCN"]:
            meta = metas.get(ml_label, {})
            args = meta.get("args", {}) if meta else {}
            if label == "λ_user / λ_item":
                lu = args.get("lambda_user")
                li = args.get("lambda_item")
                if lu is not None and li is not None:
                    cells.append(f"{f'{lu} / {li}':>15}")
                else:
                    cells.append(f"{'—':>15}")
            else:
                v = args.get(key)
                cells.append(f"{(str(v) if v is not None else '—'):>15}")
        out_lines.append(f"| {label:<16} | " + " | ".join(cells) + " |")
    return "\n".join(out_lines)


def _build_runtime_facts(metas: dict[str, dict]) -> dict:
    facts = {}
    sasrec = metas.get("SASRec", {})
    facts["num_users"] = sasrec.get("num_users")
    facts["num_items"] = sasrec.get("num_items")
    facts["dataset_args"] = sasrec.get("args", {})
    enc_meta = metas.get("RLMRec+LightGCN") or metas.get("RLMRec+SASRec") or {}
    facts["semantic_model"] = enc_meta.get("semantic_model", "unknown")
    facts["semantic_dim"] = enc_meta.get("semantic_dim", "?")
    return facts


def _ratio(num: float, den: float) -> str:
    if den == 0:
        return "n/a"
    return f"{(num / den) * 100:.1f}%"


def _scale_paragraph(dataset: str, nu, ni, da: dict, cfg: dict) -> str:
    """Dataset-specific scale/preprocessing description for §1 header."""
    if dataset == "yelp":
        sy = da.get("yelp_start_year", 2018)
        mi = da.get("yelp_min_inter", 5)
        return (
            f"Dataset: **Yelp Open Dataset** (filtered: reviews from "
            f"{sy}-01-01 onward, iterative {mi}-core filter on (user, business) "
            f"interactions)\n"
            f"Resulting scale: **{nu:,} users**, **{ni:,} {cfg['item_word']}**"
        )
    if dataset == "amazon":
        sy = da.get("amazon_start_year", 2018)
        mi = da.get("amazon_min_inter", 5)
        cat = da.get("amazon_category", "Books")
        return (
            f"Dataset: **Amazon Reviews 2023 — {cat}** (filtered: reviews from "
            f"{sy}-01-01 onward, iterative {mi}-core filter on (user, "
            f"parent_asin) interactions)\n"
            f"Resulting scale: **{nu:,} users**, **{ni:,} {cfg['item_word']}**"
        )
    # ml1m
    return (
        f"Dataset: **MovieLens-1M** (standard 5-core preprocessing — users "
        f"and movies with at least 5 ratings)\n"
        f"Resulting scale: **{nu:,} users**, **{ni:,} {cfg['item_word']}**"
    )


def _paper_section(dataset: str, sasrec: dict, rls: dict, rll: dict,
                   sem_model: str, sem_dim, cfg: dict) -> str:
    """Dataset-specific §3 (paper comparison)."""
    if dataset == "amazon":
        ours_lgcn_ndcg20 = rll.get("NDCG@20")
        ours_lgcn_hr20 = rll.get("HR@20")
        paper_lgcn = RLMREC_PAPER_AMAZON["rlmrec_con_lightgcn"]
        paper_lgcn_baseline = RLMREC_PAPER_AMAZON["lightgcn_baseline"]

        s = """## 3. Comparison to the original papers

> ⚠️ **Direct numerical comparison is approximate** — see notes per model.

### SASRec (Kang & McAuley, ICDM 2018) on Amazon-Books

The original SASRec paper reports on Amazon Beauty and Amazon Games
(sampled-100-negative HR@10 / NDCG@10), but **not** on Amazon-Books. Later
full-rank LOO reproductions on Amazon-Books typically report HR@10 in the
**0.03–0.06** range and NDCG@10 in the **0.015–0.035** range, very
sensitive to start-year and k-core. Our numbers:
"""
        s += (
            f"  - HR@10 = {_fmt(sasrec.get('HR@10'))}\n"
            f"  - NDCG@10 = {_fmt(sasrec.get('NDCG@10'))}\n"
        )
        s += f"""
Architectural settings (2 blocks, 1 head, hidden=50, max_len=200, BCE loss)
are unchanged from the paper's configuration, so any deviation comes from
preprocessing (start year, k-core threshold) and the larger item catalog.

### RLMRec (Ren et al., WWW 2024) — RLMRec+LightGCN on Amazon-Books

RLMRec **does** evaluate on Amazon-Books, so we have a direct paper baseline
for this row. Their split uses LLM-generated user/item profiles with a
1536-d encoder (text-embedding-ada-002); ours uses templated text built
from product metadata (title, store/author, categories, description) +
{sem_model} ({sem_dim}-d). They report Recall@K and NDCG@K (LOO makes
Recall@K = HR@K when there's a single positive, which IS our case).

| Metric    | This project (RLMRec+LightGCN) | Paper: LightGCN baseline | Paper: RLMRec-Con on LightGCN |
|-----------|-------------------------------:|-------------------------:|------------------------------:|
"""
        s += (
            f"| NDCG@20   | {_fmt(ours_lgcn_ndcg20)} | {paper_lgcn_baseline['NDCG@20']:.4f} | {paper_lgcn['NDCG@20']:.4f} |\n"
            f"| HR@20 / Recall@20 | {_fmt(ours_lgcn_hr20)} | {paper_lgcn_baseline['Recall@20']:.4f} | {paper_lgcn['Recall@20']:.4f} |\n"
        )
        s += f"""
- The paper's RLMRec-Con on LightGCN improves NDCG@20 by roughly +9% over
  its own LightGCN baseline (0.0612 -> 0.0670) and Recall@20 by ~7%
  (0.0807 -> 0.0867) on Amazon-Books. Reaching the same *relative* lift
  here would require LLM-authored user/item profiles, which we approximate
  with product metadata only.
- The two structural deltas vs. the paper:
  1. **Profile quality**: paper uses LLM-generated user/item profiles from
     review text; we use product metadata (title, store/author, categories,
     description, average rating) and a templated user summary built from
     the user's review history.
  2. **Encoder dim**: paper uses 1536-d (text-embedding-ada-002); we use
     {sem_dim}-d ({sem_model}).
- Hyperparameters (λ=0.1, τ=0.1, dim=64, 3 LightGCN layers) match the
  paper's defaults, modulo our 64 vs. 32 embedding dim — a minor difference
  documented in the run_metadata.

### RLMRec+SASRec on Amazon-Books

This combination is **not in the original RLMRec paper** (the paper studies
CF-graph backbones: GCCF, LightGCN, SGL, SimGCL, DCCF, AutoCF). It's a
project-specific extension that wraps SASRec's user-state embedding and
item embedding with the same InfoNCE alignment used for the LightGCN
variant. No paper baseline exists.
"""
        return s

    if dataset == "yelp":
        ours_lgcn_ndcg20 = rll.get("NDCG@20")
        ours_lgcn_hr20 = rll.get("HR@20")
        paper_lgcn = RLMREC_PAPER_YELP["rlmrec_con_lightgcn"]
        paper_lgcn_baseline = RLMREC_PAPER_YELP["lightgcn_baseline"]

        s = """## 3. Comparison to the original papers

> ⚠️ **Direct numerical comparison is approximate** — see notes per model.

### SASRec (Kang & McAuley, ICDM 2018) on Yelp

The original SASRec paper does not include Yelp in its main benchmark
tables (it reports on Amazon Beauty/Games, Steam, and MovieLens-1M). Later
reproductions on Yelp under full-rank LOO typically report HR@10 in the
**0.04–0.08** range and NDCG@10 in the **0.02–0.04** range, depending on
the preprocessing. Our numbers:
"""
        s += (
            f"  - HR@10 = {_fmt(sasrec.get('HR@10'))}\n"
            f"  - NDCG@10 = {_fmt(sasrec.get('NDCG@10'))}\n"
        )
        s += f"""
Architectural settings (2 blocks, 1 head, hidden=50, max_len=200, BCE loss)
are unchanged from the paper's ML-1M configuration, so any deviation from
the literature on Yelp comes from preprocessing (start year, k-core
threshold) rather than the model itself.

### RLMRec (Ren et al., WWW 2024) — RLMRec+LightGCN on Yelp

RLMRec **does** evaluate on Yelp, so we have a direct paper baseline for
this row. Their Yelp split uses LLM-generated user/item profiles and a
1536-d encoder; ours uses business-metadata text + {sem_model} ({sem_dim}-d)
and per-user category-summary text. They report Recall@K and NDCG@K (LOO
makes Recall@K = HR@K when there's a single positive, which IS our case).

| Metric    | This project (RLMRec+LightGCN) | Paper: LightGCN baseline | Paper: RLMRec-Con on LightGCN |
|-----------|-------------------------------:|-------------------------:|------------------------------:|
"""
        s += (
            f"| NDCG@20   | {_fmt(ours_lgcn_ndcg20)} | {paper_lgcn_baseline['NDCG@20']:.4f} | {paper_lgcn['NDCG@20']:.4f} |\n"
            f"| HR@20 / Recall@20 | {_fmt(ours_lgcn_hr20)} | {paper_lgcn_baseline['Recall@20']:.4f} | {paper_lgcn['Recall@20']:.4f} |\n"
        )
        s += f"""
- The paper's RLMRec-Con on LightGCN improves NDCG@20 by roughly +10% over
  its own LightGCN baseline (0.0540 -> 0.0594) and Recall@20 by ~10%
  (0.0653 -> 0.0721) on Yelp. Reaching the same *relative* lift here would
  require LLM-authored user/item profiles, which we approximate with
  business metadata only.
- The two structural deltas vs. the paper:
  1. **Profile quality**: paper uses LLM-generated user/item profiles from
     review text; we use business metadata (name, categories, city, stars)
     and a templated user summary built from the user's review history.
  2. **Encoder dim**: paper uses 1536-d (text-embedding-ada-002); we use
     {sem_dim}-d ({sem_model}).
- Hyperparameters (λ=0.1, τ=0.1, dim=64, 3 LightGCN layers) match the
  paper's defaults, modulo our 64 vs. 32 embedding dim — a minor difference
  documented in the run_metadata.

### RLMRec+SASRec on Yelp

This combination is **not in the original RLMRec paper** (the paper studies
CF-graph backbones: GCCF, LightGCN, SGL, SimGCL, DCCF, AutoCF). It's a
project-specific extension that wraps SASRec's user-state embedding and
item embedding with the same InfoNCE alignment used for the LightGCN
variant. No paper baseline exists.
"""
        return s

    # ---- ml1m branch ----
    paper_hr10 = SASREC_PAPER_ML1M_SAMPLED["HR@10"]
    paper_ndcg10 = SASREC_PAPER_ML1M_SAMPLED["NDCG@10"]
    s = f"""## 3. Comparison to the original papers

> ⚠️ **Direct numerical comparison is approximate** — see notes per model.

### SASRec (Kang & McAuley, ICDM 2018) on MovieLens-1M

The SASRec paper reports **sampled-100-negative** HR@10 and NDCG@10 on
ML-1M (Table 3): HR@10 = {paper_hr10:.4f}, NDCG@10 = {paper_ndcg10:.4f}. Our
evaluator scores against the **full item catalog** (`eval_full_rank.py`),
which is a strictly harder task — every non-interacted movie is a candidate,
so every metric collapses to a smaller value than the sampled-100 setting.
Later full-rank ML-1M reproductions (RecBole, Petrov & Macdonald 2023)
typically report HR@10 in the **0.18–0.22** range and NDCG@10 in the
**0.10–0.13** range. Our numbers:
"""
    s += (
        f"  - HR@10 = {_fmt(sasrec.get('HR@10'))}\n"
        f"  - NDCG@10 = {_fmt(sasrec.get('NDCG@10'))}\n"
    )
    s += """
Bottom line: do **not** compare directly to the paper's headline numbers —
the protocols are different. Compare to the full-rank reproduction range,
or to the SASRec column in our own §1 table (which IS apples-to-apples
across our three models).

### RLMRec (Ren et al., WWW 2024) on MovieLens-1M

RLMRec **does not include ML-1M** in its benchmark suite (the paper
evaluates on Amazon-Books, Yelp, and Steam). There is no paper baseline
for either RLMRec+LightGCN or RLMRec+SASRec on this dataset. The most
useful comparison is therefore the within-project one in §1: does adding
the InfoNCE alignment loss to SASRec help, hurt, or no-op on ML-1M?

The Yelp overview (`artifacts_yelp/comparison/results_overview.md`) does
include a direct paper-baseline comparison for the LightGCN row, since
RLMRec's published Yelp numbers exist.

### RLMRec+SASRec on MovieLens-1M

Same caveat as above — not in the paper. The RLMRec authors only studied
CF-graph backbones (GCCF, LightGCN, SGL, SimGCL, DCCF, AutoCF). This row
is a project-specific extension: same InfoNCE alignment loss, but wrapped
around SASRec's last-position user-state embedding and the item embedding
table.
"""
    return s


def render_overview(dataset: str,
                    per_model: dict[str, dict[str, float]],
                    metas: dict[str, dict],
                    artifacts_root: Path,
                    per_model_agg: dict[str, dict[str, dict[str, float]]] | None = None,
                    seed_count: int = 1) -> str:
    cfg = DATASET_CONFIG[dataset]
    facts = _build_runtime_facts(metas)
    nu = facts["num_users"] or 0
    ni = facts["num_items"] or 0
    da = facts["dataset_args"]
    sem_model = facts.get("semantic_model", "unknown")
    sem_dim = facts.get("semantic_dim", "?")

    sasrec = per_model.get("SASRec", {})
    rls = per_model.get("RLMRec+SASRec", {})
    rll = per_model.get("RLMRec+LightGCN", {})

    def head_to_head(a: dict[str, float], b: dict[str, float]) -> dict[str, str]:
        out = {}
        for m in ("NDCG@10", "HR@10", "HR@20"):
            if a.get(m) and b.get(m):
                out[m] = _ratio(b[m], a[m])
        return out

    rls_vs_sas = head_to_head(sasrec, rls)
    rll_vs_sas = head_to_head(sasrec, rll)

    metric_table = _render_metric_table(
        {"SASRec": sasrec, "RLMRec+SASRec": rls, "RLMRec+LightGCN": rll},
        per_model_agg=per_model_agg,
    )
    config_table = _render_config_table(metas)
    scale_para = _scale_paragraph(dataset, nu, ni, da, cfg)
    paper_section = _paper_section(dataset, sasrec, rls, rll,
                                   sem_model, sem_dim, cfg)

    art = cfg["artifacts_root"]
    md = f"""# Recommender System Results — Overview ({cfg['display_name']})

{scale_para}
Evaluation: **full-rank leave-one-out** (every non-interacted {cfg['item_word'][:-1]}
is a candidate; no negative sampling)
Metrics: HR@K and NDCG@K for K ∈ {{1, 5, 10, 20}}, plus MRR

All three models share the same data splits and the same evaluator
(`eval_full_rank.py`), so the test numbers below are directly comparable.

Profile texts encoded into RLMRec's semantic side: {cfg['profile_source']},
encoded with **{sem_model}** ({sem_dim}-d).

---

## 1. Test-set results (this project)

{metric_table}

Bold = best per row. Source: `{art}/comparison/comparison.csv`.

### Run configuration

{config_table}

Sources: `{art}/{{sasrec,rlmrec_sasrec,rlmrec_lightgcn}}/run_metadata.json`.

---

## 2. Head-to-head observations

### SASRec vs. RLMRec+SASRec (semantic alignment regularizer added)
"""
    if rls_vs_sas:
        ndcg10 = rls_vs_sas.get("NDCG@10", "n/a")
        hr10 = rls_vs_sas.get("HR@10", "n/a")
        hr20 = rls_vs_sas.get("HR@20", "n/a")
        md += (
            f"- RLMRec+SASRec retains {ndcg10} of SASRec's NDCG@10 "
            f"({_fmt(rls.get('NDCG@10'))} vs. {_fmt(sasrec.get('NDCG@10'))}), "
            f"{hr10} of HR@10 ({_fmt(rls.get('HR@10'))} vs. {_fmt(sasrec.get('HR@10'))}), "
            f"and {hr20} of HR@20 ({_fmt(rls.get('HR@20'))} vs. {_fmt(sasrec.get('HR@20'))}).\n"
        )
    if dataset == "yelp":
        md += """- Compared with the ML-1M run (where the alignment loss caused a small
  regression), behaviour on Yelp is informative: the dataset is sparser and
  business categories carry more semantic signal than movie titles, so any
  benefit of the contrastive head should be more visible here.

### LightGCN-based vs. SASRec-based
"""
    elif dataset == "amazon":
        md += """- Amazon-Books has the richest per-item text in this project (full title,
  author, multi-level category hierarchy, multi-paragraph descriptions),
  so the contrastive head has the most to chew on here. If the alignment
  helps anywhere, this is the regime where the lift should be largest —
  the test of the RLMRec hypothesis, on a dataset much closer to the
  paper's own benchmark than ML-1M is.

### LightGCN-based vs. SASRec-based
"""
    else:
        md += """- The interesting question for ML-1M is whether the InfoNCE alignment
  loss adds signal on top of an already-strong sequential model. ML-1M
  has dense user histories (~165 ratings/user) and short item-text
  (just title + genres), so the contrastive head has limited new signal
  to inject. A near-tie or small regression here is the expected
  outcome; a clear *gain* would be surprising and worth investigating.

### LightGCN-based vs. SASRec-based
"""
    if rll_vs_sas:
        ndcg10 = rll_vs_sas.get("NDCG@10", "n/a")
        md += (
            f"- RLMRec+LightGCN reaches {ndcg10} of SASRec's NDCG@10 "
            f"({_fmt(rll.get('NDCG@10'))} vs. {_fmt(sasrec.get('NDCG@10'))}).\n"
        )
    if dataset == "yelp":
        md += """- LightGCN works from the user × business bipartite graph only; it has no
  notion of review timestamps, so on a "predict the next review" task its
  ceiling is intrinsically lower than a sequential model's. The gap is
  expected, but its size relative to ML-1M is one of the most useful signals
  this run produces.

### Take-away
Comparing the three models on Yelp tests the same hypothesis as ML-1M but in
a different regime: sparser interactions per user, more textual semantics
per item, and a longer tail of items. If the gaps narrow versus ML-1M, that
is evidence the semantic alignment helps where the CF signal is weaker.

---

"""
    elif dataset == "amazon":
        md += """- LightGCN works from the user × product bipartite graph only; it has no
  notion of review timestamps, so on a "predict the next review" task its
  ceiling is intrinsically lower than a sequential model's. The gap is
  expected — what matters here is whether RLMRec's alignment closes the gap
  more on Amazon-Books than it does on Yelp or ML-1M.

### Take-away
Amazon-Books is the dataset most similar to the RLMRec paper's own benchmark
suite: very sparse per-user interactions, long item tail, and rich item-side
text. If the contrastive alignment is going to pay off, this is where the
project's hypothesis (RLMRec's lift transfers to a sequential backbone) is
most testable.

---

"""
    else:
        md += """- LightGCN is a non-sequential graph model — it never sees the order of a
  user's ratings, only the bipartite user-movie graph. SASRec's sequential
  attention is a much better fit for the LOO "next item" task on a dense
  dataset like ML-1M, so a sizable gap (LightGCN behind SASRec) is the
  expected baseline behaviour, not a bug.

### Take-away
ML-1M is a *dense* sequential dataset where SASRec is near its strongest
regime. RLMRec's value-add was demonstrated in the paper on sparser
graph-CF datasets (Yelp, Amazon-Books, Steam), so the cleaner test of the
RLMRec hypothesis lives in the Yelp / Amazon-Books overviews, not this one.

---

"""
    md += paper_section
    if seed_count > 1:
        seed_caveat = (
            f"- **{seed_count} seeds** per run; the table in §1 reports "
            "**mean ± std** across them, and `comparison_bars.png` shows "
            "error bars. Per-seed numbers live in "
            f"`{art}/comparison/comparison_per_seed.csv`."
        )
    else:
        seed_caveat = ("- **Single seed (42)** for every run — variance is "
                       "unmeasured. Re-run with `--seeds 42 43 44 45 46` "
                       "(or use the `run_all.py` default) for std.")
    md += f"""
---

## 4. Caveats & honest limitations

{seed_caveat}
- **Approximated profiles**: templated user/item text is a weaker signal
  than the LLM-generated profiles used by the RLMRec paper. The paper's
  reported gains explicitly depend on richer profile text.
- **Hyperparameters were not tuned** for the RLMRec variants — λ, τ, and
  dim were taken from sensible defaults rather than swept.
- **Best-checkpoint selection on val NDCG@10**: epochs at peak val are in
  the run_metadata; if a run is still rising at the cap, it had headroom.

---

## 5. Files referenced

- `{art}/comparison/comparison.csv` — per-model means used in §1
- `{art}/comparison/comparison_aggregate.csv` — per-model **mean / std / n** across seeds
- `{art}/comparison/comparison_per_seed.csv` — long-format `(model, seed, metric, value)`
- `{art}/comparison/comparison_bars.png` — bar chart with ±std error bars
- `{art}/comparison/val_NDCG@10_curves.png` — validation NDCG@10 mean curve + ±1σ band per model
- `{art}/comparison/summary.txt` — formatted aggregate + per-seed tables
- `{art}/{{sasrec,rlmrec_sasrec,rlmrec_lightgcn}}/seed_<N>/` — full per-seed training artifacts (history, plots, checkpoint)
- `{art}/{{sasrec,rlmrec_sasrec,rlmrec_lightgcn}}/test_metrics_per_seed.csv` — per-model per-seed metrics
- `{art}/{{sasrec,rlmrec_sasrec,rlmrec_lightgcn}}/test_metrics_aggregate.csv` — per-model mean / std / n
- `{art}/{{sasrec,rlmrec_sasrec,rlmrec_lightgcn}}/run_metadata.json` — hyperparameters (copied from one representative seed)
"""
    return md


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ml1m", "yelp", "amazon"],
                        required=True)
    parser.add_argument("--artifacts_root", type=str, default=None,
                        help="Override artifacts root (default depends on --dataset).")
    parser.add_argument("--out", type=str, default=None,
                        help="Override output path (default: <root>/comparison/results_overview.md).")
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    root = Path(args.artifacts_root or cfg["artifacts_root"])
    out_path = Path(args.out or (root / "comparison" / "results_overview.md"))

    per_model: dict[str, dict[str, float]] = {}
    per_model_agg: dict[str, dict[str, dict[str, float]]] = {}
    metas: dict[str, dict] = {}
    for m in MODELS:
        d = root / m["name"]
        per_model[m["label"]] = _load_metrics(d / "test_metrics.csv")
        per_model_agg[m["label"]] = _load_aggregate(
            d / "test_metrics_aggregate.csv")
        metas[m["label"]] = _load_metadata(d / "run_metadata.json")

    if not any(per_model.values()):
        runner = {"ml1m": "run_all.py", "yelp": "run_all_yelp.py",
                  "amazon": "run_all_amazon.py"}[args.dataset]
        raise SystemExit(
            f"No test_metrics.csv found under {root}/. Run `python {runner}` "
            f"first (or use `python run_everything.py`).")

    seed_count = max(
        (int(a["n"]) for d in per_model_agg.values() for a in d.values()),
        default=1,
    )
    md = render_overview(args.dataset, per_model, metas, root,
                         per_model_agg=per_model_agg, seed_count=seed_count)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(md)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
