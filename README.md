# Bringing Language-Model Semantics into Sequential Recommendation

A reproducible reference implementation for the paper *Bringing Language-Model Semantics into Sequential Recommendation: A Contrastive Alignment Bridge Between SASRec and RLMRec*.

The research question is simple: **RLMRec's contrastive semantic-alignment recipe was published on graph-CF backbones (LightGCN, SGL, SimGCL, DCCF, etc.). Does that same recipe transfer to a sequential SASRec backbone?** To answer it we train three models head-to-head on the same data splits and evaluator:

1. **SASRec** — paper-faithful baseline (Kang & McAuley, ICDM 2018).
2. **RLMRec+LightGCN** — paper-faithful RLMRec-Con reproduction (Ren et al., WWW 2024).
3. **RLMRec+SASRec** — project-specific extension: SASRec backbone with RLMRec's two InfoNCE alignment heads (user-state ↔ user-profile-text, item-embedding ↔ item-profile-text). Not in the original RLMRec paper.

All three are evaluated on **MovieLens-1M**, **Yelp** (Open Dataset), and **Amazon-Books** (Reviews 2023), under a strict full-rank leave-one-out protocol.

---

## Table of contents

1. [Repository layout](#1-repository-layout)
2. [What every file does](#2-what-every-file-does)
3. [Environment setup](#3-environment-setup)
4. [Getting the datasets](#4-getting-the-datasets)
5. [Reproducing the paper](#5-reproducing-the-paper)
6. [Per-script flag reference](#6-per-script-flag-reference)
7. [Output / artifact layout](#7-output--artifact-layout)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Repository layout

```
recommender_system/
├── README.md                                ← this file
│
├── Research Paper/
│    ├── RLMRec_SASRec.pdf                   ← Research Paper
│    ├── RLMRec_SASRec_Presentation.pdf      ← Presentation as an overview of the Research Paper
│ 
├── sasrec_model.py                          ← SASRec architecture
├── rlmrec_model.py                          ← RLMRecSASRec wrapper + InfoNCE
├── lightgcn_model.py                        ← LightGCN + LightGCNRLMRec wrapper
│
├── sasrec_data.py                           ← MovieLens-1M loader (auto-download)
├── yelp_data.py                             ← Yelp loader (manual download)
├── amazon_data.py                           ← Amazon Reviews 2023 loader (auto-download)
├── rlmrec_data.py                           ← ML-1M-specific semantic-profile builder
├── eval_full_rank.py                        ← shared full-rank LOO evaluator
│
├── sasrec_train.py                          ← train pure SASRec
├── rlmrec_sasrec_train.py                   ← train RLMRec+SASRec
├── rlmrec_lightgcn_train.py                 ← train RLMRec+LightGCN
│
├── seed_aggregation.py                      ← multi-seed aggregation helpers
├── run_all.py                               ← orchestrator: ML-1M, 3 models × N seeds
├── run_all_yelp.py                          ← orchestrator: Yelp,   3 models × N seeds
├── run_all_amazon.py                        ← orchestrator: Amazon, 3 models × N seeds
├── run_everything.py                        ← top-level: runs all three orchestrators + make_overview
├── run_ablations.py                         ← ablation grid for RLMRec+SASRec (and RLMRec+LightGCN)
├── make_overview.py                         ← generate per-dataset results_overview.md
```

---

## 2. What every file does

### 2.1 Model files

| File | Purpose |
|------|---------|
| `sasrec_model.py` | Implements `SASRec`: an item-embedding table, sinusoidal-position-add, N causal self-attention blocks, and a final-position dot-product head. Exposes `forward(input_seq, pos_seq, neg_seq)` for BCE-style training and `score_all_items(input_seq) -> (B, num_items+1)` for full-rank scoring. Defaults match the paper (`hidden=50, blocks=2, heads=1, dropout=0.2, max_len=200`). |
| `rlmrec_model.py` | Defines `ProjectionHead` (linear or 2-layer GELU MLP), `RLMRecSASRec` (wraps `SASRec` with two `ProjectionHead`s, one for user-state and one for item-embedding), and `info_nce(projected, target, temperature)` — symmetric cross-entropy on cosine-similarity logits. This is the InfoNCE alignment loss that ties CF embeddings to the frozen sentence-transformer space. |
| `lightgcn_model.py` | Implements `LightGCN` (He et al., 2020) using edge-based `index_add_` propagation (works on MPS without sparse matmul), `LightGCNRLMRec` (wraps `LightGCN` with the same `ProjectionHead`s), and `build_norm_adjacency(...)` which constructs the symmetric-normalized bipartite adjacency. |

### 2.2 Data + preprocessing modules

All three dataset modules share the same interface: `load_<ds>_with_semantics(data_dir, cache_dir, model_name, device, ...) -> bundle dict`.

| File | Purpose |
|------|---------|
| `sasrec_data.py` | MovieLens-1M loader: `download_ml1m` (auto-fetches the zip from `files.grouplens.org`), `build_sequences` (5-core implicit, sorts by timestamp, remaps to 1-indexed IDs), `split_sequences` (leave-one-out), `SASRecTrainDataset` (yields `(input_seq, pos_seq, neg_seq)` for BCE), and `build_eval_inputs`. |
| `yelp_data.py` | Yelp Open Dataset loader. **Manual download** is required (Yelp's TOS). Streams the ~5 GB review JSON, filters by `start_year`, applies iterative k-core (`min_inter` default 5), builds business + user profile texts from metadata, encodes them with a frozen sentence-transformer, and caches the result as `.npy`. Function `load_yelp_with_semantics` is the one-stop loader used by both RLMRec training scripts. |
| `amazon_data.py` | Amazon Reviews 2023 (McAuley Lab) loader. **Auto-downloads** the gzipped review and metadata JSONL for one category (default `Books` — chosen because it carries the richest per-item text and matches the RLMRec paper's Amazon-Books benchmark). Same preprocessing recipe as Yelp (start-year + iterative k-core). |
| `rlmrec_data.py` | ML-1M-specific semantic-profile builder. Constructs item texts as `"Title (Year). Genres: A, B, C."` and user texts as `"A <age> <gender> <occupation> who enjoys <top genres>. Recently watched: <recent titles>."`. Encodes them with `sentence-transformers` and caches the result. |
| `eval_full_rank.py` | Shared full-rank leave-one-out evaluator. Computes HR@K and NDCG@K for K ∈ {1, 5, 10, 20} plus MRR. Used identically by all three trainers, so headline numbers are directly comparable. `build_exclusion_mask` masks each user's training history (and val item, for test) so we never score against an item the model has already seen. |

### 2.3 Training scripts

| File | Purpose |
|------|---------|
| `sasrec_train.py` | End-to-end pure-SASRec trainer. BCE-with-logits loss; auto-picks CUDA → MPS → CPU; selects the best epoch by val NDCG@10; writes `<out_dir>/{train_history.csv, sasrec_best.pt, test_metrics.csv, run_metadata.json, *.png}`. Supports `--dataset {ml1m,yelp,amazon}`. |
| `rlmrec_sasrec_train.py` | RLMRec+SASRec trainer. Same SASRec backbone, plus two InfoNCE alignment heads at every batch. Loss is `cf_loss + λ_user · user_align + λ_item · item_align`. |
| `rlmrec_lightgcn_train.py` | RLMRec+LightGCN trainer (paper-faithful RLMRec). BPR ranking loss + L2 reg on layer-0 embeddings + two InfoNCE heads, identical loss schema to the SASRec variant. |

### 2.4 Orchestration / aggregation

| File | Purpose |
|------|---------|
| `seed_aggregation.py` | Shared helpers: read per-seed `test_metrics.csv`, compute mean/std, write per-seed/aggregate/comparison CSVs, plot grouped bars with error bars and validation curves with ±1σ bands. |
| `run_all.py` | ML-1M orchestrator. For each of 3 models × N seeds, invokes the trainer as a subprocess into `artifacts/<model>/seed_<N>/`, then aggregates into `artifacts/comparison/`. |
| `run_all_yelp.py` | Same as `run_all.py`, but writes to `artifacts_yelp/` and forwards `--dataset yelp`. Adds Yelp-specific filter flags. |
| `run_all_amazon.py` | Same as `run_all.py`, but writes to `artifacts_amazon/` and forwards `--dataset amazon`. Adds Amazon-specific filter and category flags. |
| `run_everything.py` | Top-level: invokes `run_all.py`, `run_all_yelp.py`, `run_all_amazon.py`, then `make_overview.py` for each dataset. This is the one-shot reproduction entrypoint. |
| `run_ablations.py` | Separate ablation orchestrator for the RLMRec+SASRec ablation grid (λ sweep, τ sweep, MLP-vs-linear projection head, MiniLM-vs-BGE encoder, loss-component flips, plus optional E5-Mistral-7B). 50 epochs × 3 seeds × N variants × 3 datasets by default. Uses paired t-tests against an in-ablation reference variant (`sasrec_full` / `lightgcn_full`). |
| `make_overview.py` | Reads the comparison CSVs and per-model `run_metadata.json` for one dataset and renders `<artifacts_root>/comparison/results_overview.md` with the headline table, run-config table, head-to-head observations, and a dataset-specific paper-comparison section. |

---

## 3. Environment setup

The repository ships with a fully-configured Python 3.12 virtual environment at `venv/`. **You must activate it before running anything**, otherwise `python` will resolve to the system interpreter, which does not have `torch`, `sentence-transformers`, or `scipy` installed:

```bash
cd "/path/to/dir"
source venv/bin/activate
```

If you need to recreate the environment from scratch (e.g. on a new machine), install the following packages into a Python 3.12 venv:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch numpy scipy matplotlib sentence-transformers huggingface_hub tqdm transformers
```

The exact versions in `venv/` (as of the paper submission) are:

- `torch==2.11.0`
- `numpy==2.4.2`
- `scipy==1.17.1`
- `matplotlib==3.10.8`
- `sentence-transformers==5.4.1`
- `huggingface-hub==1.11.0`
- `transformers==5.5.4`
- `tqdm==4.67.3`

### Device autodetection

Every trainer calls `sasrec_train.pick_device`, which prefers **CUDA → MPS → CPU** in that order. On a Mac M-series chip it will land on MPS automatically. No flag is needed.

---

## 4. Getting the datasets

All three datasets should be placed under `data/`. **Two of them auto-download on first run** (MovieLens-1M and Amazon). Yelp requires a manual download because of Yelp's terms-of-service click-through.

The expected on-disk layout after all downloads:

```
data/
├── ml-1m/
│   ├── ratings.dat                                (24 MB)
│   ├── movies.dat                                 (171 KB)
│   ├── users.dat                                  (134 KB)
│   └── README                                     (5 KB, from GroupLens)
│
├── yelp/
│   ├── yelp_academic_dataset_review.json          (5.3 GB)
│   ├── yelp_academic_dataset_business.json        (119 MB)
│   ├── yelp_academic_dataset_user.json            (3.3 GB)
│   ├── yelp_academic_dataset_checkin.json         (287 MB, optional)
│   ├── yelp_academic_dataset_tip.json             (181 MB, optional)
│   └── Dataset_User_Agreement.pdf                 (80 KB)
│
├── amazon/
│   ├── Books.jsonl.gz                             (6.2 GB)
│   └── meta_Books.jsonl.gz                        (4.9 GB)
│
└── semantic_cache/                                (auto-populated *.npy cache, see §4.4)
```

### 4.1 MovieLens-1M (auto-downloaded)

The loader (`sasrec_data.download_ml1m`) fetches the zip from GroupLens and extracts it under `data/ml-1m/` on first run. **You don't have to do anything**. If you want to download it manually:

```bash
mkdir -p data
cd data
curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
# This creates data/ml-1m/ containing ratings.dat, movies.dat, users.dat, README.
cd ..
```

Source URL: <https://files.grouplens.org/datasets/movielens/ml-1m.zip>
Project page: <https://grouplens.org/datasets/movielens/1m/>

### 4.2 Yelp Open Dataset (manual download required)

The Yelp Open Dataset is too large to auto-download anonymously, and Yelp requires you to accept their dataset terms before downloading. Steps:

1. Open <https://www.yelp.com/dataset> in a browser.
2. Click **Download Dataset**, accept the dataset agreement, and download the **JSON** archive (e.g. `yelp_dataset.tar` or `yelp_dataset.tgz`). The download is ~4 GB compressed.
3. Extract it. You should end up with files named `yelp_academic_dataset_business.json`, `yelp_academic_dataset_review.json`, `yelp_academic_dataset_user.json`, `yelp_academic_dataset_checkin.json`, `yelp_academic_dataset_tip.json`.
4. Move them into `data/yelp/`:

```bash
mkdir -p data/yelp
mv /path/to/extracted/yelp_academic_dataset_*.json data/yelp/
```

The three files actually used are `*_review.json`, `*_business.json`, `*_user.json`. The checkin and tip files are optional (the loader doesn't read them).

If the JSON files are missing, `yelp_data.ensure_yelp_files` will raise a `FileNotFoundError` with a clear message telling you where to put them.

### 4.3 Amazon Reviews 2023 — Books (auto-downloaded)

`amazon_data.ensure_amazon_files` will auto-fetch the gzipped Books review and metadata files from the McAuley Lab's public hosting on first run, no authentication needed. The download is ~11 GB total and can take a while.

The two URLs the loader hits:

- Reviews: <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Books.jsonl.gz>
- Metadata: <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz>

To download manually instead (e.g. if your machine has a poor connection during training):

```bash
mkdir -p data/amazon
cd data/amazon
curl -O https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Books.jsonl.gz
curl -O https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz
cd ../..
```

If you'd rather benchmark a different category (e.g. `Electronics`, `Movies_and_TV`), the URL template is the same — just swap `Books` for the category name listed on the [Amazon Reviews 2023 page](https://amazon-reviews-2023.github.io/). Pass `--amazon_category <Name>` to any trainer or orchestrator. **Note**: the paper's headline results use `Books` — change it only if you understand you'll no longer be reproducing the paper.

### 4.4 Sentence-transformer embedding cache

The semantic-profile pipeline encodes user/item texts with a frozen sentence-transformer and caches the result as `.npy` under `data/semantic_cache/`. **You don't download this cache** — it's generated on first run.

The cache key is `{dataset}_{user|item}_emb_{model_name_with_slashes_replaced}.npy`. The reference set of files (after running the headline grid with the default BGE-large encoder, plus the MiniLM ablation):

```
data/semantic_cache/
├── user_emb_BAAI__bge-large-en-v1.5.npy                              (ML-1M, 24 MB)
├── item_emb_BAAI__bge-large-en-v1.5.npy                              (ML-1M, 15 MB)
├── yelp_user_emb_BAAI__bge-large-en-v1.5.npy                         (426 MB)
├── yelp_item_emb_BAAI__bge-large-en-v1.5.npy                         (221 MB)
├── amazon_books_user_emb_BAAI__bge-large-en-v1.5.npy                 (640 MB)
├── amazon_books_item_emb_BAAI__bge-large-en-v1.5.npy                 (460 MB)
├── user_emb_sentence-transformers__all-MiniLM-L6-v2.npy              (ML-1M MiniLM)
├── item_emb_sentence-transformers__all-MiniLM-L6-v2.npy              (ML-1M MiniLM)
├── yelp_user_emb_sentence-transformers__all-MiniLM-L6-v2.npy
├── yelp_item_emb_sentence-transformers__all-MiniLM-L6-v2.npy
├── amazon_books_user_emb_sentence-transformers__all-MiniLM-L6-v2.npy
└── amazon_books_item_emb_sentence-transformers__all-MiniLM-L6-v2.npy
```

If you delete these `.npy` files, the next training run will regenerate them — but that costs significant wall-clock time. Leave them in place.

---

## 5. Reproducing the paper

### 5.1 One command (the full grid)

The simplest path — train SASRec, RLMRec+LightGCN, and RLMRec+SASRec on all three datasets, with the paper's default settings (5 seeds × 75 epochs):

```bash
source venv/bin/activate
python run_everything.py
```

After this finishes you'll have:

- `artifacts/comparison/comparison.csv` and `results_overview.md` (ML-1M)
- `artifacts_yelp/comparison/comparison.csv` and `results_overview.md` (Yelp)
- `artifacts_amazon/comparison/comparison.csv` and `results_overview.md` (Amazon-Books)

These are the files the paper's Table 1 and Figures 2–3 are derived from.

### 5.2 Single dataset

```bash
# ML-1M only
python run_everything.py --datasets ml1m

# Yelp only
python run_everything.py --datasets yelp

# Amazon-Books only
python run_everything.py --datasets amazon
```

Or equivalently, invoke the per-dataset orchestrators directly:

```bash
python run_all.py            # ML-1M
python run_all_yelp.py       # Yelp
python run_all_amazon.py     # Amazon-Books
```

### 5.3 Single model on a single dataset, single seed

If you only want to sanity-check one variant:

```bash
python sasrec_train.py            --dataset ml1m   --seed 42 --out_dir artifacts/sasrec/seed_42
python rlmrec_sasrec_train.py     --dataset yelp   --seed 42 --out_dir artifacts_yelp/rlmrec_sasrec/seed_42
python rlmrec_lightgcn_train.py   --dataset amazon --seed 42 --out_dir artifacts_amazon/rlmrec_lightgcn/seed_42
```

### 5.4 Smoke test (a few epochs)

To confirm the pipeline works end-to-end before launching the real grid:

```bash
python sasrec_train.py --dataset ml1m --seed 42 --epochs 5 --eval_every 5 \
    --out_dir /tmp/smoke_sasrec_ml1m
```

### 5.5 Aggregation-only (no retraining)

If per-seed CSVs already exist (e.g. you killed the run halfway, or you just want to regenerate the markdown and PNGs), skip training entirely:

```bash
python run_everything.py --compare_only
# or per-dataset:
python run_all.py        --compare_only
python run_all_yelp.py   --compare_only
python run_all_amazon.py --compare_only
```

You can also rebuild a single dataset's markdown without invoking the runner:

```bash
python make_overview.py --dataset ml1m
python make_overview.py --dataset yelp
python make_overview.py --dataset amazon
```

### 5.6 Ablation grid

The ablation grid (Table 2 in the paper) is a separate run — it uses 50 epochs × 3 seeds × 14 variants × 3 datasets, with paired t-tests against an in-ablation reference (`sasrec_full` / `lightgcn_full`).

```bash
# Full ablation grid
python run_ablations.py

# Smoke test on one variant
python run_ablations.py --datasets ml1m --variants sasrec_no_user_align \
    --seeds 42 --epochs 5

# Re-aggregate existing per-seed CSVs without retraining
python run_ablations.py --aggregate_only

# Add the heavy E5-Mistral-7B encoder variant (not included in research paper)
python run_ablations.py --include_e5_mistral
```

Output goes to `artifacts_ablation/{ml1m,yelp,amazon}/comparison/ablation_overview.md`.

---

## 6. Per-script flag reference

### 6.1 `sasrec_train.py`

Pure SASRec trainer.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset` | str | `ml1m` | One of `ml1m`, `yelp`, `amazon`. |
| `--epochs` | int | `75` | Number of training epochs. |
| `--batch_size` | int | `128` | Mini-batch size. |
| `--max_len` | int | `200` | Max sequence length (SASRec paper default). |
| `--hidden` | int | `50` | Hidden dimension (SASRec paper default). |
| `--blocks` | int | `2` | Number of self-attention blocks. |
| `--heads` | int | `1` | Number of attention heads. |
| `--dropout` | float | `0.2` | Dropout rate. |
| `--lr` | float | `1e-3` | Adam learning rate. |
| `--l2` | float | `0.0` | Adam weight decay. |
| `--eval_every` | int | `20` | Run validation every N epochs (+ always on the last epoch). |
| `--seed` | int | `42` | RNG seed (PyTorch + NumPy + Python random). |
| `--num_workers` | int | `0` | DataLoader workers. |
| `--data_dir` | str | `data` | Where to find / download data. |
| `--out_dir` | str | `artifacts/sasrec` | Where to write checkpoints, history, metrics. |
| `--yelp_start_year` | int | `2018` | Drop Yelp reviews before YYYY-01-01. |
| `--yelp_min_inter` | int | `5` | K for Yelp k-core filter. |
| `--amazon_start_year` | int | `2018` | Drop Amazon reviews before YYYY-01-01. |
| `--amazon_min_inter` | int | `5` | K for Amazon k-core filter. |
| `--amazon_category` | str | `Books` | Amazon Reviews 2023 category. |

### 6.2 `rlmrec_sasrec_train.py`

Same flags as `sasrec_train.py`, **plus**:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--cache_dir` | str | `data/semantic_cache` | Where to cache the `.npy` profile embeddings. |
| `--out_dir` | str | `artifacts/rlmrec_sasrec` | (Default differs from `sasrec_train.py`.) |
| `--model_name` | str | `BAAI/bge-large-en-v1.5` | Sentence-transformer / HF model used to encode user & item profile texts. |
| `--lambda_item` | float | `0.1` | Weight on the item-alignment InfoNCE loss. |
| `--lambda_user` | float | `0.1` | Weight on the user-alignment InfoNCE loss. |
| `--temperature` | float | `0.1` | InfoNCE temperature. |
| `--proj_head` | str | `mlp` | `mlp` (2-layer GELU) or `linear` (single nn.Linear). |
| `--items_per_batch` | int | `512` | Unique items sampled per batch for item-alignment InfoNCE. |

### 6.3 `rlmrec_lightgcn_train.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset` | str | `ml1m` | One of `ml1m`, `yelp`, `amazon`. |
| `--epochs` | int | `75` | Number of training epochs. |
| `--batch_size` | int | `2048` | Mini-batch size (BPR pairs). |
| `--dim` | int | `64` | LightGCN embedding dimension. |
| `--num_layers` | int | `3` | LightGCN propagation depth. |
| `--lr` | float | `1e-3` | Adam learning rate. |
| `--l2` | float | `1e-4` | L2 reg on layer-0 embeddings. |
| `--eval_every` | int | `20` | Validation cadence. |
| `--eval_batch_size` | int | `256` | Eval-time user batch size. |
| `--seed` | int | `42` | RNG seed. |
| `--data_dir` | str | `data` | Data directory. |
| `--cache_dir` | str | `data/semantic_cache` | Embedding cache. |
| `--out_dir` | str | `artifacts/rlmrec_lightgcn` | Output directory. |
| `--model_name` | str | `BAAI/bge-large-en-v1.5` | Sentence-transformer model. |
| `--lambda_item` | float | `0.1` | Item-alignment InfoNCE weight. |
| `--lambda_user` | float | `0.1` | User-alignment InfoNCE weight. |
| `--temperature` | float | `0.1` | InfoNCE temperature. |
| `--yelp_start_year` / `--yelp_min_inter` | int | `2018` / `5` | Yelp filter knobs. |
| `--amazon_start_year` / `--amazon_min_inter` / `--amazon_category` | int / int / str | `2018` / `5` / `Books` | Amazon filter knobs. |

### 6.4 `run_all.py` (ML-1M orchestrator)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--skip` | list | `[]` | Models to skip; one or more of `sasrec`, `rlmrec_lightgcn`, `rlmrec_sasrec`. |
| `--compare_only` | bool | `False` | Skip training; only re-aggregate existing per-seed CSVs. |
| `--compare_dir` | str | `artifacts/comparison` | Where to write the comparison files. |
| `--epochs` | int | _(trainer default 75)_ | Override `--epochs` for every training subprocess. |
| `--eval_every` | int | _(trainer default 20)_ | Override `--eval_every`. |
| `--seeds` | int+ | `42 43 44 45 46` | One or more seeds to train each model with. |
| `--seed` | int | _none_ | Legacy single-seed alias for `--seeds <N>`. |
| `extra` | REMAINDER | _none_ | Anything after `--` is forwarded verbatim to every training subprocess. |

`run_all_yelp.py` and `run_all_amazon.py` add their dataset-specific filter flags (`--yelp_start_year`, `--yelp_min_inter`, `--amazon_start_year`, `--amazon_min_inter`, `--amazon_category`) and an `--also_make_overview` flag that runs `make_overview.py` immediately after the comparison.

### 6.5 `run_everything.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--datasets` | list | `ml1m yelp amazon` | Which datasets to run. |
| `--skip_models` | list | `[]` | Models to skip on every dataset. |
| `--skip_models_ml`, `--skip_models_yelp`, `--skip_models_amazon` | list | _none_ | Per-dataset overrides for `--skip_models`. |
| `--compare_only` | bool | `False` | Skip training, just regenerate comparisons + overviews. |
| `--no_overview` | bool | `False` | Skip the markdown overview generation step. |
| `--epochs` | int | _(trainer default)_ | Cap epochs for every training run. |
| `--eval_every` | int | _(trainer default)_ | Validation cadence for every training run. |
| `--seeds` | int+ | _(runner default `42 43 44 45 46`)_ | Seeds for every training run. |
| `--seed` | int | _none_ | Single-seed legacy alias. |
| `--model_name` | str | _(trainer default BGE-large)_ | Sentence-transformer encoder name (only affects RLMRec runs). |
| `--yelp_start_year` / `--yelp_min_inter` | int | _(trainer default)_ | Yelp filter knobs. |
| `--amazon_start_year` / `--amazon_min_inter` / `--amazon_category` | int / int / str | _(trainer default)_ | Amazon filter knobs. |

### 6.6 `run_ablations.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--datasets` | list | `ml1m yelp amazon` | Datasets to run. |
| `--variants` | list | _all_ | Restrict to a subset of variants. See the docstring for the full list (`sasrec_full`, `sasrec_no_user_align`, `sasrec_lambda_0.5`, etc.). |
| `--skip` | list | `[]` | Variants to skip. |
| `--seeds` | int+ | `42 43 44` | Seeds per variant. |
| `--epochs` | int | `50` | Epochs per training run. |
| `--eval_every` | int | `10` | Validation cadence. |
| `--include_e5_mistral` | bool | `False` | Add the heavy E5-Mistral-7B encoder variants. |
| `--aggregate_only` | bool | `False` | Skip training; rebuild aggregates from existing per-seed CSVs. |
| `extra` | REMAINDER | _none_ | Forwarded verbatim to every trainer subprocess. |

### 6.7 `make_overview.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset` | str (required) | _none_ | `ml1m`, `yelp`, or `amazon`. |
| `--artifacts_root` | str | _(depends on `--dataset`: `artifacts`, `artifacts_yelp`, or `artifacts_amazon`)_ | Override artifacts root. |
| `--out` | str | `<root>/comparison/results_overview.md` | Override the output path. |

---

## 7. Output / artifact layout

After `run_everything.py` completes, the structure is:

```
artifacts/                                 ← ML-1M
├── sasrec/
│   ├── seed_42/
│   │   ├── sasrec_best.pt                 ← best-by-val-NDCG@10 checkpoint
│   │   ├── train_history.csv              ← per-epoch loss + val metrics
│   │   ├── test_metrics.csv               ← final test metrics (this seed)
│   │   ├── run_metadata.json              ← all flags + best epoch + final metrics
│   │   ├── loss_curve.png
│   │   ├── val_HR@{1,5,10,20}.png
│   │   ├── val_NDCG@{1,5,10,20}.png
│   │   ├── val_MRR.png
│   │   ├── val_metrics_all.png
│   │   └── test_metrics.png
│   ├── seed_43/...
│   ├── seed_44/...
│   ├── seed_45/...
│   ├── seed_46/...
│   ├── test_metrics.csv                   ← mean over seeds (legacy schema)
│   ├── test_metrics_per_seed.csv          ← rows: seed, cols: metrics
│   ├── test_metrics_aggregate.csv         ← rows: metric, cols: mean,std,n
│   └── run_metadata.json                  ← copied from one representative seed
├── rlmrec_sasrec/                         ← same layout as sasrec/
├── rlmrec_lightgcn/                       ← same layout (checkpoints: rlmrec_lightgcn_best.pt)
└── comparison/
    ├── comparison.csv                     ← metric × model (mean only)
    ├── comparison_per_seed.csv            ← long format
    ├── comparison_aggregate.csv           ← per-model mean / std / n
    ├── comparison_bars.png                ← grouped bars with error bars
    ├── val_NDCG@10_curves.png             ← per-model mean curve + ±1σ band
    ├── summary.txt                        ← formatted mean ± std table + per-seed tables
    └── results_overview.md                ← human-readable narrative
```

`artifacts_yelp/` and `artifacts_amazon/` mirror this exactly.

`artifacts_ablation/{ml1m,yelp,amazon}/` follows a similar layout but keyed by variant name (e.g. `sasrec_lambda_0.5/seed_42/...`) and the comparison directory contains `ablation.csv`, `ablation_aggregate.csv`, `ablation_per_seed.csv`, `ablation_significance.csv`, `ablation_bars.png`, and `ablation_overview.md`.

---
