"""Yelp Open Dataset loader + sequence preprocessing.

The Yelp Open Dataset (https://www.yelp.com/dataset) is not auto-downloaded
because it requires accepting Yelp's terms of service. Place the JSON files
under `data/yelp/`:

    data/yelp/yelp_academic_dataset_review.json
    data/yelp/yelp_academic_dataset_business.json
    data/yelp/yelp_academic_dataset_user.json

Preprocessing follows the spirit of the RLMRec paper's Yelp split:
  - keep only reviews from `--start_year` onward (default 2018)
  - iterative k-core filter on (user, business) interactions until every
    user has >= `min_inter` reviews and every business has >= `min_inter`
    reviews (default 5)
  - sort each user's reviews by timestamp
  - leave-one-out split: last -> test, 2nd-to-last -> val, rest -> train
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

REVIEW_FILE = "yelp_academic_dataset_review.json"
BUSINESS_FILE = "yelp_academic_dataset_business.json"
USER_FILE = "yelp_academic_dataset_user.json"


def _yelp_dir(data_dir: Path) -> Path:
    return data_dir / "yelp"


def ensure_yelp_files(data_dir: Path) -> Path:
    yd = _yelp_dir(data_dir)
    missing = [f for f in (REVIEW_FILE, BUSINESS_FILE, USER_FILE)
               if not (yd / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing Yelp dataset files in {yd}: {missing}\n"
            "Download the Yelp Open Dataset from https://www.yelp.com/dataset, "
            f"extract it, and place the JSON files at {yd}/."
        )
    return yd


def _parse_ts(s: str) -> int:
    """Yelp review date format: 'YYYY-MM-DD HH:MM:SS' -> unix seconds."""
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp())


def load_raw_interactions(yelp_dir: Path,
                          start_year: int = 2018) -> list[tuple[str, str, int, int]]:
    """Stream the review JSON; return list of (user_id, business_id, ts, stars).

    Only reviews dated start_year-01-01 or later are kept. The file is large
    (~5GB), so we never hold it all in memory simultaneously.
    """
    path = yelp_dir / REVIEW_FILE
    cutoff_ts = int(datetime(start_year, 1, 1).timestamp())
    out: list[tuple[str, str, int, int]] = []
    n_seen = n_kept = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            n_seen += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _parse_ts(rec["date"])
            if ts < cutoff_ts:
                continue
            out.append((rec["user_id"], rec["business_id"], ts,
                        int(rec.get("stars", 0))))
            n_kept += 1
            if n_seen % 1_000_000 == 0:
                print(f"  ... scanned {n_seen:,} reviews, kept {n_kept:,}")
    print(f"Loaded {n_kept:,} of {n_seen:,} reviews "
          f"(>= {start_year}-01-01)")
    return out


def kcore_filter(interactions: list[tuple[str, str, int, int]],
                 min_inter: int = 5, max_iters: int = 10
                 ) -> list[tuple[str, str, int, int]]:
    """Iteratively drop users and items with fewer than `min_inter` interactions."""
    cur = interactions
    for it in range(max_iters):
        u_cnt: Counter = Counter()
        b_cnt: Counter = Counter()
        for u, b, _, _ in cur:
            u_cnt[u] += 1
            b_cnt[b] += 1
        keep_u = {u for u, c in u_cnt.items() if c >= min_inter}
        keep_b = {b for b, c in b_cnt.items() if c >= min_inter}
        nxt = [r for r in cur if r[0] in keep_u and r[1] in keep_b]
        print(f"  k-core iter {it + 1}: users {len(u_cnt):,} -> {len(keep_u):,}, "
              f"items {len(b_cnt):,} -> {len(keep_b):,}, "
              f"interactions {len(cur):,} -> {len(nxt):,}")
        if len(nxt) == len(cur):
            break
        cur = nxt
    return cur


def build_sequences(yelp_dir: Path, start_year: int = 2018,
                    min_inter: int = 5):
    """Return (sequences, num_users, num_items, user_map, item_map).

    Index 0 is reserved as a padding slot for both users and items.
    """
    print("Reading Yelp reviews ...")
    raw = load_raw_interactions(yelp_dir, start_year=start_year)
    print(f"Applying iterative {min_inter}-core filter ...")
    raw = kcore_filter(raw, min_inter=min_inter)

    # Group by user, sort by ts (stable on (ts, business_id) for determinism).
    by_user: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for u, b, ts, _ in raw:
        by_user[u].append((ts, b))

    user_ids = sorted(by_user.keys())
    user_map = {u: i + 1 for i, u in enumerate(user_ids)}

    all_items = sorted({b for events in by_user.values() for _, b in events})
    item_map = {b: i + 1 for i, b in enumerate(all_items)}

    sequences: dict[int, list[int]] = {}
    for u in user_ids:
        events = sorted(by_user[u], key=lambda x: (x[0], x[1]))
        seq = [item_map[b] for _, b in events]
        if len(seq) < 5:
            continue
        sequences[user_map[u]] = seq

    num_users = len(user_map)
    num_items = len(all_items)
    total = sum(len(s) for s in sequences.values())
    print(f"Users: {num_users:,} | Items: {num_items:,} | Interactions: {total:,}")
    return sequences, num_users, num_items, user_map, item_map


def split_sequences(sequences: dict[int, list[int]]):
    """Same leave-one-out split as ML-1M."""
    train, val, test = {}, {}, {}
    for u, seq in sequences.items():
        train[u] = seq[:-2]
        val[u] = seq[-2]
        test[u] = seq[-1]
    return train, val, test


# Reuse the same training-time dataset class as ML-1M (it's dataset-agnostic).
from sasrec_data import SASRecTrainDataset, build_eval_inputs  # noqa: E402,F401


# --------------------------------------------------------------------------- #
# Semantic profiles
# --------------------------------------------------------------------------- #

def load_business_meta(yelp_dir: Path) -> dict[str, dict]:
    """Read the business JSON, return {business_id -> meta dict}."""
    path = yelp_dir / BUSINESS_FILE
    out: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[rec["business_id"]] = {
                "name": rec.get("name", "") or "",
                "city": rec.get("city", "") or "",
                "state": rec.get("state", "") or "",
                "stars": rec.get("stars"),
                "categories": rec.get("categories") or "",
            }
    return out


def load_user_meta(yelp_dir: Path) -> dict[str, dict]:
    """Read the user JSON, return {user_id -> meta dict (review_count, avg_stars)}."""
    path = yelp_dir / USER_FILE
    out: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[rec["user_id"]] = {
                "review_count": int(rec.get("review_count", 0) or 0),
                "average_stars": float(rec.get("average_stars", 0.0) or 0.0),
                "yelping_since": rec.get("yelping_since", "") or "",
            }
    return out


def build_business_texts(business_meta: dict[str, dict],
                         item_map_raw_to_new: dict[str, int]) -> dict[int, str]:
    """{new_item_id -> 'Name. Categories: A, B. Located in City, ST. 4.0 stars.'}."""
    texts: dict[int, str] = {}
    for raw_id, new_id in item_map_raw_to_new.items():
        m = business_meta.get(raw_id)
        if m is None:
            texts[new_id] = "Unknown business."
            continue
        parts = []
        name = m["name"].strip() or "An unnamed business"
        parts.append(name)
        cats = (m["categories"] or "").strip()
        if cats:
            parts.append(f"Categories: {cats}")
        loc = ", ".join(p for p in (m["city"].strip(), m["state"].strip()) if p)
        if loc:
            parts.append(f"Located in {loc}")
        if m["stars"] is not None:
            parts.append(f"{float(m['stars']):.1f} stars")
        texts[new_id] = ". ".join(parts) + "."
    return texts


def _categories_for_item(business_meta: dict[str, dict],
                         inv_item_map: dict[int, str]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for new_id, raw_id in inv_item_map.items():
        m = business_meta.get(raw_id)
        if m and m["categories"]:
            out[new_id] = [c.strip() for c in m["categories"].split(",")
                           if c.strip()]
        else:
            out[new_id] = []
    return out


def build_user_texts(user_meta_raw: dict[str, dict],
                     user_map_raw_to_new: dict[str, int],
                     train_seqs: dict[int, list[int]],
                     business_texts: dict[int, str],
                     item_categories: dict[int, list[str]],
                     top_items: int = 5,
                     top_categories: int = 3) -> dict[int, str]:
    """Short profile per user: yelp tenure + favorite categories + recent visits."""
    texts: dict[int, str] = {}
    inv_user_map = {new: raw for raw, new in user_map_raw_to_new.items()}
    for u, seq in train_seqs.items():
        raw_id = inv_user_map.get(u)
        m = user_meta_raw.get(raw_id, {}) if raw_id else {}

        cat_counter: Counter[str] = Counter()
        for iid in seq:
            for c in item_categories.get(iid, []):
                cat_counter[c] += 1
        fav = [c for c, _ in cat_counter.most_common(top_categories)]
        fav_str = ", ".join(fav) if fav else "a variety of places"

        recent = []
        for iid in seq[-top_items:][::-1]:
            t = business_texts.get(iid, "")
            if t:
                # Keep just the business name (first sentence).
                recent.append(t.split(".")[0])
        recent_str = "; ".join(recent) if recent else "various businesses"

        rc = m.get("review_count", len(seq))
        avg = m.get("average_stars", 0.0)
        since = (m.get("yelping_since") or "")[:4]
        tenure = f" who has been on Yelp since {since}" if since else ""
        texts[u] = (
            f"A Yelp user{tenure} with {rc} reviews and an average rating of "
            f"{avg:.1f} stars, who frequently visits {fav_str}. "
            f"Recently reviewed: {recent_str}."
        )
    return texts


def encode_semantic(texts: list[str], model_name: str, device: str,
                    batch_size: int = 128) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    print(f"Encoding {len(texts):,} texts with {model_name} on {device} ...")
    emb = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                       show_progress_bar=True, normalize_embeddings=True)
    return emb.astype(np.float32)


def prepare_semantic_embeddings(data_dir: Path, cache_dir: Path,
                                yelp_dir: Path,
                                item_map: dict[str, int],
                                user_map: dict[str, int],
                                train_seqs: dict[int, list[int]],
                                model_name: str,
                                device: str
                                ) -> tuple[np.ndarray, np.ndarray, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = model_name.replace("/", "__")
    item_cache = cache_dir / f"yelp_item_emb_{safe_tag}.npy"
    user_cache = cache_dir / f"yelp_user_emb_{safe_tag}.npy"

    if item_cache.exists() and user_cache.exists():
        item_emb = np.load(item_cache)
        user_emb = np.load(user_cache)
        if item_emb.shape[0] == len(item_map) + 1 and user_emb.shape[0] == len(user_map) + 1:
            print(f"Loaded cached Yelp semantic embeddings from {cache_dir}")
            return item_emb, user_emb, int(item_emb.shape[1])
        print("Cached Yelp semantic embeddings have stale shape; rebuilding.")

    business_meta = load_business_meta(yelp_dir)
    user_meta_raw = load_user_meta(yelp_dir)

    business_texts = build_business_texts(business_meta, item_map)
    inv_item_map = {new: raw for raw, new in item_map.items()}
    item_categories = _categories_for_item(business_meta, inv_item_map)
    user_texts = build_user_texts(user_meta_raw, user_map, train_seqs,
                                  business_texts, item_categories)

    num_items = len(item_map)
    num_users = len(user_map)

    ordered_item_texts = [business_texts.get(i, "Unknown business.")
                          for i in range(1, num_items + 1)]
    ordered_user_texts = [user_texts.get(u, "A Yelp user.")
                          for u in range(1, num_users + 1)]

    item_arr = encode_semantic(ordered_item_texts, model_name, device)
    user_arr = encode_semantic(ordered_user_texts, model_name, device)

    dim = item_arr.shape[1]
    item_emb = np.zeros((num_items + 1, dim), dtype=np.float32)
    item_emb[1:] = item_arr
    user_emb = np.zeros((num_users + 1, dim), dtype=np.float32)
    user_emb[1:] = user_arr

    np.save(item_cache, item_emb)
    np.save(user_cache, user_emb)
    print(f"Cached Yelp semantic embeddings in {cache_dir}")
    return item_emb, user_emb, dim


def load_yelp_with_semantics(data_dir: Path, cache_dir: Path, model_name: str,
                             device: str, start_year: int = 2018,
                             min_inter: int = 5):
    """One-stop loader. Mirrors `load_ml1m_with_semantics` exactly in shape."""
    yelp_dir = ensure_yelp_files(data_dir)
    sequences, num_users, num_items, user_map, item_map = build_sequences(
        yelp_dir, start_year=start_year, min_inter=min_inter)
    train_seqs, val_targets, test_targets = split_sequences(sequences)

    item_sem, user_sem, sem_dim = prepare_semantic_embeddings(
        data_dir, cache_dir, yelp_dir, item_map, user_map, train_seqs,
        model_name, device)

    return {
        "sequences": sequences,
        "train_seqs": train_seqs,
        "val_targets": val_targets,
        "test_targets": test_targets,
        "num_users": num_users,
        "num_items": num_items,
        "user_map": user_map,
        "item_map": item_map,
        "item_sem": torch.from_numpy(item_sem),
        "user_sem": torch.from_numpy(user_sem),
        "sem_dim": sem_dim,
    }
