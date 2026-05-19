"""Amazon Reviews 2023 dataset loader + sequence preprocessing.

We use the McAuley Lab "Amazon Reviews 2023" release (Hou et al., 2024) and
default to the **Books** category, which carries the richest per-item text
(title, multi-paragraph descriptions, multi-level category hierarchy, author
in `store`, structured details such as the publisher and publication date).
Books is also the same category the RLMRec paper benchmarks on.

The raw review and metadata JSONL.gz files are auto-downloaded on first run
from:

    https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/

(no auth, no terms-of-service click-through, unlike Yelp). They land at:

    data/amazon/Books.jsonl.gz
    data/amazon/meta_Books.jsonl.gz

Preprocessing follows the Yelp recipe exactly:
  - keep only reviews from `--start_year` onward (default 2018)
  - iterative k-core filter on (user, parent_asin) interactions until every
    user has >= `min_inter` reviews and every product has >= `min_inter`
    reviews (default 5)
  - sort each user's reviews by timestamp
  - leave-one-out split: last -> test, 2nd-to-last -> val, rest -> train

The item id used everywhere is `parent_asin` (canonical product), so
variants of the same product fold together.
"""

from __future__ import annotations

import gzip
import json
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_CATEGORY = "Books"

REVIEW_URL_TPL = ("https://mcauleylab.ucsd.edu/public_datasets/data/"
                  "amazon_2023/raw/review_categories/{category}.jsonl.gz")
META_URL_TPL = ("https://mcauleylab.ucsd.edu/public_datasets/data/"
                "amazon_2023/raw/meta_categories/meta_{category}.jsonl.gz")


def _amazon_dir(data_dir: Path) -> Path:
    return data_dir / "amazon"


def _review_path(amazon_dir: Path, category: str) -> Path:
    return amazon_dir / f"{category}.jsonl.gz"


def _meta_path(amazon_dir: Path, category: str) -> Path:
    return amazon_dir / f"meta_{category}.jsonl.gz"


def _stream_download(url: str, dest: Path) -> None:
    """Stream-download a (possibly large) file with a temp-file rename, so
    interrupted downloads don't leave behind a partial file the next run
    will mistake for complete."""
    print(f"Downloading {url}\n  -> {dest}\n  (this can take a while; the "
          f"raw Books file is several GiB)", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    bytes_done = 0
    last_print = 0
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            bytes_done += len(chunk)
            if bytes_done - last_print >= 100 * 1024 * 1024:
                print(f"  ... {bytes_done / (1024 * 1024):.0f} MiB", flush=True)
                last_print = bytes_done
    tmp.rename(dest)
    print(f"  done ({bytes_done / (1024 * 1024):.0f} MiB total)")


def ensure_amazon_files(data_dir: Path,
                        category: str = DEFAULT_CATEGORY) -> Path:
    """Download missing review/meta JSONL.gz files for `category`.
    Returns the amazon directory."""
    ad = _amazon_dir(data_dir)
    review_path = _review_path(ad, category)
    meta_path = _meta_path(ad, category)
    if not review_path.exists():
        _stream_download(REVIEW_URL_TPL.format(category=category), review_path)
    if not meta_path.exists():
        _stream_download(META_URL_TPL.format(category=category), meta_path)
    return ad


def _normalize_ts_to_seconds(ts) -> int:
    """Amazon 2023 timestamps are unix milliseconds; we work in seconds.
    Anything beyond year-3000-in-seconds (~3.25e10) is clearly ms."""
    try:
        ts = int(ts)
    except (TypeError, ValueError):
        return 0
    if ts > 32_503_680_000:
        return ts // 1000
    return ts


def load_raw_interactions(amazon_dir: Path, category: str = DEFAULT_CATEGORY,
                          start_year: int = 2018
                          ) -> list[tuple[str, str, int, float]]:
    """Stream the review JSONL.gz; return (user_id, parent_asin, ts_seconds, rating).

    Only reviews dated start_year-01-01 or later are kept.
    """
    path = _review_path(amazon_dir, category)
    cutoff_ts = int(datetime(start_year, 1, 1).timestamp())
    out: list[tuple[str, str, int, float]] = []
    n_seen = n_kept = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            n_seen += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _normalize_ts_to_seconds(rec.get("timestamp"))
            if ts < cutoff_ts:
                continue
            user = rec.get("user_id")
            asin = rec.get("parent_asin") or rec.get("asin")
            if not user or not asin:
                continue
            out.append((user, asin, ts, float(rec.get("rating", 0) or 0)))
            n_kept += 1
            if n_seen % 1_000_000 == 0:
                print(f"  ... scanned {n_seen:,} reviews, kept {n_kept:,}",
                      flush=True)
    print(f"Loaded {n_kept:,} of {n_seen:,} reviews "
          f"(>= {start_year}-01-01, category={category})")
    return out


def kcore_filter(interactions: list[tuple[str, str, int, float]],
                 min_inter: int = 5, max_iters: int = 10
                 ) -> list[tuple[str, str, int, float]]:
    """Iteratively drop users and items with fewer than `min_inter` interactions."""
    cur = interactions
    for it in range(max_iters):
        u_cnt: Counter = Counter()
        i_cnt: Counter = Counter()
        for u, i, _, _ in cur:
            u_cnt[u] += 1
            i_cnt[i] += 1
        keep_u = {u for u, c in u_cnt.items() if c >= min_inter}
        keep_i = {i for i, c in i_cnt.items() if c >= min_inter}
        nxt = [r for r in cur if r[0] in keep_u and r[1] in keep_i]
        print(f"  k-core iter {it + 1}: users {len(u_cnt):,} -> {len(keep_u):,}, "
              f"items {len(i_cnt):,} -> {len(keep_i):,}, "
              f"interactions {len(cur):,} -> {len(nxt):,}")
        if len(nxt) == len(cur):
            break
        cur = nxt
    return cur


def build_sequences(amazon_dir: Path, category: str = DEFAULT_CATEGORY,
                    start_year: int = 2018, min_inter: int = 5):
    """Return (sequences, num_users, num_items, user_map, item_map).

    Index 0 is reserved as a padding slot for both users and items.
    """
    print(f"Reading Amazon-{category} reviews ...")
    raw = load_raw_interactions(amazon_dir, category=category,
                                start_year=start_year)
    print(f"Applying iterative {min_inter}-core filter ...")
    raw = kcore_filter(raw, min_inter=min_inter)

    by_user: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for u, i, ts, _ in raw:
        by_user[u].append((ts, i))

    user_ids = sorted(by_user.keys())
    user_map = {u: i + 1 for i, u in enumerate(user_ids)}

    all_items = sorted({i for events in by_user.values() for _, i in events})
    item_map = {i: idx + 1 for idx, i in enumerate(all_items)}

    sequences: dict[int, list[int]] = {}
    for u in user_ids:
        events = sorted(by_user[u], key=lambda x: (x[0], x[1]))
        seq = [item_map[i] for _, i in events]
        if len(seq) < 5:
            continue
        sequences[user_map[u]] = seq

    num_users = len(user_map)
    num_items = len(all_items)
    total = sum(len(s) for s in sequences.values())
    print(f"Users: {num_users:,} | Items: {num_items:,} | "
          f"Interactions: {total:,}")
    return sequences, num_users, num_items, user_map, item_map


def split_sequences(sequences: dict[int, list[int]]):
    """Same leave-one-out split as ML-1M / Yelp."""
    train, val, test = {}, {}, {}
    for u, seq in sequences.items():
        train[u] = seq[:-2]
        val[u] = seq[-2]
        test[u] = seq[-1]
    return train, val, test


# Reuse the same training-time dataset class as ML-1M / Yelp.
from sasrec_data import SASRecTrainDataset, build_eval_inputs  # noqa: E402,F401


# --------------------------------------------------------------------------- #
# Semantic profiles
# --------------------------------------------------------------------------- #

def load_item_meta(amazon_dir: Path, category: str = DEFAULT_CATEGORY,
                   keep_ids: set[str] | None = None) -> dict[str, dict]:
    """Read the meta JSONL.gz, return {parent_asin -> normalized meta dict}.

    `keep_ids`, if given, restricts the result to that set (useful because the
    meta file is huge and we only need entries for items that survived k-core).
    """
    path = _meta_path(amazon_dir, category)
    out: dict[str, dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = rec.get("parent_asin")
            if not asin:
                continue
            if keep_ids is not None and asin not in keep_ids:
                continue
            desc = rec.get("description") or []
            if isinstance(desc, list):
                desc_text = " ".join(d for d in desc
                                     if isinstance(d, str)).strip()
            else:
                desc_text = str(desc).strip()
            cats = rec.get("categories") or []
            if not isinstance(cats, list):
                cats = []
            features = rec.get("features") or []
            if not isinstance(features, list):
                features = []
            details = rec.get("details") or {}
            if not isinstance(details, dict):
                details = {}
            out[asin] = {
                "title": (rec.get("title") or "").strip(),
                "store": (rec.get("store") or "").strip(),
                "categories": [c for c in cats if isinstance(c, str) and c],
                "description": desc_text,
                "features": [str(x) for x in features if x],
                "average_rating": rec.get("average_rating"),
                "details": details,
            }
    return out


def build_item_texts(item_meta: dict[str, dict],
                     item_map_raw_to_new: dict[str, int]) -> dict[int, str]:
    """{new_item_id -> 'Title. By Store. Categories: A, B. <description>. 4.5 stars.'}."""
    texts: dict[int, str] = {}
    for raw_id, new_id in item_map_raw_to_new.items():
        m = item_meta.get(raw_id)
        if m is None:
            texts[new_id] = "Unknown product."
            continue
        parts: list[str] = []
        title = m["title"] or "An untitled product"
        parts.append(title)
        store = m["store"]
        if store:
            parts.append(f"By {store}")
        cats = m["categories"]
        if cats:
            parts.append(f"Categories: {', '.join(cats)}")
        details = m["details"]
        author = details.get("Author") if isinstance(details, dict) else None
        if isinstance(author, str) and author and author != store:
            parts.append(f"Author: {author}")
        desc = m["description"]
        if desc:
            if len(desc) > 400:
                desc = desc[:400].rstrip() + "..."
            parts.append(desc)
        ar = m["average_rating"]
        if ar is not None:
            try:
                parts.append(f"{float(ar):.1f} average stars")
            except (TypeError, ValueError):
                pass
        texts[new_id] = ". ".join(parts) + "."
    return texts


def _categories_for_item(item_meta: dict[str, dict],
                         inv_item_map: dict[int, str]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for new_id, raw_id in inv_item_map.items():
        m = item_meta.get(raw_id)
        out[new_id] = (m["categories"] if m else []) or []
    return out


def build_user_texts(user_map_raw_to_new: dict[str, int],
                     train_seqs: dict[int, list[int]],
                     item_texts: dict[int, str],
                     item_categories: dict[int, list[str]],
                     top_items: int = 5,
                     top_categories: int = 3) -> dict[int, str]:
    """Short profile per user: review count + favorite categories + recent items.

    We don't have a per-user metadata file in Amazon Reviews 2023 (unlike
    Yelp's user.json), so the profile is derived purely from the user's
    training history.
    """
    texts: dict[int, str] = {}
    for u, seq in train_seqs.items():
        cat_counter: Counter[str] = Counter()
        for iid in seq:
            for c in item_categories.get(iid, []):
                cat_counter[c] += 1
        fav = [c for c, _ in cat_counter.most_common(top_categories)]
        fav_str = ", ".join(fav) if fav else "a variety of products"

        recent: list[str] = []
        for iid in seq[-top_items:][::-1]:
            t = item_texts.get(iid, "")
            if t:
                recent.append(t.split(".")[0])
        recent_str = "; ".join(recent) if recent else "various products"

        texts[u] = (
            f"An Amazon shopper with {len(seq)} reviews who frequently "
            f"buys {fav_str}. Recently reviewed: {recent_str}."
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
                                amazon_dir: Path,
                                item_map: dict[str, int],
                                user_map: dict[str, int],
                                train_seqs: dict[int, list[int]],
                                model_name: str,
                                device: str,
                                category: str = DEFAULT_CATEGORY
                                ) -> tuple[np.ndarray, np.ndarray, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = model_name.replace("/", "__")
    cat_tag = category.lower()
    item_cache = cache_dir / f"amazon_{cat_tag}_item_emb_{safe_tag}.npy"
    user_cache = cache_dir / f"amazon_{cat_tag}_user_emb_{safe_tag}.npy"

    if item_cache.exists() and user_cache.exists():
        item_emb = np.load(item_cache)
        user_emb = np.load(user_cache)
        if (item_emb.shape[0] == len(item_map) + 1
                and user_emb.shape[0] == len(user_map) + 1):
            print(f"Loaded cached Amazon-{category} semantic embeddings "
                  f"from {cache_dir}")
            return item_emb, user_emb, int(item_emb.shape[1])
        print("Cached Amazon semantic embeddings have stale shape; rebuilding.")

    keep_ids = set(item_map.keys())
    item_meta = load_item_meta(amazon_dir, category=category,
                               keep_ids=keep_ids)
    item_texts = build_item_texts(item_meta, item_map)
    inv_item_map = {new: raw for raw, new in item_map.items()}
    item_categories = _categories_for_item(item_meta, inv_item_map)
    user_texts = build_user_texts(user_map, train_seqs, item_texts,
                                  item_categories)

    num_items = len(item_map)
    num_users = len(user_map)

    ordered_item_texts = [item_texts.get(i, "Unknown product.")
                          for i in range(1, num_items + 1)]
    ordered_user_texts = [user_texts.get(u, "An Amazon shopper.")
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
    print(f"Cached Amazon-{category} semantic embeddings in {cache_dir}")
    return item_emb, user_emb, dim


def load_amazon_with_semantics(data_dir: Path, cache_dir: Path,
                               model_name: str, device: str,
                               start_year: int = 2018,
                               min_inter: int = 5,
                               category: str = DEFAULT_CATEGORY):
    """One-stop loader. Mirrors `load_yelp_with_semantics` exactly in shape."""
    amazon_dir = ensure_amazon_files(data_dir, category=category)
    sequences, num_users, num_items, user_map, item_map = build_sequences(
        amazon_dir, category=category, start_year=start_year,
        min_inter=min_inter)
    train_seqs, val_targets, test_targets = split_sequences(sequences)

    item_sem, user_sem, sem_dim = prepare_semantic_embeddings(
        data_dir, cache_dir, amazon_dir, item_map, user_map, train_seqs,
        model_name, device, category=category)

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
