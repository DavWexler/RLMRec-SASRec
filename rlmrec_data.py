"""RLMRec data utilities.

Reuses SASRec's MovieLens-1M pipeline, then builds natural-language profiles
for items (title + genres) and users (demographics + genre summary + recent
titles) and encodes them with a small frozen sentence-transformer.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch

from sasrec_data import download_ml1m

AGE_LABELS = {
    1: "under 18", 18: "18-24", 25: "25-34", 35: "35-44",
    45: "45-49", 50: "50-55", 56: "56+",
}

OCCUPATION_LABELS = {
    0: "person with unspecified occupation", 1: "academic or educator",
    2: "artist", 3: "clerical/admin worker", 4: "college or graduate student",
    5: "customer service worker", 6: "doctor or healthcare worker",
    7: "executive or manager", 8: "farmer", 9: "homemaker",
    10: "K-12 student", 11: "lawyer", 12: "programmer", 13: "retiree",
    14: "sales or marketing professional", 15: "scientist",
    16: "self-employed person", 17: "technician or engineer",
    18: "tradesman or craftsman", 19: "unemployed person", 20: "writer",
}


def load_movie_texts(movies_path: Path,
                     item_map_raw_to_new: dict[int, int]) -> dict[int, str]:
    """Return {new_item_id -> 'Title (Year). Genres: A, B, C.'}."""
    texts: dict[int, str] = {}
    with open(movies_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 3:
                continue
            raw_id = int(parts[0])
            if raw_id not in item_map_raw_to_new:
                continue
            title = parts[1].strip()
            genres = parts[2].replace("|", ", ")
            texts[item_map_raw_to_new[raw_id]] = f"{title}. Genres: {genres}."
    return texts


def load_user_meta(users_path: Path,
                   user_map_raw_to_new: dict[int, int]) -> dict[int, dict]:
    meta: dict[int, dict] = {}
    with open(users_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 5:
                continue
            raw_id = int(parts[0])
            if raw_id not in user_map_raw_to_new:
                continue
            meta[user_map_raw_to_new[raw_id]] = {
                "gender": "female" if parts[1].upper() == "F" else "male",
                "age": AGE_LABELS.get(int(parts[2]), "unspecified age"),
                "occupation": OCCUPATION_LABELS.get(int(parts[3]), "person"),
            }
    return meta


def build_user_texts(user_meta: dict[int, dict],
                     train_seqs: dict[int, list[int]],
                     item_texts: dict[int, str],
                     item_genres_map: dict[int, list[str]],
                     top_movies: int = 5,
                     top_genres: int = 3) -> dict[int, str]:
    """Build a short profile text per user."""
    texts: dict[int, str] = {}
    for u, seq in train_seqs.items():
        m = user_meta.get(u, {
            "gender": "person", "age": "unspecified age",
            "occupation": "person with unspecified occupation",
        })
        genre_counter: Counter[str] = Counter()
        for iid in seq:
            for g in item_genres_map.get(iid, []):
                genre_counter[g] += 1
        fav = [g for g, _ in genre_counter.most_common(top_genres)]
        fav_str = ", ".join(fav) if fav else "various genres"

        titles = []
        for iid in seq[-top_movies:][::-1]:
            t = item_texts.get(iid, "")
            if t:
                titles.append(t.split(". Genres:")[0])
        titles_str = "; ".join(titles) if titles else "various films"

        texts[u] = (
            f"A {m['age']} {m['gender']} {m['occupation']} who enjoys "
            f"{fav_str} movies. Recently watched: {titles_str}."
        )
    return texts


def parse_item_genres(movies_path: Path,
                      item_map_raw_to_new: dict[int, int]) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    with open(movies_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 3:
                continue
            raw_id = int(parts[0])
            if raw_id not in item_map_raw_to_new:
                continue
            result[item_map_raw_to_new[raw_id]] = parts[2].split("|")
    return result


def encode_semantic(texts: list[str], model_name: str, device: str,
                    batch_size: int = 128) -> np.ndarray:
    """Encode texts to a (N, D) L2-normalized array with sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    print(f"Encoding {len(texts)} texts with {model_name} on {device} ...")
    emb = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                       show_progress_bar=True, normalize_embeddings=True)
    return emb.astype(np.float32)


def prepare_semantic_embeddings(data_dir: Path, cache_dir: Path,
                                item_map: dict[int, int],
                                user_map: dict[int, int],
                                train_seqs: dict[int, list[int]],
                                model_name: str,
                                device: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Build (or load cached) item and user semantic embedding matrices.

    Returns (item_emb, user_emb, dim) where:
      - item_emb[i] is the L2-normalized embedding of item id i (i >= 1)
      - user_emb[u] is the embedding of user id u (u >= 1)
      - index 0 is a zero row (reserved for padding / missing).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = model_name.replace("/", "__")
    item_cache = cache_dir / f"item_emb_{safe_tag}.npy"
    user_cache = cache_dir / f"user_emb_{safe_tag}.npy"

    if item_cache.exists() and user_cache.exists():
        item_emb = np.load(item_cache)
        user_emb = np.load(user_cache)
        print(f"Loaded cached semantic embeddings from {cache_dir}")
        return item_emb, user_emb, int(item_emb.shape[1])

    movies_path = data_dir / "ml-1m" / "movies.dat"
    users_path = data_dir / "ml-1m" / "users.dat"

    item_texts = load_movie_texts(movies_path, item_map)
    item_genres = parse_item_genres(movies_path, item_map)
    user_meta = load_user_meta(users_path, user_map)
    user_texts = build_user_texts(user_meta, train_seqs, item_texts, item_genres)

    num_items = len(item_map)
    num_users = len(user_map)

    ordered_item_texts = [
        item_texts.get(i, "Unknown movie.") for i in range(1, num_items + 1)
    ]
    ordered_user_texts = [
        user_texts.get(u, "A MovieLens user.") for u in range(1, num_users + 1)
    ]

    item_arr = encode_semantic(ordered_item_texts, model_name, device)
    user_arr = encode_semantic(ordered_user_texts, model_name, device)

    dim = item_arr.shape[1]
    item_emb = np.zeros((num_items + 1, dim), dtype=np.float32)
    item_emb[1:] = item_arr
    user_emb = np.zeros((num_users + 1, dim), dtype=np.float32)
    user_emb[1:] = user_arr

    np.save(item_cache, item_emb)
    np.save(user_cache, user_emb)
    print(f"Cached semantic embeddings in {cache_dir}")
    return item_emb, user_emb, dim


def load_ml1m_with_semantics(data_dir: Path, cache_dir: Path, model_name: str,
                             device: str):
    """One-stop loader used by both RLMRec training scripts."""
    from sasrec_data import build_sequences, split_sequences

    ratings_path = download_ml1m(data_dir)
    sequences, num_users, num_items, user_map, item_map = build_sequences(ratings_path)
    train_seqs, val_targets, test_targets = split_sequences(sequences)

    item_sem, user_sem, sem_dim = prepare_semantic_embeddings(
        data_dir, cache_dir, item_map, user_map, train_seqs, model_name, device)

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
