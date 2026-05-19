"""MovieLens-1M download + sequence preprocessing for SASRec."""

from __future__ import annotations

import io
import random
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_ml1m(data_dir: Path) -> Path:
    """Download and extract MovieLens-1M if missing. Returns path to ratings.dat."""
    data_dir.mkdir(parents=True, exist_ok=True)
    ratings_path = data_dir / "ml-1m" / "ratings.dat"
    if ratings_path.exists():
        return ratings_path
    print(f"Downloading MovieLens-1M from {ML1M_URL} ...")
    with urllib.request.urlopen(ML1M_URL) as resp:
        buf = io.BytesIO(resp.read())
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(data_dir)
    assert ratings_path.exists(), f"Expected {ratings_path} after extraction"
    print(f"Extracted to {ratings_path.parent}")
    return ratings_path


def build_sequences(ratings_path: Path):
    """Group ratings by user, sort by timestamp, remap item IDs to [1, num_items].

    Returns (sequences, num_users, num_items, user_map, item_map). Index 0 is
    reserved as a padding slot for both users and items.
    """
    raw = defaultdict(list)
    with open(ratings_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 4:
                continue
            uid, mid, _rating, ts = parts[:4]
            raw[int(uid)].append((int(ts), int(mid)))

    user_ids = sorted(raw.keys())
    user_map = {u: i + 1 for i, u in enumerate(user_ids)}

    all_items = sorted({m for events in raw.values() for _, m in events})
    item_map = {m: i + 1 for i, m in enumerate(all_items)}

    sequences: dict[int, list[int]] = {}
    for uid in user_ids:
        events = sorted(raw[uid], key=lambda x: x[0])
        seq = [item_map[m] for _, m in events]
        if len(seq) < 5:
            continue
        sequences[user_map[uid]] = seq

    num_users = len(user_map)
    num_items = len(all_items)
    print(f"Users: {num_users} | Items: {num_items} | "
          f"Interactions: {sum(len(s) for s in sequences.values())}")
    return sequences, num_users, num_items, user_map, item_map


def split_sequences(sequences: dict[int, list[int]]):
    """Leave-one-out split: last -> test, 2nd-to-last -> val, rest -> train."""
    train, val, test = {}, {}, {}
    for u, seq in sequences.items():
        train[u] = seq[:-2]
        val[u] = seq[-2]
        test[u] = seq[-1]
    return train, val, test


class SASRecTrainDataset(Dataset):
    """Yields (input_seq, pos_seq, neg_seq) of length `max_len` for each user."""

    def __init__(self, train_seqs: dict[int, list[int]], num_items: int,
                 max_len: int, seed: int = 0):
        self.users = list(train_seqs.keys())
        self.train_seqs = train_seqs
        self.num_items = num_items
        self.max_len = max_len
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.users)

    def _sample_neg(self, user_items: set[int]) -> int:
        while True:
            n = self.rng.randint(1, self.num_items)
            if n not in user_items:
                return n

    def __getitem__(self, idx: int):
        u = self.users[idx]
        seq = self.train_seqs[u]
        user_items = set(seq)

        input_seq = np.zeros(self.max_len, dtype=np.int64)
        pos_seq = np.zeros(self.max_len, dtype=np.int64)
        neg_seq = np.zeros(self.max_len, dtype=np.int64)

        nxt = seq[-1]
        i = self.max_len - 1
        for item in reversed(seq[:-1]):
            input_seq[i] = item
            pos_seq[i] = nxt
            neg_seq[i] = self._sample_neg(user_items)
            nxt = item
            i -= 1
            if i < 0:
                break
        return (torch.from_numpy(input_seq),
                torch.from_numpy(pos_seq),
                torch.from_numpy(neg_seq))


def build_eval_inputs(eval_seqs: dict[int, list[int]], max_len: int):
    """Build full-rank eval inputs for sequential recommenders.

    Returns (users_ordered, input_tensor) where input_tensor has shape
    (N, max_len) and each row is the user's left-padded history.
    """
    users = list(eval_seqs.keys())
    arr = np.zeros((len(users), max_len), dtype=np.int64)
    for b, u in enumerate(users):
        seq = eval_seqs[u]
        i = max_len - 1
        for item in reversed(seq):
            arr[b, i] = item
            i -= 1
            if i < 0:
                break
    return users, torch.from_numpy(arr)
