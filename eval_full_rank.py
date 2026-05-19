"""Shared full-rank leave-one-out evaluation.

Used by all three models (SASRec, RLMRec+SASRec, RLMRec+LightGCN) so metrics
are directly comparable. Each model provides a `score_all_fn(inputs) ->
(B, num_items+1)` callable; this module handles masking, ranking, and metric
computation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

K_VALUES = (1, 5, 10, 20)


def _metrics_from_ranks(ranks: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in K_VALUES:
        hit = (ranks < k).astype(np.float64)
        metrics[f"HR@{k}"] = float(hit.mean())
        ndcg = np.where(ranks < k, 1.0 / np.log2(ranks + 2.0), 0.0)
        metrics[f"NDCG@{k}"] = float(ndcg.mean())
    metrics["MRR"] = float((1.0 / (ranks + 1.0)).mean())
    return metrics


def build_exclusion_mask(user_histories: list[set[int]],
                         num_items: int) -> torch.Tensor:
    """mask[i, j] = True means item j should be excluded for eval user i.

    Index 0 is always masked (padding slot).
    """
    n = len(user_histories)
    mask = torch.zeros(n, num_items + 1, dtype=torch.bool)
    mask[:, 0] = True
    for i, hist in enumerate(user_histories):
        if hist:
            idx = torch.tensor(list(hist), dtype=torch.long)
            mask[i, idx] = True
    return mask


def evaluate_full_rank(
    score_all_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    exclusion_mask: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, float]:
    """Score every item for every eval user and compute HR@K / NDCG@K / MRR."""
    n = inputs.size(0)
    ranks = np.empty(n, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            inp = inputs[start:end].to(device)
            tgt = targets[start:end].to(device)
            mask = exclusion_mask[start:end].to(device)

            scores = score_all_fn(inp)
            scores = scores.masked_fill(mask, float("-inf"))
            tgt_scores = scores.gather(1, tgt.view(-1, 1))
            rank = (scores > tgt_scores).sum(dim=1).cpu().numpy()
            ranks[start:end] = rank
    return _metrics_from_ranks(ranks)
