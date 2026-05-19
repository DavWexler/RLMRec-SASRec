"""RLMRec-Con on a SASRec backbone.

Two projection heads map CF embeddings (item and user) into the semantic
space produced by a frozen sentence transformer. InfoNCE alignment
regularizes the CF space toward the semantic space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sasrec_model import SASRec


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int | None = None,
                 *, kind: str = "mlp"):
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(in_dim, out_dim)
        elif kind == "mlp":
            hidden = hidden or max(in_dim, out_dim)
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_dim),
            )
        else:
            raise ValueError(f"Unknown projection-head kind: {kind!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RLMRecSASRec(nn.Module):
    def __init__(self, num_items: int, semantic_dim: int, *,
                 max_len: int = 200, hidden_units: int = 50, num_blocks: int = 2,
                 num_heads: int = 1, dropout: float = 0.2,
                 proj_head: str = "mlp"):
        super().__init__()
        self.backbone = SASRec(num_items=num_items, max_len=max_len,
                               hidden_units=hidden_units, num_blocks=num_blocks,
                               num_heads=num_heads, dropout=dropout)
        self.item_proj = ProjectionHead(hidden_units, semantic_dim, kind=proj_head)
        self.user_proj = ProjectionHead(hidden_units, semantic_dim, kind=proj_head)

    @property
    def item_emb(self) -> nn.Embedding:
        return self.backbone.item_emb

    def forward(self, input_seq, pos_seq, neg_seq):
        return self.backbone(input_seq, pos_seq, neg_seq)

    def encode(self, input_seq: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(input_seq)

    @torch.no_grad()
    def score_all_items(self, input_seq: torch.Tensor) -> torch.Tensor:
        return self.backbone.score_all_items(input_seq)

    def project_user(self, last_hidden: torch.Tensor) -> torch.Tensor:
        return self.user_proj(last_hidden)

    def project_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_proj(self.item_emb(item_ids))


def info_nce(projected: torch.Tensor, target: torch.Tensor,
             temperature: float = 0.1) -> torch.Tensor:
    """Symmetric InfoNCE between two same-shape L2-normalized batches."""
    q = F.normalize(projected, dim=-1)
    k = F.normalize(target, dim=-1)
    logits = q @ k.t() / temperature
    labels = torch.arange(q.size(0), device=q.device)
    return 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits.t(), labels))
