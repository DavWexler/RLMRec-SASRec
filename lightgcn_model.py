"""LightGCN (He et al., SIGIR 2020) + RLMRec-Con wrapper on the same backbone.

Propagation uses edge-based `index_add_` so the model runs on CPU, CUDA, or MPS
without relying on sparse matmul (MPS sparse support is limited).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rlmrec_model import ProjectionHead


class LightGCN(nn.Module):
    """Symmetric-normalized graph propagation on the user-item bipartite graph.

    Users occupy rows [0, num_users), items occupy [num_users, num_users+num_items)
    in a single embedding table E. After `num_layers` propagation steps we
    average the per-layer embeddings and split back into user/item views.
    """

    def __init__(self, num_users: int, num_items: int, dim: int,
                 num_layers: int, edge_src: torch.Tensor,
                 edge_dst: torch.Tensor, norm_coef: torch.Tensor):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.num_layers = num_layers

        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

        # Registered as buffers so .to(device) moves them alongside parameters.
        self.register_buffer("edge_src", edge_src)
        self.register_buffer("edge_dst", edge_dst)
        self.register_buffer("norm_coef", norm_coef)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        E = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        layers = [E]
        for _ in range(self.num_layers):
            msg = E[self.edge_src] * self.norm_coef.unsqueeze(-1)
            new_E = torch.zeros_like(E)
            new_E.index_add_(0, self.edge_dst, msg)
            E = new_E
            layers.append(E)
        final = torch.stack(layers, dim=0).mean(dim=0)
        return final[: self.num_users], final[self.num_users:]

    def forward(self, user_idx: torch.Tensor, pos_idx: torch.Tensor,
                neg_idx: torch.Tensor):
        """All indices are 0-indexed (internal). Returns (pos_score, neg_score,
        user_vec, pos_vec, neg_vec) where vecs are the propagated embeddings."""
        user_emb, item_emb = self.propagate()
        u = user_emb[user_idx]
        p = item_emb[pos_idx]
        n = item_emb[neg_idx]
        return (u * p).sum(-1), (u * n).sum(-1), u, p, n

    @torch.no_grad()
    def score_all_items(self, user_ids_1indexed: torch.Tensor) -> torch.Tensor:
        """Return (B, num_items + 1). Column 0 is -inf to match SASRec's layout."""
        user_emb, item_emb = self.propagate()
        user_vec = user_emb[user_ids_1indexed - 1]
        scores = user_vec @ item_emb.t()
        pad = torch.full((scores.size(0), 1), float("-inf"), device=scores.device)
        return torch.cat([pad, scores], dim=1)


class LightGCNRLMRec(nn.Module):
    """LightGCN + two projection heads (user, item) aligned via InfoNCE."""

    def __init__(self, num_users: int, num_items: int, dim: int,
                 num_layers: int, edge_src: torch.Tensor,
                 edge_dst: torch.Tensor, norm_coef: torch.Tensor,
                 semantic_dim: int):
        super().__init__()
        self.backbone = LightGCN(num_users, num_items, dim, num_layers,
                                 edge_src, edge_dst, norm_coef)
        self.user_proj = ProjectionHead(dim, semantic_dim)
        self.item_proj = ProjectionHead(dim, semantic_dim)

    def propagate(self):
        return self.backbone.propagate()

    def forward(self, user_idx, pos_idx, neg_idx):
        return self.backbone(user_idx, pos_idx, neg_idx)

    @torch.no_grad()
    def score_all_items(self, user_ids_1indexed: torch.Tensor) -> torch.Tensor:
        return self.backbone.score_all_items(user_ids_1indexed)

    def project_user(self, u_vec: torch.Tensor) -> torch.Tensor:
        return self.user_proj(u_vec)

    def project_item(self, i_vec: torch.Tensor) -> torch.Tensor:
        return self.item_proj(i_vec)


def build_norm_adjacency(train_seqs: dict[int, list[int]], num_users: int,
                         num_items: int):
    """Build (edge_src, edge_dst, norm_coef) for the symmetric-normalized graph.

    The adjacency is undirected: each (user u, item i) interaction contributes
    both (u -> i) and (i -> u) edges so propagation `E'[dst] += norm * E[src]`
    moves information both ways. `norm_coef = 1/sqrt(deg_src * deg_dst)`.

    Users are placed at internal indices [0, num_users) (user_id - 1), items at
    [num_users, num_users + num_items) (num_users + item_id - 1).
    """
    src_list: list[int] = []
    dst_list: list[int] = []
    for u, seq in train_seqs.items():
        u_node = u - 1
        for i in seq:
            i_node = num_users + (i - 1)
            src_list.append(u_node); dst_list.append(i_node)
            src_list.append(i_node); dst_list.append(u_node)

    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)

    total_nodes = num_users + num_items
    deg = torch.zeros(total_nodes, dtype=torch.float32)
    deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    deg_inv_sqrt = deg.clamp(min=1).pow(-0.5)

    norm_coef = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    return src, dst, norm_coef
