"""SASRec model (Kang & McAuley, 2018) in PyTorch."""

from __future__ import annotations

import torch
import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, H)
        y = x.transpose(-1, -2)
        y = self.dropout1(self.relu(self.conv1(y)))
        y = self.dropout2(self.conv2(y))
        y = y.transpose(-1, -2)
        return y + x


class SASRec(nn.Module):
    def __init__(self, num_items: int, max_len: int = 200, hidden_units: int = 50,
                 num_blocks: int = 2, num_heads: int = 1, dropout: float = 0.2):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.hidden_units = hidden_units

        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_units)
        self.emb_dropout = nn.Dropout(p=dropout)

        self.attn_layernorms = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        self.ffn_layernorms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.attn_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attn_layers.append(
                nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout,
                                      batch_first=True))
            self.ffn_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.ffn_layers.append(PointWiseFeedForward(hidden_units, dropout))
        self.final_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_normal_(m.weight.data)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias.data)

    def encode(self, input_seq: torch.Tensor) -> torch.Tensor:
        """input_seq: (B, L) int -> hidden states (B, L, H)."""
        seqs = self.item_emb(input_seq) * (self.hidden_units ** 0.5)
        positions = torch.arange(input_seq.size(1), device=input_seq.device)
        positions = positions.unsqueeze(0).expand_as(input_seq)
        seqs = seqs + self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        pad_mask = (input_seq == 0)
        seqs = seqs * (~pad_mask).unsqueeze(-1).float()

        L = input_seq.size(1)
        causal_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=input_seq.device),
            diagonal=1,
        )

        for i in range(len(self.attn_layers)):
            q = self.attn_layernorms[i](seqs)
            attn_out, _ = self.attn_layers[i](q, seqs, seqs,
                                              attn_mask=causal_mask,
                                              need_weights=False)
            seqs = q + attn_out
            seqs = seqs * (~pad_mask).unsqueeze(-1).float()

            seqs = self.ffn_layernorms[i](seqs)
            seqs = self.ffn_layers[i](seqs)
            seqs = seqs * (~pad_mask).unsqueeze(-1).float()

        return self.final_layernorm(seqs)

    def forward(self, input_seq: torch.Tensor,
                pos_seq: torch.Tensor,
                neg_seq: torch.Tensor):
        """Return (pos_logits, neg_logits) of shape (B, L) for BCE loss."""
        hidden = self.encode(input_seq)  # (B, L, H)
        pos_emb = self.item_emb(pos_seq)
        neg_emb = self.item_emb(neg_seq)
        pos_logits = (hidden * pos_emb).sum(dim=-1)
        neg_logits = (hidden * neg_emb).sum(dim=-1)
        return pos_logits, neg_logits

    @torch.no_grad()
    def score_all_items(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Score every item for each sequence using the last position's hidden.

        input_seq: (B, L) -> scores (B, num_items+1). Column 0 is the score
        against the padding embedding and the caller must mask it out.
        """
        hidden = self.encode(input_seq)[:, -1, :]
        return hidden @ self.item_emb.weight.t()
