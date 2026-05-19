"""End-to-end RLMRec-Con training with a LightGCN backbone on MovieLens-1M.

Paper-faithful RLMRec recipe (Ren et al., WWW 2024): graph-based CF backbone,
BPR ranking loss, InfoNCE alignment between projected CF embeddings and
frozen LLM-derived semantic embeddings. Evaluation uses the shared full-rank
protocol for direct comparison with the two sequential variants.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from eval_full_rank import K_VALUES, build_exclusion_mask, evaluate_full_rank
from lightgcn_model import LightGCNRLMRec, build_norm_adjacency
from rlmrec_data import load_ml1m_with_semantics
from rlmrec_model import info_nce
from sasrec_train import (DATASET_LABELS, pick_device, plot_all_val_metrics,
                          plot_loss, plot_test_metrics, plot_val_metric,
                          set_seed, write_history_csv)
from yelp_data import load_yelp_with_semantics
from amazon_data import load_amazon_with_semantics
import amazon_data


def plot_loss_components(history: list[dict], out_path: Path,
                         dataset: str = "ml1m") -> None:
    keys = [k for k in ("train_loss", "train_bpr", "train_item_align",
                        "train_user_align") if k in history[0]]
    if not keys:
        return
    plt.figure(figsize=(9, 5))
    for k in keys:
        xs = [h["epoch"] for h in history if h.get(k) is not None]
        ys = [h[k] for h in history if h.get(k) is not None]
        plt.plot(xs, ys, marker="o", linewidth=1.2, label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    label = DATASET_LABELS.get(dataset, dataset)
    plt.title(f"RLMRec-LightGCN loss components — {label}")
    plt.grid(alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor,
             u: torch.Tensor, p: torch.Tensor, n: torch.Tensor,
             l2_reg: float) -> torch.Tensor:
    rank = -F.logsigmoid(pos_score - neg_score).mean()
    reg = 0.5 * (u.pow(2).sum(-1).mean()
                 + p.pow(2).sum(-1).mean()
                 + n.pow(2).sum(-1).mean())
    return rank + l2_reg * reg


def sample_negatives(batch_users_1: np.ndarray,
                     user_pos_set: dict[int, set[int]],
                     num_items: int, rng: np.random.Generator) -> np.ndarray:
    negs = rng.integers(1, num_items + 1, size=batch_users_1.shape[0])
    for j, u in enumerate(batch_users_1):
        u_pos = user_pos_set[int(u)]
        while int(negs[j]) in u_pos:
            negs[j] = rng.integers(1, num_items + 1)
    return negs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--cache_dir", type=str, default="data/semantic_cache")
    parser.add_argument("--out_dir", type=str, default="artifacts/rlmrec_lightgcn")
    parser.add_argument("--model_name", type=str,
                        default="BAAI/bge-large-en-v1.5",
                        help="HF / sentence-transformers model used to encode "
                             "user & item profile texts. Default: "
                             "BGE-large (1024-dim, 335M params, MTEB 64). "
                             "Was E5-Mistral-7B-Instruct (4096-dim, 7B, "
                             "MTEB ~66); originally MiniLM-L6 (384-dim, 22M, "
                             "MTEB 56).")
    parser.add_argument("--lambda_item", type=float, default=0.1)
    parser.add_argument("--lambda_user", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--dataset", type=str, default="ml1m",
                        choices=["ml1m", "yelp", "amazon"])
    parser.add_argument("--yelp_start_year", type=int, default=2018)
    parser.add_argument("--yelp_min_inter", type=int, default=5)
    parser.add_argument("--amazon_start_year", type=int, default=2018)
    parser.add_argument("--amazon_min_inter", type=int, default=5)
    parser.add_argument("--amazon_category", type=str,
                        default=amazon_data.DEFAULT_CATEGORY,
                        help="Amazon Reviews 2023 category to load. "
                             "Default: Books (richest semantic content).")
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device()
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data + semantic embeddings ---
    encoder_device = "mps" if device.type == "mps" else ("cuda" if device.type == "cuda" else "cpu")
    if args.dataset == "ml1m":
        bundle = load_ml1m_with_semantics(
            Path(args.data_dir), Path(args.cache_dir), args.model_name,
            device=encoder_device)
    elif args.dataset == "yelp":
        bundle = load_yelp_with_semantics(
            Path(args.data_dir), Path(args.cache_dir), args.model_name,
            device=encoder_device, start_year=args.yelp_start_year,
            min_inter=args.yelp_min_inter)
    else:  # amazon
        bundle = load_amazon_with_semantics(
            Path(args.data_dir), Path(args.cache_dir), args.model_name,
            device=encoder_device, start_year=args.amazon_start_year,
            min_inter=args.amazon_min_inter, category=args.amazon_category)
    train_seqs = bundle["train_seqs"]
    val_targets = bundle["val_targets"]
    test_targets = bundle["test_targets"]
    num_users = bundle["num_users"]
    num_items = bundle["num_items"]
    sem_dim = bundle["sem_dim"]
    item_sem = bundle["item_sem"].to(device)
    user_sem = bundle["user_sem"].to(device)

    # Flat (user, pos_item) pairs for BPR training. user/item are 1-indexed.
    pairs = np.fromiter(
        ((u, i) for u, seq in train_seqs.items() for i in seq),
        dtype=np.dtype((np.int64, 2)),
    )
    user_pos_set = {u: set(seq) for u, seq in train_seqs.items()}
    print(f"Train interactions: {len(pairs):,}")

    # --- graph construction ---
    edge_src, edge_dst, norm_coef = build_norm_adjacency(
        train_seqs, num_users, num_items)
    print(f"Edges (directed): {edge_src.numel():,} over "
          f"{num_users + num_items:,} nodes")

    # --- model ---
    model = LightGCNRLMRec(num_users=num_users, num_items=num_items,
                           dim=args.dim, num_layers=args.num_layers,
                           edge_src=edge_src, edge_dst=edge_dst,
                           norm_coef=norm_coef, semantic_dim=sem_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params (LightGCN+projection heads): {n_params:,}")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- eval setup (full-rank) ---
    val_user_list = list(val_targets.keys())
    test_user_list = list(test_targets.keys())
    val_inputs = torch.tensor(val_user_list, dtype=torch.long)
    test_inputs = torch.tensor(test_user_list, dtype=torch.long)
    val_tgt = torch.tensor([val_targets[u] for u in val_user_list], dtype=torch.long)
    test_tgt = torch.tensor([test_targets[u] for u in test_user_list], dtype=torch.long)
    val_mask = build_exclusion_mask(
        [set(train_seqs[u]) for u in val_user_list], num_items)
    test_mask = build_exclusion_mask(
        [set(train_seqs[u]) | {val_targets[u]} for u in test_user_list], num_items)
    print(f"Val users: {len(val_user_list)} | Test users: {len(test_user_list)} | "
          f"semantic dim: {sem_dim} | full-rank over {num_items} items")

    rng = np.random.default_rng(args.seed)
    history: list[dict] = []
    best_val_ndcg = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        loss_sum = bpr_sum = item_sum = user_sum = 0.0
        n_batches = 0

        perm = rng.permutation(len(pairs))
        for bstart in range(0, len(pairs), args.batch_size):
            idx = perm[bstart:bstart + args.batch_size]
            batch = pairs[idx]
            users_1 = batch[:, 0]
            pos_1 = batch[:, 1]
            negs_1 = sample_negatives(users_1, user_pos_set, num_items, rng)

            u_idx = torch.from_numpy(users_1 - 1).to(device)
            p_idx = torch.from_numpy(pos_1 - 1).to(device)
            n_idx = torch.from_numpy(negs_1 - 1).to(device)

            pos_score, neg_score, u_vec, p_vec, _ = model(u_idx, p_idx, n_idx)
            # L2 reg on layer-0 embeddings only (original LightGCN convention).
            bpr = bpr_loss(pos_score, neg_score,
                           model.backbone.user_emb(u_idx),
                           model.backbone.item_emb(p_idx),
                           model.backbone.item_emb(n_idx),
                           args.l2)

            user_sem_batch = user_sem[torch.from_numpy(users_1).to(device)]
            item_sem_batch = item_sem[torch.from_numpy(pos_1).to(device)]
            align_u = info_nce(model.project_user(u_vec), user_sem_batch,
                               temperature=args.temperature)
            align_i = info_nce(model.project_item(p_vec), item_sem_batch,
                               temperature=args.temperature)

            loss = bpr + args.lambda_user * align_u + args.lambda_item * align_i
            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_sum += float(loss.item())
            bpr_sum += float(bpr.item())
            user_sum += float(align_u.item())
            item_sum += float(align_i.item())
            n_batches += 1

        row: dict = {
            "epoch": epoch,
            "train_loss": round(loss_sum / n_batches, 6),
            "train_bpr": round(bpr_sum / n_batches, 6),
            "train_item_align": round(item_sum / n_batches, 6),
            "train_user_align": round(user_sum / n_batches, 6),
            "epoch_seconds": round(time.time() - epoch_start, 3),
        }

        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if do_eval:
            val_metrics = evaluate_full_rank(
                model.score_all_items, val_inputs, val_tgt, val_mask, device,
                batch_size=args.eval_batch_size)
            for k, v in val_metrics.items():
                row[f"val_{k}"] = round(v, 6)
            if val_metrics["NDCG@10"] > best_val_ndcg:
                best_val_ndcg = val_metrics["NDCG@10"]
                best_epoch = epoch
                torch.save(model.state_dict(), out_dir / "rlmrec_lightgcn_best.pt")

        history.append(row)
        msg = (f"epoch {epoch:>3} | loss {row['train_loss']:.4f} "
               f"(bpr {row['train_bpr']:.3f} "
               f"u {row['train_user_align']:.3f} "
               f"i {row['train_item_align']:.3f}) "
               f"| {row['epoch_seconds']:.1f}s")
        if do_eval:
            msg += (f" | val HR@10 {row['val_HR@10']:.4f} "
                    f"NDCG@10 {row['val_NDCG@10']:.4f}")
        print(msg)

        write_history_csv(out_dir / "train_history.csv", history)

    # --- plots ---
    plot_loss(history, out_dir / "loss_curve.png", dataset=args.dataset)
    plot_loss_components(history, out_dir / "loss_components.png",
                         dataset=args.dataset)
    for k in K_VALUES:
        plot_val_metric(history, f"HR@{k}", out_dir / f"val_HR@{k}.png",
                        dataset=args.dataset)
        plot_val_metric(history, f"NDCG@{k}", out_dir / f"val_NDCG@{k}.png",
                        dataset=args.dataset)
    plot_val_metric(history, "MRR", out_dir / "val_MRR.png",
                    dataset=args.dataset)
    plot_all_val_metrics(history, out_dir / "val_metrics_all.png",
                         dataset=args.dataset)

    # --- final test ---
    if best_epoch > 0:
        print(f"Loading best checkpoint from epoch {best_epoch} "
              f"(val NDCG@10={best_val_ndcg:.4f})")
        model.load_state_dict(
            torch.load(out_dir / "rlmrec_lightgcn_best.pt", map_location=device))
    test_metrics = evaluate_full_rank(
        model.score_all_items, test_inputs, test_tgt, test_mask, device,
        batch_size=args.eval_batch_size)
    print("Test metrics:")
    for k, v in sorted(test_metrics.items()):
        print(f"  {k:>10}: {v:.4f}")

    with open(out_dir / "test_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in sorted(test_metrics.items()):
            w.writerow([k, f"{v:.6f}"])

    plot_test_metrics(test_metrics, out_dir / "test_metrics.png",
                      dataset=args.dataset)

    meta = {
        "args": vars(args),
        "dataset": args.dataset,
        "device": str(device),
        "num_users": num_users,
        "num_items": num_items,
        "semantic_dim": sem_dim,
        "model_params": n_params,
        "semantic_model": args.model_name,
        "best_val_epoch": best_epoch,
        "best_val_NDCG@10": best_val_ndcg,
        "test_metrics": test_metrics,
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
