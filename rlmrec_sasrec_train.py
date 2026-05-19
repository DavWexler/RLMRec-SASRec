"""End-to-end RLMRec-Con training with a SASRec backbone on MovieLens-1M.

Same history / metrics / plots as pure SASRec, plus alignment-loss components.
Evaluation uses the shared full-rank protocol.
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
import torch

from sasrec_data import SASRecTrainDataset, build_eval_inputs
from eval_full_rank import K_VALUES, build_exclusion_mask, evaluate_full_rank
from sasrec_train import (DATASET_LABELS, bce_with_logits_masked, pick_device,
                          plot_all_val_metrics, plot_loss, plot_test_metrics,
                          plot_val_metric, set_seed, write_history_csv)
from amazon_data import load_amazon_with_semantics
import amazon_data
from rlmrec_data import load_ml1m_with_semantics
from rlmrec_model import RLMRecSASRec, info_nce
from yelp_data import load_yelp_with_semantics


def plot_loss_components(history: list[dict], out_path: Path,
                         dataset: str = "ml1m") -> None:
    keys = [k for k in ("train_loss", "train_cf_loss", "train_item_align",
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
    plt.title(f"RLMRec loss components — {label}")
    plt.grid(alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--cache_dir", type=str, default="data/semantic_cache")
    parser.add_argument("--out_dir", type=str, default="artifacts/rlmrec_sasrec")
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
    parser.add_argument("--proj_head", type=str, default="mlp",
                        choices=["mlp", "linear"],
                        help="Projection-head architecture for both user and "
                             "item heads. 'mlp' = 2-layer w/ GELU (default); "
                             "'linear' = single nn.Linear (ablation).")
    parser.add_argument("--items_per_batch", type=int, default=512,
                        help="Unique items sampled for item-alignment InfoNCE.")
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

    train_ds = SASRecTrainDataset(train_seqs, num_items, args.max_len,
                                  seed=args.seed)

    val_eval_seqs = {u: train_seqs[u] for u in val_targets}
    test_eval_seqs = {u: train_seqs[u] + [val_targets[u]] for u in test_targets}
    val_users, val_inputs = build_eval_inputs(val_eval_seqs, args.max_len)
    test_users, test_inputs = build_eval_inputs(test_eval_seqs, args.max_len)
    val_tgt = torch.tensor([val_targets[u] for u in val_users], dtype=torch.long)
    test_tgt = torch.tensor([test_targets[u] for u in test_users], dtype=torch.long)
    val_mask = build_exclusion_mask([set(train_seqs[u]) for u in val_users], num_items)
    test_mask = build_exclusion_mask(
        [set(train_seqs[u]) | {val_targets[u]} for u in test_users], num_items)
    print(f"Val users: {len(val_users)} | Test users: {len(test_users)} | "
          f"semantic dim: {sem_dim} | full-rank over {num_items} items")

    model = RLMRecSASRec(num_items=num_items, semantic_dim=sem_dim,
                         max_len=args.max_len, hidden_units=args.hidden,
                         num_blocks=args.blocks, num_heads=args.heads,
                         dropout=args.dropout,
                         proj_head=args.proj_head).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params (SASRec+projection heads): {n_params:,}")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                             betas=(0.9, 0.98), weight_decay=args.l2)

    # Precompute a per-user -> id tensor keyed by position in batch dataset.
    # Dataset order is `train_ds.users`.
    user_id_tensor = torch.tensor(train_ds.users, dtype=torch.long)

    history: list[dict] = []
    best_val_ndcg = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        loss_sum = cf_sum = item_sum = user_sum = 0.0
        n_batches = 0

        # We need user_ids per batch. Rebuild a DataLoader that also yields
        # the user id. Simpler: iterate manually by shuffled indices.
        perm = torch.randperm(len(train_ds))
        for bstart in range(0, len(perm), args.batch_size):
            idx = perm[bstart:bstart + args.batch_size]
            batch = [train_ds[int(i)] for i in idx]
            input_seq = torch.stack([b[0] for b in batch]).to(device)
            pos_seq = torch.stack([b[1] for b in batch]).to(device)
            neg_seq = torch.stack([b[2] for b in batch]).to(device)
            batch_user_ids = user_id_tensor[idx].to(device)

            # Full encode (shared between CF head and user alignment).
            hidden = model.encode(input_seq)  # (B, L, H)
            pos_emb = model.item_emb(pos_seq)
            neg_emb = model.item_emb(neg_seq)
            pos_logits = (hidden * pos_emb).sum(dim=-1)
            neg_logits = (hidden * neg_emb).sum(dim=-1)
            cf_loss = bce_with_logits_masked(pos_logits, neg_logits, pos_seq)

            last_hidden = hidden[:, -1, :]
            user_proj = model.project_user(last_hidden)
            user_loss = info_nce(user_proj, user_sem[batch_user_ids],
                                 temperature=args.temperature)

            # Item alignment: sample unique items from this batch's positives.
            flat_items = pos_seq.reshape(-1)
            flat_items = flat_items[flat_items > 0]
            unique_items = torch.unique(flat_items)
            if unique_items.numel() > args.items_per_batch:
                pick = torch.randperm(unique_items.numel(),
                                      device=device)[:args.items_per_batch]
                unique_items = unique_items[pick]
            item_proj = model.project_item(unique_items)
            item_loss = info_nce(item_proj, item_sem[unique_items],
                                 temperature=args.temperature)

            loss = cf_loss + args.lambda_user * user_loss + args.lambda_item * item_loss
            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_sum += float(loss.item())
            cf_sum += float(cf_loss.item())
            user_sum += float(user_loss.item())
            item_sum += float(item_loss.item())
            n_batches += 1

        row: dict = {
            "epoch": epoch,
            "train_loss": round(loss_sum / n_batches, 6),
            "train_cf_loss": round(cf_sum / n_batches, 6),
            "train_item_align": round(item_sum / n_batches, 6),
            "train_user_align": round(user_sum / n_batches, 6),
            "epoch_seconds": round(time.time() - epoch_start, 3),
        }

        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if do_eval:
            val_metrics = evaluate_full_rank(
                model.score_all_items, val_inputs, val_tgt, val_mask, device)
            for k, v in val_metrics.items():
                row[f"val_{k}"] = round(v, 6)
            if val_metrics["NDCG@10"] > best_val_ndcg:
                best_val_ndcg = val_metrics["NDCG@10"]
                best_epoch = epoch
                torch.save(model.state_dict(), out_dir / "rlmrec_sasrec_best.pt")

        history.append(row)
        msg = (f"epoch {epoch:>3} | loss {row['train_loss']:.4f} "
               f"(cf {row['train_cf_loss']:.3f} "
               f"u {row['train_user_align']:.3f} "
               f"i {row['train_item_align']:.3f}) "
               f"| {row['epoch_seconds']:.1f}s")
        if do_eval:
            msg += (f" | val HR@10 {row['val_HR@10']:.4f} "
                    f"NDCG@10 {row['val_NDCG@10']:.4f}")
        print(msg)

        write_history_csv(out_dir / "train_history.csv", history)

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

    if best_epoch > 0:
        print(f"Loading best checkpoint from epoch {best_epoch} "
              f"(val NDCG@10={best_val_ndcg:.4f})")
        model.load_state_dict(torch.load(out_dir / "rlmrec_sasrec_best.pt",
                                         map_location=device))
    test_metrics = evaluate_full_rank(
        model.score_all_items, test_inputs, test_tgt, test_mask, device)
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
