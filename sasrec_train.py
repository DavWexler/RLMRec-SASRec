"""End-to-end SASRec training on MovieLens-1M.

Downloads the dataset, trains the model, saves per-epoch history CSV,
plots loss + all validation metrics, evaluates on the held-out test item,
and writes test metrics CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from sasrec_data import (SASRecTrainDataset, build_eval_inputs, build_sequences,
                         download_ml1m, split_sequences)
from eval_full_rank import (K_VALUES, build_exclusion_mask, evaluate_full_rank)
from sasrec_model import SASRec
import amazon_data
import yelp_data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def bce_with_logits_masked(pos_logits: torch.Tensor,
                           neg_logits: torch.Tensor,
                           pos_seq: torch.Tensor) -> torch.Tensor:
    mask = (pos_seq != 0).float()
    pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pos_logits, torch.ones_like(pos_logits), reduction="none")
    neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        neg_logits, torch.zeros_like(neg_logits), reduction="none")
    loss = ((pos_loss + neg_loss) * mask).sum() / mask.sum().clamp(min=1)
    return loss


def write_history_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    # put 'epoch' first
    if "epoch" in fieldnames:
        fieldnames = ["epoch"] + [f for f in fieldnames if f != "epoch"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


DATASET_LABELS = {"ml1m": "MovieLens-1M", "yelp": "Yelp",
                  "amazon": "Amazon-Books"}


def _dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def plot_loss(history: list[dict], out_path: Path,
              dataset: str = "ml1m") -> None:
    epochs = [h["epoch"] for h in history]
    loss = [h["train_loss"] for h in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, marker="o", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss (BCE)")
    plt.title(f"SASRec training loss — {_dataset_label(dataset)}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_val_metric(history: list[dict], metric: str, out_path: Path,
                    dataset: str = "ml1m") -> None:
    ep = [h["epoch"] for h in history if h.get(f"val_{metric}") is not None]
    v = [h[f"val_{metric}"] for h in history if h.get(f"val_{metric}") is not None]
    if not ep:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(ep, v, marker="o", linewidth=1.5, color="C2")
    plt.xlabel("Epoch")
    plt.ylabel(f"Validation {metric}")
    plt.title(f"SASRec validation {metric} — {_dataset_label(dataset)}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_all_val_metrics(history: list[dict], out_path: Path,
                         dataset: str = "ml1m") -> None:
    metric_keys = sorted({k for row in history for k in row
                          if k.startswith("val_") and row.get(k) is not None})
    if not metric_keys:
        return
    plt.figure(figsize=(10, 6))
    for mk in metric_keys:
        ep = [h["epoch"] for h in history if h.get(mk) is not None]
        v = [h[mk] for h in history if h.get(mk) is not None]
        plt.plot(ep, v, marker="o", linewidth=1.2, label=mk.replace("val_", ""))
    plt.xlabel("Epoch")
    plt.ylabel("Validation metric")
    plt.title(f"SASRec validation metrics — {_dataset_label(dataset)}")
    plt.grid(alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_test_metrics(metrics: dict[str, float], out_path: Path,
                      dataset: str = "ml1m") -> None:
    items = sorted(metrics.items(), key=lambda kv: kv[0])
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color="C0")
    plt.ylabel("Score")
    plt.title(f"SASRec test metrics — {_dataset_label(dataset)}")
    plt.ylim(0, max(values) * 1.15 + 1e-6)
    plt.xticks(rotation=30, ha="right")
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=8)
    plt.grid(axis="y", alpha=0.3)
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
    parser.add_argument("--out_dir", type=str, default="artifacts/sasrec")
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
    data_dir = Path(args.data_dir)

    # --- data ---
    if args.dataset == "ml1m":
        ratings_path = download_ml1m(data_dir)
        sequences, num_users, num_items, _, _ = build_sequences(ratings_path)
        train_seqs, val_targets, test_targets = split_sequences(sequences)
    elif args.dataset == "yelp":
        yelp_dir = yelp_data.ensure_yelp_files(data_dir)
        sequences, num_users, num_items, _, _ = yelp_data.build_sequences(
            yelp_dir, start_year=args.yelp_start_year,
            min_inter=args.yelp_min_inter)
        train_seqs, val_targets, test_targets = yelp_data.split_sequences(
            sequences)
    else:  # amazon
        amazon_dir = amazon_data.ensure_amazon_files(
            data_dir, category=args.amazon_category)
        sequences, num_users, num_items, _, _ = amazon_data.build_sequences(
            amazon_dir, category=args.amazon_category,
            start_year=args.amazon_start_year,
            min_inter=args.amazon_min_inter)
        train_seqs, val_targets, test_targets = amazon_data.split_sequences(
            sequences)

    train_ds = SASRecTrainDataset(train_seqs, num_items, args.max_len, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)

    # full-rank eval inputs (input = train history; for test also include val item)
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
          f"full-rank over {num_items} items")

    # --- model ---
    model = SASRec(num_items, max_len=args.max_len, hidden_units=args.hidden,
                   num_blocks=args.blocks, num_heads=args.heads,
                   dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                             betas=(0.9, 0.98), weight_decay=args.l2)

    # --- training loop ---
    history: list[dict] = []
    best_val_ndcg = -1.0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        loss_sum, n_batches = 0.0, 0
        for input_seq, pos_seq, neg_seq in train_loader:
            input_seq = input_seq.to(device)
            pos_seq = pos_seq.to(device)
            neg_seq = neg_seq.to(device)
            pos_logits, neg_logits = model(input_seq, pos_seq, neg_seq)
            loss = bce_with_logits_masked(pos_logits, neg_logits, pos_seq)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_sum += float(loss.item())
            n_batches += 1

        avg_loss = loss_sum / max(1, n_batches)
        row: dict = {"epoch": epoch, "train_loss": round(avg_loss, 6),
                     "epoch_seconds": round(time.time() - epoch_start, 3)}

        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if do_eval:
            val_metrics = evaluate_full_rank(
                model.score_all_items, val_inputs, val_tgt, val_mask, device)
            for k, v in val_metrics.items():
                row[f"val_{k}"] = round(v, 6)
            if val_metrics["NDCG@10"] > best_val_ndcg:
                best_val_ndcg = val_metrics["NDCG@10"]
                best_epoch = epoch
                torch.save(model.state_dict(), out_dir / "sasrec_best.pt")

        history.append(row)
        msg = f"epoch {epoch:>3} | loss {avg_loss:.4f} | {row['epoch_seconds']:.1f}s"
        if do_eval:
            msg += (f" | val HR@10 {row['val_HR@10']:.4f} "
                    f"NDCG@10 {row['val_NDCG@10']:.4f}")
        print(msg)

        # flush history each epoch so it's safe to interrupt
        write_history_csv(out_dir / "train_history.csv", history)

    # --- plots during training history ---
    plot_loss(history, out_dir / "loss_curve.png", dataset=args.dataset)
    for k in K_VALUES:
        plot_val_metric(history, f"HR@{k}", out_dir / f"val_HR@{k}.png",
                        dataset=args.dataset)
        plot_val_metric(history, f"NDCG@{k}", out_dir / f"val_NDCG@{k}.png",
                        dataset=args.dataset)
    plot_val_metric(history, "MRR", out_dir / "val_MRR.png",
                    dataset=args.dataset)
    plot_all_val_metrics(history, out_dir / "val_metrics_all.png",
                         dataset=args.dataset)

    # --- test on best checkpoint ---
    if best_epoch > 0:
        print(f"Loading best checkpoint from epoch {best_epoch} "
              f"(val NDCG@10={best_val_ndcg:.4f})")
        model.load_state_dict(torch.load(out_dir / "sasrec_best.pt",
                                         map_location=device))
    test_metrics = evaluate_full_rank(
        model.score_all_items, test_inputs, test_tgt, test_mask, device)
    print("Test metrics:")
    for k, v in sorted(test_metrics.items()):
        print(f"  {k:>10}: {v:.4f}")

    # save test metrics CSV
    with open(out_dir / "test_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in sorted(test_metrics.items()):
            w.writerow([k, f"{v:.6f}"])

    plot_test_metrics(test_metrics, out_dir / "test_metrics.png",
                      dataset=args.dataset)

    # run metadata
    meta = {
        "args": vars(args),
        "dataset": args.dataset,
        "device": str(device),
        "num_users": num_users,
        "num_items": num_items,
        "model_params": n_params,
        "best_val_epoch": best_epoch,
        "best_val_NDCG@10": best_val_ndcg,
        "test_metrics": test_metrics,
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
