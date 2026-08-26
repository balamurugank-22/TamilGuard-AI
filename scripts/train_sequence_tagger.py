"""
Train CharCNN + FastText + BiLSTM + CRF for BIO abusive-span tagging.

The trainer expects:
  - weak train rows: data/processed/train_weak_bio.csv
  - gold dev rows:  data/gold/gold_dev_tokens.csv

Use scripts/build_gold_split.py and scripts/build_train_split.py first if those
files are not present.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.sequence_tagger import (  # noqa: E402
    ID_TO_TAG,
    CharFastTextBiLSTMCRF,
    BIOTaggingDataset,
    bio_collate_fn,
    build_fasttext_embedding_matrix,
    build_vocabs,
    compute_class_weights,
    load_bio_sequences,
    save_vocab_bundle,
)

DEFAULT_TRAIN_PATH = ROOT / "data" / "processed" / "train_weak_bio.csv"
DEFAULT_DEV_PATH = ROOT / "data" / "gold" / "gold_dev_tokens.csv"
DEFAULT_FASTTEXT_PATH = ROOT / "embeddings" / "fasttext_tamil_tanglish.bin"
DEFAULT_RUN_DIR = ROOT / "models" / "checkpoints" / "sequence_tagger"


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    dev_loss: float
    accuracy: float
    macro_f1: float
    abuse_macro_f1: float
    abuse_micro_precision: float
    abuse_micro_recall: float
    abuse_micro_f1: float
    dev_tokens: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def classification_metrics(
    gold: list[int],
    pred: list[int],
    *,
    id_to_tag: dict[int, str],
) -> dict[str, Any]:
    labels = sorted(id_to_tag)
    per_class: dict[str, dict[str, float | int]] = {}
    f1s: list[float] = []
    abuse_f1s: list[float] = []
    abuse_ids = [idx for idx, tag in id_to_tag.items() if tag != "O"]

    for label_id in labels:
        tag = id_to_tag[label_id]
        tp = sum(1 for y, yhat in zip(gold, pred) if y == label_id and yhat == label_id)
        fp = sum(1 for y, yhat in zip(gold, pred) if y != label_id and yhat == label_id)
        fn = sum(1 for y, yhat in zip(gold, pred) if y == label_id and yhat != label_id)
        support = sum(1 for y in gold if y == label_id)
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        per_class[tag] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1s.append(f1)
        if tag != "O" and support > 0:
            abuse_f1s.append(f1)

    correct = sum(1 for y, yhat in zip(gold, pred) if y == yhat)
    abuse_tp = sum(1 for y, yhat in zip(gold, pred) if y in abuse_ids and yhat in abuse_ids)
    abuse_fp = sum(1 for y, yhat in zip(gold, pred) if y not in abuse_ids and yhat in abuse_ids)
    abuse_fn = sum(1 for y, yhat in zip(gold, pred) if y in abuse_ids and yhat not in abuse_ids)
    abuse_p, abuse_r, abuse_f1 = precision_recall_f1(abuse_tp, abuse_fp, abuse_fn)

    return {
        "accuracy": correct / len(gold) if gold else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "abuse_macro_f1": sum(abuse_f1s) / len(abuse_f1s) if abuse_f1s else 0.0,
        "abuse_micro_precision": abuse_p,
        "abuse_micro_recall": abuse_r,
        "abuse_micro_f1": abuse_f1,
        "per_class": per_class,
        "tokens": len(gold),
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key in ("word_ids", "char_ids", "labels", "mask", "lengths"):
        out[key] = batch[key].to(device)
    return out


def evaluate(
    model: CharFastTextBiLSTMCRF,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    batches = 0
    gold: list[int] = []
    pred: list[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            loss = model(
                batch["word_ids"],
                batch["char_ids"],
                batch["mask"],
                labels=batch["labels"],
            )
            paths = model(batch["word_ids"], batch["char_ids"], batch["mask"])
            total_loss += float(loss.detach().cpu())
            batches += 1

            labels = batch["labels"].detach().cpu().tolist()
            lengths = batch["lengths"].detach().cpu().tolist()
            for row_labels, row_pred, length in zip(labels, paths, lengths):
                gold.extend(row_labels[:length])
                pred.extend(row_pred[:length])

    metrics = classification_metrics(gold, pred, id_to_tag=ID_TO_TAG)
    return total_loss / max(batches, 1), metrics


def append_log(path: Path, row: EpochMetrics, per_class: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = [
        *asdict(row).keys(),
        "O_precision",
        "O_recall",
        "O_f1",
        "B-ABUSE_precision",
        "B-ABUSE_recall",
        "B-ABUSE_f1",
        "I-ABUSE_precision",
        "I-ABUSE_recall",
        "I-ABUSE_f1",
    ]
    flat = asdict(row)
    for tag in ("O", "B-ABUSE", "I-ABUSE"):
        stats = per_class.get(tag, {})
        flat[f"{tag}_precision"] = stats.get("precision", 0.0)
        flat[f"{tag}_recall"] = stats.get("recall", 0.0)
        flat[f"{tag}_f1"] = stats.get("f1", 0.0)

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(flat)


def save_checkpoint(
    path: Path,
    *,
    model: CharFastTextBiLSTMCRF,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    vocab_bundle: dict[str, Any],
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_bundle": vocab_bundle,
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--dev-path", type=Path, default=DEFAULT_DEV_PATH)
    parser.add_argument("--fasttext-path", type=Path, default=DEFAULT_FASTTEXT_PATH)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--max-word-len", type=int, default=32)
    parser.add_argument("--min-word-freq", type=int, default=1)
    parser.add_argument("--min-char-freq", type=int, default=1)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--char-dim", type=int, default=50)
    parser.add_argument("--char-out-channels", type=int, default=50)
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--freeze-word-embeddings", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument(
        "--early-stop-metric",
        default="abuse_macro_f1",
        choices=("abuse_macro_f1", "abuse_micro_f1", "macro_f1"),
    )
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_path.exists():
        raise FileNotFoundError(
            f"{args.train_path} not found. Run scripts/build_train_split.py first."
        )
    if not args.dev_path.exists():
        raise FileNotFoundError(
            f"{args.dev_path} not found. Run scripts/build_gold_split.py first."
        )

    set_seed(args.seed)
    device = torch.device(args.device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.run_dir / "training_log.csv"
    if log_path.exists():
        log_path.unlink()

    print(f"Loading train: {args.train_path}")
    train_sequences = load_bio_sequences(args.train_path, tag_col="tag")
    print(f"Loading dev:   {args.dev_path}")
    dev_sequences = load_bio_sequences(args.dev_path, tag_col="final_tag")
    print(f"  train sequences: {len(train_sequences):,}")
    print(f"  dev sequences:   {len(dev_sequences):,}")

    print("Building train-only vocabularies...")
    vocab_bundle = build_vocabs(
        train_sequences,
        min_word_freq=args.min_word_freq,
        min_char_freq=args.min_char_freq,
        max_words=args.max_words,
    )
    class_weights = compute_class_weights(
        vocab_bundle["tag_counts"],
        vocab_bundle["tag_to_id"],
    )
    vocab_bundle["class_weights"] = [float(x) for x in class_weights.tolist()]
    save_vocab_bundle(vocab_bundle, args.run_dir / "sequence_vocab.json")
    with (args.run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)

    print(f"  word vocab: {len(vocab_bundle['word_to_id']):,}")
    print(f"  char vocab: {len(vocab_bundle['char_to_id']):,}")
    print(f"  train tag counts: {vocab_bundle['tag_counts']}")
    print(f"  class weights: {vocab_bundle['class_weights']}")

    print("Loading FastText and building aligned embedding matrix...")
    embedding_matrix = build_fasttext_embedding_matrix(
        vocab_bundle["word_to_id"],
        args.fasttext_path,
    )
    torch.save(embedding_matrix, args.run_dir / "fasttext_embedding_matrix.pt")

    train_dataset = BIOTaggingDataset(
        train_sequences,
        vocab_bundle["word_to_id"],
        vocab_bundle["char_to_id"],
        vocab_bundle["tag_to_id"],
        max_word_len=args.max_word_len,
    )
    dev_dataset = BIOTaggingDataset(
        dev_sequences,
        vocab_bundle["word_to_id"],
        vocab_bundle["char_to_id"],
        vocab_bundle["tag_to_id"],
        max_word_len=args.max_word_len,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=bio_collate_fn,
        num_workers=args.num_workers,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=bio_collate_fn,
        num_workers=args.num_workers,
    )

    emission_weights = None if args.no_class_weights else class_weights
    model = CharFastTextBiLSTMCRF(
        num_words=len(vocab_bundle["word_to_id"]),
        num_chars=len(vocab_bundle["char_to_id"]),
        num_tags=len(vocab_bundle["tag_to_id"]),
        fasttext_embeddings=embedding_matrix,
        freeze_word_embeddings=args.freeze_word_embeddings,
        char_dim=args.char_dim,
        char_out_channels=args.char_out_channels,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        emission_class_weights=emission_weights,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metric = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(
                batch["word_ids"],
                batch["char_ids"],
                batch["mask"],
                labels=batch["labels"],
            )
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            batches += 1
            if step % 50 == 0:
                print(
                    f"  epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss={train_loss / batches:.4f}",
                    end="\r",
                    flush=True,
                )
        if batches:
            print(" " * 80, end="\r", flush=True)

        avg_train_loss = train_loss / max(batches, 1)
        dev_loss, dev_metrics = evaluate(model, dev_loader, device)
        row = EpochMetrics(
            epoch=epoch,
            train_loss=avg_train_loss,
            dev_loss=dev_loss,
            accuracy=dev_metrics["accuracy"],
            macro_f1=dev_metrics["macro_f1"],
            abuse_macro_f1=dev_metrics["abuse_macro_f1"],
            abuse_micro_precision=dev_metrics["abuse_micro_precision"],
            abuse_micro_recall=dev_metrics["abuse_micro_recall"],
            abuse_micro_f1=dev_metrics["abuse_micro_f1"],
            dev_tokens=dev_metrics["tokens"],
        )
        append_log(log_path, row, dev_metrics["per_class"])

        metric = float(dev_metrics[args.early_stop_metric])
        improved = metric > best_metric + args.min_delta
        if improved:
            best_metric = metric
            stale_epochs = 0
            save_checkpoint(
                args.run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_metric,
                vocab_bundle=vocab_bundle,
                args=args,
                metrics=dev_metrics,
            )
        else:
            stale_epochs += 1

        save_checkpoint(
            args.run_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_metric,
            vocab_bundle=vocab_bundle,
            args=args,
            metrics=dev_metrics,
        )
        save_checkpoint(
            args.run_dir / f"epoch_{epoch:03d}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_metric,
            vocab_bundle=vocab_bundle,
            args=args,
            metrics=dev_metrics,
        )

        b_stats = dev_metrics["per_class"].get("B-ABUSE", {})
        i_stats = dev_metrics["per_class"].get("I-ABUSE", {})
        print(
            f"epoch={epoch} train_loss={avg_train_loss:.4f} "
            f"dev_loss={dev_loss:.4f} {args.early_stop_metric}={metric:.4f} "
            f"B(P/R/F1)={b_stats.get('precision', 0):.3f}/"
            f"{b_stats.get('recall', 0):.3f}/{b_stats.get('f1', 0):.3f} "
            f"I(P/R/F1)={i_stats.get('precision', 0):.3f}/"
            f"{i_stats.get('recall', 0):.3f}/{i_stats.get('f1', 0):.3f}"
        )

        if stale_epochs >= args.patience:
            print(
                f"Early stopping after {epoch} epochs; "
                f"best {args.early_stop_metric}={best_metric:.4f}."
            )
            break

    print(f"Saved checkpoints and curves under {args.run_dir}")


if __name__ == "__main__":
    main()
