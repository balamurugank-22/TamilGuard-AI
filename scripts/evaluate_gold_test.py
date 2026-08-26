"""
Final, test-only evaluation for the BIO abusive-span tagger.

Reports:
  - token-level per-class precision/recall/F1
  - derived sentence-level unsafe precision/recall/F1
  - lexicon-only baseline comparison
  - false-negative spelling-variant slur examples
  - false-positive context-dependent examples
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.sequence_tagger import (  # noqa: E402
    ID_TO_TAG,
    TAG_TO_ID,
    CharFastTextBiLSTMCRF,
    BIOTaggingDataset,
    bio_collate_fn,
    load_bio_sequences,
)
from normalize import Token, tag_tokens  # noqa: E402
from scripts.generate_weak_bio import Lexicon, _nfc  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "models" / "checkpoints" / "sequence_tagger" / "best.pt"
DEFAULT_TEST_PATH = ROOT / "data" / "gold" / "gold_test_tokens.csv"
DEFAULT_OUT_DIR = ROOT / "eval" / "gold_test"
DEFAULT_LEXICON_PATH = ROOT / "lexicon" / "abusive_lexicon.json"


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def token_metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    rows = {}
    macro_f1_values = []
    for tag in ("O", "B-ABUSE", "I-ABUSE"):
        tp = sum(1 for y, yhat in zip(gold, pred) if y == tag and yhat == tag)
        fp = sum(1 for y, yhat in zip(gold, pred) if y != tag and yhat == tag)
        fn = sum(1 for y, yhat in zip(gold, pred) if y == tag and yhat != tag)
        precision, recall, f1 = prf(tp, fp, fn)
        support = sum(1 for y in gold if y == tag)
        rows[tag] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        macro_f1_values.append(f1)

    abuse_tags = {"B-ABUSE", "I-ABUSE"}
    tp = sum(1 for y, yhat in zip(gold, pred) if y in abuse_tags and yhat in abuse_tags)
    fp = sum(1 for y, yhat in zip(gold, pred) if y not in abuse_tags and yhat in abuse_tags)
    fn = sum(1 for y, yhat in zip(gold, pred) if y in abuse_tags and yhat not in abuse_tags)
    p, r, f1 = prf(tp, fp, fn)
    return {
        "per_class": rows,
        "macro_f1": sum(macro_f1_values) / len(macro_f1_values),
        "abuse_micro": {
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": sum(1 for y in gold if y in abuse_tags),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        "accuracy": sum(1 for y, yhat in zip(gold, pred) if y == yhat) / len(gold),
    }


def sentence_metrics(df: pd.DataFrame, pred_col: str) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for _, group in df.groupby("gold_id", sort=False):
        gold_unsafe = (group["final_tag"] != "O").any()
        pred_unsafe = (group[pred_col] != "O").any()
        if gold_unsafe and pred_unsafe:
            tp += 1
        elif not gold_unsafe and pred_unsafe:
            fp += 1
        elif gold_unsafe and not pred_unsafe:
            fn += 1
        else:
            tn += 1
    precision, recall, f1 = prf(tp, fp, fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support_unsafe": tp + fn,
        "support_safe": tn + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def token_to_lexicon_token(surface: str) -> Token:
    pieces = tag_tokens(surface)
    if len(pieces) == 1:
        return pieces[0]
    if any(piece.script == "TAMIL" for piece in pieces):
        return Token(surface, "TAMIL")
    if any(piece.script == "LATIN" for piece in pieces):
        return Token(surface, "LATIN")
    return Token(surface, "SYMBOL")


def lexicon_predict_group(tokens: list[str], lexicon: Lexicon) -> list[str]:
    n = len(tokens)
    labels = ["O"] * n

    i = 0
    while i < n - 1:
        phrase_key = _nfc((tokens[i] + " " + tokens[i + 1]).lower())
        hit = lexicon.multi.get(phrase_key)
        if hit:
            labels[i] = "B-ABUSE"
            labels[i + 1] = "I-ABUSE"
            i += 2
            continue
        i += 1

    for idx, token in enumerate(tokens):
        if labels[idx] != "O":
            continue
        hit = lexicon.match_token(token_to_lexicon_token(token))
        if hit:
            labels[idx] = "B-ABUSE"
    return labels


def load_checkpoint_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[CharFastTextBiLSTMCRF, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab_bundle = checkpoint["vocab_bundle"]
    args = checkpoint["args"]
    state = checkpoint["model_state_dict"]
    embedding_matrix = state["word_embedding.weight"].detach().clone()
    weights = torch.tensor(vocab_bundle.get("class_weights", [1.0, 1.0, 1.0]))
    if args.get("no_class_weights", False):
        weights = None
    model = CharFastTextBiLSTMCRF(
        num_words=len(vocab_bundle["word_to_id"]),
        num_chars=len(vocab_bundle["char_to_id"]),
        num_tags=len(vocab_bundle["tag_to_id"]),
        fasttext_embeddings=embedding_matrix,
        freeze_word_embeddings=args.get("freeze_word_embeddings", False),
        char_dim=int(args.get("char_dim", 50)),
        char_out_channels=int(args.get("char_out_channels", 50)),
        lstm_hidden_size=int(args.get("lstm_hidden_size", 256)),
        lstm_layers=int(args.get("lstm_layers", 1)),
        dropout=float(args.get("dropout", 0.3)),
        emission_class_weights=weights,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, vocab_bundle, checkpoint


def model_predictions(
    model: CharFastTextBiLSTMCRF,
    vocab_bundle: dict[str, Any],
    test_path: Path,
    *,
    batch_size: int,
    max_word_len: int,
    device: torch.device,
) -> list[list[str]]:
    sequences = load_bio_sequences(test_path, sent_id_col="gold_id", tag_col="final_tag")
    dataset = BIOTaggingDataset(
        sequences,
        vocab_bundle["word_to_id"],
        vocab_bundle["char_to_id"],
        vocab_bundle["tag_to_id"],
        max_word_len=max_word_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=bio_collate_fn,
    )
    all_paths: list[list[str]] = []
    with torch.no_grad():
        for batch in loader:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            mask = batch["mask"].to(device)
            paths = model(word_ids, char_ids, mask)
            lengths = batch["lengths"].tolist()
            for path, length in zip(paths, lengths):
                all_paths.append([ID_TO_TAG[idx] for idx in path[:length]])
    return all_paths


def flatten_by_gold_id(df: pd.DataFrame, paths: list[list[str]]) -> list[str]:
    flat = []
    if len(paths) != df["gold_id"].nunique():
        raise ValueError("Prediction sequence count does not match test sentence count")
    for (_, group), path in zip(df.groupby("gold_id", sort=False), paths):
        if len(group) != len(path):
            raise ValueError(
                f"Prediction length mismatch for gold_id={group['gold_id'].iloc[0]}"
            )
        flat.extend(path)
    return flat


def build_surface_lookup(lexicon_path: Path) -> dict[str, dict[str, str]]:
    with lexicon_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    lookup: dict[str, dict[str, str]] = {}
    for canon, rec in data["entries"].items():
        surfaces = [(canon, "canonical")] + [
            (variant, "variant") for variant in rec.get("variants", [])
        ]
        for surface, kind in surfaces:
            key = _nfc(surface.lower())
            lookup[key] = {
                "canon": canon,
                "surface_kind": kind,
                "category": rec.get("category", ""),
                "severity": rec.get("severity", ""),
            }
    return lookup


def enrich_predictions(df: pd.DataFrame, surface_lookup: dict[str, dict[str, str]]) -> pd.DataFrame:
    rows = []
    sentence_text = {
        gid: " ".join(group["token"].tolist())
        for gid, group in df.groupby("gold_id", sort=False)
    }
    for _, row in df.iterrows():
        surface = _nfc(str(row["token"]).lower())
        meta = surface_lookup.get(surface, {})
        rows.append({
            **row.to_dict(),
            "sentence_text": sentence_text[row["gold_id"]],
            "lexicon_canon": meta.get("canon", ""),
            "lexicon_surface_kind": meta.get("surface_kind", ""),
            "lexicon_category": meta.get("category", ""),
            "lexicon_severity": meta.get("severity", ""),
        })
    return pd.DataFrame(rows)


def sample_errors(df: pd.DataFrame) -> dict[str, Any]:
    model_fn = df[(df["final_tag"] != "O") & (df["model_tag"] == "O")].copy()
    model_fp = df[(df["final_tag"] == "O") & (df["model_tag"] != "O")].copy()

    variant_fn = model_fn[model_fn["lexicon_surface_kind"] == "variant"].copy()
    variant_counts = (
        variant_fn["token"].str.lower().value_counts().head(20).to_dict()
        if not variant_fn.empty
        else {}
    )
    fn_examples = variant_fn[
        [
            "gold_id",
            "token",
            "final_tag",
            "model_tag",
            "baseline_tag",
            "canon",
            "lexicon_canon",
            "lexicon_category",
            "sentence_text",
        ]
    ].head(20)

    context_fp = model_fp.copy()
    context_fp["context_rank"] = context_fp["lexicon_category"].isin(
        ["profanity", "sexual"]
    ).astype(int)
    context_fp = context_fp.sort_values(
        ["context_rank", "lexicon_surface_kind", "token"],
        ascending=[False, False, True],
    )
    fp_counts = model_fp["token"].str.lower().value_counts().head(20).to_dict()
    fp_examples = context_fp[
        [
            "gold_id",
            "token",
            "final_tag",
            "model_tag",
            "baseline_tag",
            "lexicon_canon",
            "lexicon_category",
            "lexicon_severity",
            "sentence_text",
        ]
    ].head(20)

    return {
        "model_fn_count": int(len(model_fn)),
        "model_fp_count": int(len(model_fp)),
        "variant_fn_count": int(len(variant_fn)),
        "variant_fn_token_counts": variant_counts,
        "model_fp_token_counts": fp_counts,
        "variant_fn_examples": fn_examples,
        "context_fp_examples": fp_examples,
    }


def write_metrics_csv(path: Path, name: str, metrics: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "system",
                "level",
                "label",
                "precision",
                "recall",
                "f1",
                "support",
                "tp",
                "fp",
                "fn",
                "tn",
            ],
        )
        if not exists:
            writer.writeheader()
        for tag, row in metrics["token"]["per_class"].items():
            writer.writerow({
                "system": name,
                "level": "token",
                "label": tag,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "support": row["support"],
                "tp": row["tp"],
                "fp": row["fp"],
                "fn": row["fn"],
                "tn": "",
            })
        sent = metrics["sentence"]
        writer.writerow({
            "system": name,
            "level": "sentence",
            "label": "unsafe",
            "precision": sent["precision"],
            "recall": sent["recall"],
            "f1": sent["f1"],
            "support": sent["support_unsafe"],
            "tp": sent["tp"],
            "fp": sent["fp"],
            "fn": sent["fn"],
            "tn": sent["tn"],
        })


def fmt_pct(value: float) -> str:
    return f"{value:.4f}"


def markdown_table(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    path: Path,
    *,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    model_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    errors: dict[str, Any],
) -> None:
    lines = []
    lines.append("# Gold Test Evaluation\n")
    lines.append(f"- Checkpoint: `{checkpoint_path}`")
    lines.append(f"- Dev-selected epoch in checkpoint: `{checkpoint.get('epoch')}`")
    lines.append("- Test set: `data/gold/gold_test_tokens.csv`\n")

    def add_system(title: str, metrics: dict[str, Any]) -> None:
        lines.append(f"## {title}\n")
        lines.append("### Token-Level\n")
        lines.append("| class | precision | recall | f1 | support |")
        lines.append("|---|---:|---:|---:|---:|")
        for tag in ("O", "B-ABUSE", "I-ABUSE"):
            row = metrics["token"]["per_class"][tag]
            lines.append(
                f"| {tag} | {fmt_pct(row['precision'])} | "
                f"{fmt_pct(row['recall'])} | {fmt_pct(row['f1'])} | "
                f"{row['support']} |"
            )
        abuse = metrics["token"]["abuse_micro"]
        lines.append(
            f"| ABUSE micro | {fmt_pct(abuse['precision'])} | "
            f"{fmt_pct(abuse['recall'])} | {fmt_pct(abuse['f1'])} | "
            f"{abuse['support']} |"
        )
        lines.append("\n### Sentence-Level Unsafe\n")
        sent = metrics["sentence"]
        lines.append(
            f"precision={fmt_pct(sent['precision'])}, "
            f"recall={fmt_pct(sent['recall'])}, f1={fmt_pct(sent['f1'])}, "
            f"support_unsafe={sent['support_unsafe']}, support_safe={sent['support_safe']}"
        )
        lines.append("")

    add_system("ML Model", model_metrics)
    add_system("Lexicon-Only Baseline", baseline_metrics)

    model_abuse = model_metrics["token"]["abuse_micro"]["f1"]
    base_abuse = baseline_metrics["token"]["abuse_micro"]["f1"]
    model_sent = model_metrics["sentence"]["f1"]
    base_sent = baseline_metrics["sentence"]["f1"]
    lines.append("## Added Value\n")
    lines.append(
        f"- Token abuse micro-F1 delta: {fmt_pct(model_abuse - base_abuse)} "
        f"({fmt_pct(model_abuse)} model vs {fmt_pct(base_abuse)} baseline)"
    )
    lines.append(
        f"- Sentence unsafe F1 delta: {fmt_pct(model_sent - base_sent)} "
        f"({fmt_pct(model_sent)} model vs {fmt_pct(base_sent)} baseline)"
    )
    lines.append("")

    lines.append("## Error Analysis\n")
    lines.append(
        f"- Model false negatives: {errors['model_fn_count']} tokens; "
        f"spelling-variant lexicon FNs: {errors['variant_fn_count']} tokens."
    )
    lines.append(
        f"- Model false positives: {errors['model_fp_count']} tokens."
    )
    lines.append("")

    lines.append("### False Negatives On Spelling-Variant Slurs\n")
    if errors["variant_fn_token_counts"]:
        lines.append(
            ", ".join(
                f"`{tok}`={count}"
                for tok, count in errors["variant_fn_token_counts"].items()
            )
        )
    else:
        lines.append("No model false negatives were exact curated spelling variants.")
    lines.append("")
    if not errors["variant_fn_examples"].empty:
        lines.extend(markdown_table(errors["variant_fn_examples"]))
    lines.append("")

    lines.append("### False Positives On Context-Dependent Words\n")
    if errors["model_fp_token_counts"]:
        lines.append(
            ", ".join(
                f"`{tok}`={count}"
                for tok, count in errors["model_fp_token_counts"].items()
            )
        )
    else:
        lines.append("No model false positives.")
    lines.append("")
    if not errors["context_fp_examples"].empty:
        lines.extend(markdown_table(errors["context_fp_examples"]))
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--lexicon-path", type=Path, default=DEFAULT_LEXICON_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, vocab_bundle, checkpoint = load_checkpoint_model(args.checkpoint, device=device)
    ckpt_args = checkpoint["args"]
    max_word_len = int(ckpt_args.get("max_word_len", 32))

    print(f"Loading gold test: {args.test_path}")
    df = pd.read_csv(args.test_path, dtype=str, keep_default_na=False)
    if (df.get("split", "test") != "test").any():
        raise ValueError("Test file contains non-test rows")

    print("Running ML model predictions...")
    model_paths = model_predictions(
        model,
        vocab_bundle,
        args.test_path,
        batch_size=args.batch_size,
        max_word_len=max_word_len,
        device=device,
    )
    df["model_tag"] = flatten_by_gold_id(df, model_paths)

    print("Running lexicon-only baseline...")
    lexicon = Lexicon(args.lexicon_path)
    baseline_flat: list[str] = []
    for _, group in df.groupby("gold_id", sort=False):
        baseline_flat.extend(lexicon_predict_group(group["token"].tolist(), lexicon))
    df["baseline_tag"] = baseline_flat

    surface_lookup = build_surface_lookup(args.lexicon_path)
    df = enrich_predictions(df, surface_lookup)

    model_metrics = {
        "token": token_metrics(df["final_tag"].tolist(), df["model_tag"].tolist()),
        "sentence": sentence_metrics(df, "model_tag"),
    }
    baseline_metrics = {
        "token": token_metrics(df["final_tag"].tolist(), df["baseline_tag"].tolist()),
        "sentence": sentence_metrics(df, "baseline_tag"),
    }
    errors = sample_errors(df)

    predictions_path = args.out_dir / "gold_test_predictions.csv"
    metrics_path = args.out_dir / "gold_test_metrics.csv"
    report_path = args.out_dir / "gold_test_report.md"
    if metrics_path.exists():
        metrics_path.unlink()
    df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    write_metrics_csv(metrics_path, "model", model_metrics)
    write_metrics_csv(metrics_path, "lexicon", baseline_metrics)
    write_report(
        report_path,
        checkpoint_path=args.checkpoint,
        checkpoint=checkpoint,
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
        errors=errors,
    )

    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics:     {metrics_path}")
    print(f"Saved report:      {report_path}")
    print(
        "Model token abuse micro-F1="
        f"{model_metrics['token']['abuse_micro']['f1']:.4f}; "
        "baseline="
        f"{baseline_metrics['token']['abuse_micro']['f1']:.4f}"
    )
    print(
        "Model sentence unsafe F1="
        f"{model_metrics['sentence']['f1']:.4f}; "
        "baseline="
        f"{baseline_metrics['sentence']['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
