"""
build_train_split.py
====================
Run AFTER build_gold_split.py (gold_split_manifest.csv must exist).

What it does
------------
1. Reads gold_split_manifest.csv → set of sent_ids used in dev/test.
2. Reads weak_bio_labels.csv     → full weak-label corpus.
3. Excludes all gold sent_ids from the corpus.
4. Writes data/processed/train_weak_bio.csv — the training set.
   This file retains all original columns from weak_bio_labels.csv;
   gold sentences are NEVER included here so they cannot leak into training.

Output columns (same as weak_bio_labels.csv)
---------------------------------------------
    sent_id | sentence_label | token | tag |
    canon | category | severity | match_type | needs_review
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOLD_DIR      = ROOT / "data" / "gold"
WEAK_BIO_PATH = ROOT / "data" / "processed" / "weak_bio_labels.csv"
MANIFEST_PATH = GOLD_DIR / "gold_split_manifest.csv"
OUT_TRAIN     = ROOT / "data" / "processed" / "train_weak_bio.csv"


def main() -> None:
    # ── Guard ─────────────────────────────────────────────────────────────────
    if not MANIFEST_PATH.exists():
        print(f"ERROR: {MANIFEST_PATH} not found.")
        print("Run build_gold_split.py first.")
        sys.exit(1)

    # ── Load manifest ─────────────────────────────────────────────────────────
    print("Loading gold_split_manifest.csv …")
    manifest = pd.read_csv(MANIFEST_PATH, dtype=str, keep_default_na=False)
    gold_sent_ids = set(manifest["sent_id"].unique())
    dev_count  = (manifest["split"] == "dev").sum()
    test_count = (manifest["split"] == "test").sum()
    print(f"  Gold sentences : {len(gold_sent_ids):,}  "
          f"(dev={dev_count}, test={test_count})")

    # ── Load full weak-label corpus ───────────────────────────────────────────
    print("Loading weak_bio_labels.csv …")
    tok_df = pd.read_csv(WEAK_BIO_PATH, dtype=str, keep_default_na=False)
    total_sents = tok_df["sent_id"].nunique()
    total_toks  = len(tok_df)
    print(f"  Total sentences : {total_sents:,}")
    print(f"  Total tokens    : {total_toks:,}")

    # ── Remove gold sentences ─────────────────────────────────────────────────
    train_tok = tok_df[~tok_df["sent_id"].isin(gold_sent_ids)].reset_index(drop=True)
    train_sents = train_tok["sent_id"].nunique()
    train_toks  = len(train_tok)

    removed_sents = total_sents - train_sents
    removed_toks  = total_toks  - train_toks

    print(f"\nRemoved {removed_sents:,} gold sentences "
          f"({removed_toks:,} token rows)")
    print(f"Training set   : {train_sents:,} sentences, {train_toks:,} tokens")

    # Sanity: removed count must equal gold count
    assert removed_sents == len(gold_sent_ids), (
        f"Expected to remove {len(gold_sent_ids)} sentences, "
        f"but only removed {removed_sents}"
    )
    assert len(set(train_tok["sent_id"].unique()) & gold_sent_ids) == 0, \
        "Gold sent_ids leaked into training set!"

    # ── Write training file ───────────────────────────────────────────────────
    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    train_tok.to_csv(OUT_TRAIN, index=False, encoding="utf-8-sig")
    print(f"\nWrote {OUT_TRAIN}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("FINAL DATA SPLIT SUMMARY")
    print("=" * 56)
    print(f"  {'Set':<12}  {'Sentences':>10}  {'Tokens':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}")
    print(f"  {'Dev (gold)':<12}  {dev_count:>10,}  "
          f"{'(see gold_dev_tokens.csv)':>10}")
    print(f"  {'Test (gold)':<12}  {test_count:>10,}  "
          f"{'(see gold_test_tokens.csv)':>10}")
    print(f"  {'Train':<12}  {train_sents:>10,}  {train_toks:>10,}")
    print(f"  {'Total':<12}  {total_sents:>10,}  {total_toks:>10,}")
    print("=" * 56)

    # ── Tag & needs_review distribution in training set ───────────────────────
    print("\nTraining set — tag distribution:")
    tag_counts = train_tok["tag"].value_counts()
    for tag, cnt in tag_counts.items():
        print(f"  {tag:<12s} {cnt:>8,}  ({cnt/train_toks:.2%})")

    print("\nTraining set — needs_review breakdown:")
    for v, cnt in train_tok.drop_duplicates("sent_id")["needs_review"].value_counts().items():
        print(f"  needs_review={v}: {cnt:,} sentences")

    print("\nTraining set — label distribution:")
    lbl_df = train_tok.drop_duplicates("sent_id")
    for lbl, cnt in lbl_df["sentence_label"].value_counts().items():
        pct = cnt / train_sents
        print(f"  {lbl:<45s} {cnt:>5,}  ({pct:.1%})")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("\n── Sanity checks ──")
    assert train_sents + len(gold_sent_ids) == total_sents, \
        "Train + gold ≠ total sentences"
    assert train_toks + removed_toks == total_toks, \
        "Train token count does not reconcile with total"
    print(f"  ✓ {train_sents:,} train + {len(gold_sent_ids):,} gold "
          f"= {total_sents:,} total sentences")
    print(f"  ✓ No gold sent_ids present in training set")
    print("\nAll done! Pipeline complete.")
    print("  Annotate  : data/gold/gold_candidates_tokens.csv")
    print("  Dev set   : data/gold/gold_dev_tokens.csv")
    print("  Test set  : data/gold/gold_test_tokens.csv")
    print("  Train set : data/processed/train_weak_bio.csv")


if __name__ == "__main__":
    main()
