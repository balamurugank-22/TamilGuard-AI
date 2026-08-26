"""
build_gold_split.py
===================
Run AFTER manual BIO correction of gold_candidates_tokens.csv.

What it does
------------
1. Reads the annotated token sheet (gold_candidates_tokens.csv).
2. Resolves the final tag:
       corrected_tag if non-empty  →  use annotator correction
       else weak_tag               →  confirmed as-is
3. Performs a stratified 50 / 50 split on gold_id
   (stratified by sentence_label).
4. Writes:
       data/gold/gold_dev_tokens.csv        — dev set token rows (final tags)
       data/gold/gold_test_tokens.csv       — test set token rows (final tags)
       data/gold/gold_split_manifest.csv    — sentence-level manifest (gold_id → split)

Columns in dev/test token files
--------------------------------
    gold_id | sent_id | sentence_label | token | final_tag |
    canon | category | severity | match_type | needs_review | split
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOLD_DIR    = ROOT / "data" / "gold"
IN_TOKENS   = GOLD_DIR / "gold_candidates_tokens.csv"
OUT_DEV     = GOLD_DIR / "gold_dev_tokens.csv"
OUT_TEST    = GOLD_DIR / "gold_test_tokens.csv"
OUT_MANIFEST= GOLD_DIR / "gold_split_manifest.csv"

TEST_SIZE   = 0.50   # 50 % dev, 50 % test
SEED        = 42


def main() -> None:
    if not IN_TOKENS.exists():
        print(f"ERROR: {IN_TOKENS} not found.")
        print("Run sample_gold_candidates.py first and complete annotation.")
        sys.exit(1)

    print("Loading annotated token sheet …")
    tok = pd.read_csv(IN_TOKENS, dtype=str, keep_default_na=False)
    print(f"  Token rows : {len(tok):,}")
    print(f"  Sentences  : {tok['gold_id'].nunique():,}")

    # ── Resolve final tag ─────────────────────────────────────────────────────
    def resolve(row: pd.Series) -> str:
        ct = str(row["corrected_tag"]).strip()
        return ct if ct else str(row["weak_tag"]).strip()

    tok["final_tag"] = tok.apply(resolve, axis=1)

    # Report how many tokens were manually corrected
    changed = (tok["corrected_tag"].str.strip() != "").sum()
    total   = len(tok)
    print(f"\n  Manually corrected tokens : {changed:,} / {total:,} "
          f"({changed/total:.1%})")

    # Tag distribution after correction
    print("\n  Final tag distribution:")
    for tag, cnt in tok["final_tag"].value_counts().items():
        print(f"    {tag:<12s} {cnt:>7,}  ({cnt/total:.2%})")

    # ── Sentence-level frame for splitting ────────────────────────────────────
    sent_df = (
        tok
        .drop_duplicates("gold_id")[["gold_id", "sent_id", "sentence_label", "needs_review"]]
        .reset_index(drop=True)
    )

    # ── Stratified 50/50 split ────────────────────────────────────────────────
    print(f"\nSplitting {len(sent_df):,} sentences 50/50 stratified by sentence_label …")

    dev_ids, test_ids = train_test_split(
        sent_df["gold_id"].tolist(),
        test_size=TEST_SIZE,
        stratify=sent_df["sentence_label"].tolist(),
        random_state=SEED,
    )
    dev_ids  = set(dev_ids)
    test_ids = set(test_ids)

    print(f"  Dev  : {len(dev_ids):,} sentences")
    print(f"  Test : {len(test_ids):,} sentences")
    assert len(dev_ids & test_ids) == 0, "dev/test overlap detected!"

    # ── Annotate tokens with split ────────────────────────────────────────────
    tok["split"] = tok["gold_id"].apply(
        lambda gid: "dev" if gid in dev_ids else "test"
    )

    # Output columns
    out_cols = [
        "gold_id", "sent_id", "sentence_label", "token", "final_tag",
        "canon", "category", "severity", "match_type", "needs_review", "split",
    ]
    tok_out = tok[out_cols]

    dev_tok  = tok_out[tok_out["split"] == "dev"].reset_index(drop=True)
    test_tok = tok_out[tok_out["split"] == "test"].reset_index(drop=True)

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    dev_tok.to_csv(OUT_DEV,  index=False, encoding="utf-8-sig")
    test_tok.to_csv(OUT_TEST, index=False, encoding="utf-8-sig")
    print(f"\nWrote {OUT_DEV}  ({len(dev_tok):,} token rows)")
    print(f"Wrote {OUT_TEST} ({len(test_tok):,} token rows)")

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest = sent_df.copy()
    manifest["split"] = manifest["gold_id"].apply(
        lambda gid: "dev" if gid in dev_ids else "test"
    )
    manifest.to_csv(OUT_MANIFEST, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUT_MANIFEST}")

    # ── Label distribution check ──────────────────────────────────────────────
    print("\n── Label distribution (dev vs test) ──")
    for split_name, subset in [("dev", dev_tok), ("test", test_tok)]:
        sents = subset.drop_duplicates("gold_id")
        print(f"\n  {split_name.upper()} ({len(sents)} sentences):")
        for lbl, cnt in sents["sentence_label"].value_counts().items():
            print(f"    {lbl:<45s} {cnt:>4d}  ({cnt/len(sents):.1%})")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("\n── Sanity checks ──")
    total_gold = len(sent_df)
    assert len(dev_ids) + len(test_ids) == total_gold, \
        f"Split sizes don't add up: {len(dev_ids)}+{len(test_ids)} ≠ {total_gold}"
    assert len(dev_tok) + len(test_tok) == total, \
        "Token row counts don't add up across dev+test"
    print(f"  ✓ {len(dev_ids)} dev + {len(test_ids)} test = {total_gold} total sentences")
    print(f"  ✓ Token rows: {len(dev_tok)} dev + {len(test_tok)} test = {total} total")
    print("  ✓ No dev/test overlap")
    print("\nDone. Run build_train_split.py to carve out the training set.")


if __name__ == "__main__":
    main()
