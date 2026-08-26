"""
sample_gold_candidates.py
=========================
Stratified-sample ~1,000 sentences from weak_bio_labels.csv and export them
as two annotation-ready files for manual BIO correction.

Strategy
--------
  • Draw N_AUTO  sentences from the auto-tagged pool (needs_review=False)
  • Draw N_REVIEW sentences from the needs-review pool  (needs_review=True)
  • Total target: N_AUTO + N_REVIEW  (default 600 + 400 = 1 000)

Within each pool the sample is stratified by sentence_label so that the
class distribution in the gold set mirrors the full corpus.

Outputs
-------
data/gold/gold_candidates.csv
    Sentence-level view — one row per sentence.
    Columns:
        gold_id | sent_id | sentence_label | needs_review | text | tokens_json | weak_tags_json

data/gold/gold_candidates_tokens.csv
    Token-level annotation sheet — one row per token.
    Annotators fill in the `corrected_tag` column (leave blank = confirmed).
    Columns:
        gold_id | sent_id | sentence_label | token | weak_tag | corrected_tag |
        canon | category | severity | match_type | needs_review
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WEAK_BIO_PATH = ROOT / "data" / "processed" / "weak_bio_labels.csv"
GOLD_DIR      = ROOT / "data" / "gold"
OUT_SENT      = GOLD_DIR / "gold_candidates.csv"
OUT_TOKENS    = GOLD_DIR / "gold_candidates_tokens.csv"

# ─── Sampling parameters ──────────────────────────────────────────────────────
N_AUTO   = 600   # sentences from the auto-tagged pool (needs_review=False)
N_REVIEW = 400   # sentences from the needs-review   pool (needs_review=True)
SEED     = 42


# ─── Helpers ──────────────────────────────────────────────────────────────────

def stratified_sample(
    sent_df: pd.DataFrame,
    n: int,
    strat_col: str = "sentence_label",
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Sample exactly `n` rows from `sent_df`, stratified by `strat_col`.

    If a stratum has fewer rows than its proportional quota the entire stratum
    is included (no over-sampling).  The shortfall is distributed to larger
    strata via a second proportional pass.
    """
    if len(sent_df) <= n:
        return sent_df.copy()

    counts   = sent_df[strat_col].value_counts()
    total    = len(sent_df)
    quota    = (counts / total * n).round().astype(int)

    # Fix rounding drift so sum == n
    diff = n - quota.sum()
    if diff != 0:
        # add/subtract from the largest stratum
        largest = quota.idxmax()
        quota[largest] += diff

    frames: list[pd.DataFrame] = []
    for label, q in quota.items():
        pool = sent_df[sent_df[strat_col] == label]
        k    = min(q, len(pool))
        frames.append(pool.sample(n=k, random_state=seed))

    sampled = pd.concat(frames, ignore_index=True)

    # Top up if total < n due to small strata
    if len(sampled) < n:
        remaining = sent_df.loc[~sent_df.index.isin(sampled.index)]
        topup = remaining.sample(n=min(n - len(sampled), len(remaining)),
                                 random_state=seed)
        sampled = pd.concat([sampled, topup], ignore_index=True)

    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading weak_bio_labels.csv …")
    tok_df = pd.read_csv(WEAK_BIO_PATH, dtype=str, keep_default_na=False)
    print(f"  Token rows : {len(tok_df):,}")

    # ── Build sentence-level summary ─────────────────────────────────────────
    # Reconstruct full sentence text from tokens
    sent_agg = (
        tok_df
        .groupby("sent_id", sort=False)
        .agg(
            sentence_label=("sentence_label", "first"),
            needs_review   =("needs_review",   "first"),
            tokens_json    =("token",          lambda ts: json.dumps(list(ts), ensure_ascii=False)),
            weak_tags_json =("tag",            lambda ts: json.dumps(list(ts), ensure_ascii=False)),
        )
        .reset_index()
    )
    # Reconstruct readable text (space-join tokens)
    sent_agg["text"] = sent_agg["tokens_json"].apply(
        lambda j: " ".join(json.loads(j))
    )
    print(f"  Sentences  : {len(sent_agg):,}")

    # ── Split into pools ──────────────────────────────────────────────────────
    auto_pool   = sent_agg[sent_agg["needs_review"] == "False"].copy()
    review_pool = sent_agg[sent_agg["needs_review"] == "True"].copy()
    print(f"  Auto-tagged pool  : {len(auto_pool):,}")
    print(f"  Needs-review pool : {len(review_pool):,}")

    # ── Sample from each pool ─────────────────────────────────────────────────
    print(f"\nSampling {N_AUTO} auto-tagged + {N_REVIEW} needs-review …")
    auto_sample   = stratified_sample(auto_pool,   N_AUTO,   seed=SEED)
    review_sample = stratified_sample(review_pool, N_REVIEW, seed=SEED)

    gold_sent = pd.concat([auto_sample, review_sample], ignore_index=True)
    gold_sent = gold_sent.sample(frac=1, random_state=SEED).reset_index(drop=True)
    gold_sent.insert(0, "gold_id", range(len(gold_sent)))

    print(f"  Gold set size : {len(gold_sent):,} sentences")
    print("\nLabel distribution in gold set:")
    dist = gold_sent["sentence_label"].value_counts()
    for lbl, cnt in dist.items():
        print(f"  {lbl:<45s} {cnt:>4d}  ({cnt/len(gold_sent):.1%})")

    print(f"\nNeeds-review breakdown:")
    nr = gold_sent["needs_review"].value_counts()
    for v, c in nr.items():
        print(f"  needs_review={v}: {c}")

    # ── Write sentence-level file ─────────────────────────────────────────────
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    sent_out = gold_sent[[
        "gold_id", "sent_id", "sentence_label",
        "needs_review", "text", "tokens_json", "weak_tags_json",
    ]]
    sent_out.to_csv(OUT_SENT, index=False, encoding="utf-8-sig")
    print(f"\nWrote {OUT_SENT}")

    # ── Write token-level annotation sheet ───────────────────────────────────
    # Filter weak_bio token rows to only sampled sent_ids
    gold_sent_id_map = gold_sent.set_index("sent_id")["gold_id"].to_dict()
    sampled_ids      = set(gold_sent_id_map.keys())

    tok_gold = tok_df[tok_df["sent_id"].isin(sampled_ids)].copy()

    # Attach gold_id
    tok_gold.insert(0, "gold_id",
                    tok_gold["sent_id"].map(gold_sent_id_map).astype(int))

    # Rename tag → weak_tag; add blank corrected_tag column
    tok_gold = tok_gold.rename(columns={"tag": "weak_tag"})
    tok_gold.insert(
        tok_gold.columns.get_loc("weak_tag") + 1,
        "corrected_tag",
        "",
    )

    # Reorder columns
    col_order = [
        "gold_id", "sent_id", "sentence_label", "token",
        "weak_tag", "corrected_tag",
        "canon", "category", "severity", "match_type", "needs_review",
    ]
    tok_gold = tok_gold[col_order]

    # Sort by gold_id then original row order
    tok_gold = tok_gold.reset_index(drop=False).rename(columns={"index": "orig_idx"})
    tok_gold = tok_gold.sort_values(["gold_id", "orig_idx"], ascending=True)
    tok_gold = tok_gold.drop(columns=["orig_idx"]).reset_index(drop=True)

    tok_gold.to_csv(OUT_TOKENS, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUT_TOKENS}")
    print(f"  Token rows in annotation sheet: {len(tok_gold):,}")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("\n-- Sanity checks --")
    assert tok_gold["gold_id"].nunique() == len(gold_sent), \
        "Mismatch: gold_id count in token sheet ≠ sentence count"
    assert tok_gold["sent_id"].nunique() == len(gold_sent), \
        "Mismatch: sent_id count in token sheet ≠ sentence count"
    assert set(tok_gold["sent_id"].unique()) == sampled_ids, \
        "Mismatch: token sheet sent_ids differ from sampled set"
    overlap_with_train = len(sampled_ids) + (len(sent_agg) - len(gold_sent))
    assert overlap_with_train == len(sent_agg), \
        "Overlap detected between gold and remaining training pool"
    print("  ✓ gold_id count matches sentence count")
    print("  ✓ sent_id set matches sampled set")
    print(f"  ✓ Gold ({len(gold_sent)}) + Remaining ({len(sent_agg)-len(gold_sent)}) = "
          f"Total ({len(sent_agg)})")
    print("\nDone. Open data/gold/gold_candidates_tokens.csv in Excel/Sheets,")
    print("fill in the 'corrected_tag' column (B-ABUSE / I-ABUSE / O),")
    print("leave blank where the weak_tag is already correct,")
    print("then run build_gold_split.py to create dev/test sets.")


if __name__ == "__main__":
    main()
