"""
Build data/processed/offensive_master.csv.

Pipeline:
1. Load the three offensive/abusive labeled sources (post header-restore):
   - tamil_offensive_full.csv                     (label, text)
   - Abusive_ Tamil Text (Train dataset).csv      (label, text)
   - Abusive_ Tamil Text (Test dataset).csv       (label, text)
2. Standardize schema to [text, label] across all sources and concatenate
   into one master labeled pool.
3. Filter out rows that are:
   - explicitly labeled not-Tamil / Not-Tamil, or
   - written in a non-Tamil script (Devanagari, Malayalam, Telugu, Kannada,
     Gurmukhi, Bengali, Gujarati, Arabic, Thai, ...) which rules out
     Kannada/Hindi/Malayalam/Arabic noise while still allowing Tanglish
     (Tamil written in Latin script), or
   - genuine other-language Latin-script text (e.g. French), detected via
     a French-stopword heuristic, while leaving decorative/elongated
     Tanglish comments (e.g. "Thàlàaaa") alone.
4. Deduplicate exact and near-duplicate sentences using a normalized key
   (case-folded, whitespace-collapsed, punctuation-stripped, elongated
   character runs collapsed).
5. Save the cleaned master pool to data/processed/offensive_master.csv
   with columns [text, label].
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
OTHER_INDIC_RE = re.compile(
    r"[\u0900-\u097F"  # Devanagari (Hindi/Marathi)
    r"\u0980-\u09FF"  # Bengali
    r"\u0A00-\u0A7F"  # Gurmukhi (Punjabi)
    r"\u0A80-\u0AFF"  # Gujarati
    r"\u0B00-\u0B7F"  # Oriya
    r"\u0C00-\u0C7F"  # Telugu
    r"\u0C80-\u0CFF"  # Kannada
    r"\u0D00-\u0D7F"  # Malayalam
    r"\u0600-\u06FF"  # Arabic
    r"\u0E00-\u0E7F"  # Thai
    r"]"
)

NOT_TAMIL_LABEL_RE = re.compile(r"^not[-_ ]?tamil$", re.IGNORECASE)

FRENCH_STOPWORDS = {
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "le", "la", "les", "un", "une", "des", "du", "de", "au", "aux",
    "et", "ou", "mais", "donc", "car", "ni", "que", "qui", "quoi",
    "avec", "sans", "sur", "sous", "dans", "pour", "par", "chez",
    "est", "sont", "était", "être", "avoir", "ont", "se", "ça",
    "très", "plus", "moins", "bien", "aussi", "après", "avant",
    "cette", "ce", "cet", "ces", "mon", "ma", "mes", "ton", "ta",
    "vraiment", "merci", "bonjour", "où", "envie", "savoir",
}
FRENCH_STOPWORD_MIN_HITS = 3


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)


def is_french_like(text: str) -> bool:
    words = _tokenize_words(text)
    hits = sum(1 for w in words if w in FRENCH_STOPWORDS)
    return hits >= FRENCH_STOPWORD_MIN_HITS


def load_label_text_csv(filename: str, sep: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / filename, sep=sep, dtype=str)
    return df[["text", "label"]]


def load_master_pool() -> pd.DataFrame:
    full = load_label_text_csv("tamil_offensive_full.csv", sep="\t")
    train_abusive = load_label_text_csv(
        "Abusive_ Tamil Text (Train dataset).csv", sep=","
    )
    test_abusive = load_label_text_csv(
        "Abusive_ Tamil Text (Test dataset).csv", sep=","
    )
    new_data = load_label_text_csv(
        "new_dataset.csv", sep=","
    )

    pool = pd.concat(
        [full, train_abusive, test_abusive, new_data], ignore_index=True
    )
    return pool


def filter_non_tamil_noise(df: pd.DataFrame) -> pd.DataFrame:
    text = df["text"].fillna("")
    label = df["label"].fillna("")

    is_labeled_not_tamil = label.str.match(NOT_TAMIL_LABEL_RE)
    has_other_indic_script = text.str.contains(OTHER_INDIC_RE)
    has_tamil_script = text.str.contains(TAMIL_RE)
    is_foreign_latin = (~has_tamil_script) & text.apply(is_french_like)

    drop_mask = is_labeled_not_tamil | has_other_indic_script | is_foreign_latin
    kept = df[~drop_mask].copy()

    print(f"  dropped (label=not-Tamil):      {is_labeled_not_tamil.sum()}")
    print(f"  dropped (other Indic/Arabic/Thai script): "
          f"{(has_other_indic_script & ~is_labeled_not_tamil).sum()}")
    print(f"  dropped (French-like Latin text): "
          f"{(is_foreign_latin & ~is_labeled_not_tamil & ~has_other_indic_script).sum()}")
    return kept


def normalize_for_dedup(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    # Collapse elongated character runs: "thalaaaaa" -> "thalaa"
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    # Strip punctuation/symbols, keep word chars (incl. Tamil) and spaces
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    dedup_key = df["text"].apply(normalize_for_dedup)
    before = len(df)
    out = df[~dedup_key.duplicated(keep="first")].copy()
    print(f"  dropped (exact/near duplicates): {before - len(out)}")
    return out


def main() -> None:
    print("Loading master pool...")
    pool = load_master_pool()
    print(f"  raw pool size: {len(pool)}")

    pool = pool.dropna(subset=["text"])
    pool["text"] = pool["text"].astype(str).str.strip()
    pool["label"] = pool["label"].fillna("").astype(str).str.strip()
    pool = pool[pool["text"] != ""]
    print(f"  after dropping empty text: {len(pool)}")

    print("Filtering non-Tamil / foreign-language noise...")
    pool = filter_non_tamil_noise(pool)
    print(f"  remaining: {len(pool)}")

    print("Deduplicating...")
    pool = deduplicate(pool)
    print(f"  remaining: {len(pool)}")

    pool = pool.reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "offensive_master.csv"
    pool.to_csv(out_path, index=False, columns=["text", "label"])
    print(f"Saved {len(pool)} rows to {out_path}")
    print(pool["label"].value_counts())


if __name__ == "__main__":
    main()
