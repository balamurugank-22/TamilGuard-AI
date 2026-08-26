"""
Build the unlabeled Tamil/Tanglish text pool and train FastText embeddings.

Outputs
-------
data/processed/unlabeled_fasttext_pool.txt
    One post-normalization sentence per line.

embeddings/fasttext_tamil_tanglish.bin
    Native FastText binary exported through gensim's Facebook-compatible
    serializer.

embeddings/fasttext_tamil_tanglish_neighbors.txt
    Nearest-neighbor sanity check for lexicon terms and variants.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model, save_facebook_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from normalize import normalize_text, tag_tokens  # type: ignore[import-untyped]

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
DATASETS_DIR = ROOT / "Datasets"
EMBEDDINGS_DIR = ROOT / "embeddings"
LEXICON_PATH = ROOT / "lexicon" / "abusive_lexicon.json"

DEFAULT_CORPUS_PATH = PROCESSED_DIR / "unlabeled_fasttext_pool.txt"
DEFAULT_MODEL_PATH = EMBEDDINGS_DIR / "fasttext_tamil_tanglish.bin"
DEFAULT_REPORT_PATH = EMBEDDINGS_DIR / "fasttext_tamil_tanglish_neighbors.txt"

TEXT_COLUMNS = {
    "text",
    "sentence",
    "comment",
    "content",
    "review",
    "tweet",
    "anchor",
    "positive",
    "negative",
}
LABEL_COLUMNS = {
    "label",
    "labels",
    "sentiment",
    "category",
    "class",
    "target",
    "tag",
}


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    columns: tuple[str, ...] | None = None
    source_type: str = "csv"


class LineSentenceCorpus:
    """Memory-light iterable over tokenized normalized corpus lines."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __iter__(self) -> Iterable[list[str]]:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                tokens = tokenize_for_fasttext(line)
                if tokens:
                    yield tokens


def tokenize_for_fasttext(text: str) -> list[str]:
    tokens = []
    for token in tag_tokens(text):
        if token.script in {"TAMIL", "LATIN", "DIGIT"}:
            surface = token.text.lower()
            if surface:
                tokens.append(surface)
    return tokens


def normalize_for_pool(text: str | None) -> str:
    return normalize_text(text).lower()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)


def choose_text_columns(df: pd.DataFrame, requested: tuple[str, ...] | None) -> list[str]:
    if requested is not None:
        return [col for col in requested if col in df.columns]

    normalized = {str(col).strip().lower(): col for col in df.columns}
    chosen = [normalized[name] for name in TEXT_COLUMNS if name in normalized]
    if chosen:
        return chosen

    candidates = [
        col for col in df.columns
        if str(col).strip().lower() not in LABEL_COLUMNS
        and pd.api.types.is_object_dtype(df[col])
    ]
    if not candidates:
        return []

    def avg_len(col: str) -> float:
        sample = df[col].dropna().astype(str).head(200)
        if sample.empty:
            return 0.0
        return float(sample.str.len().mean())

    return [max(candidates, key=avg_len)]


def iter_csv_text(spec: SourceSpec) -> Iterable[str]:
    df = read_table(spec.path)
    columns = choose_text_columns(df, spec.columns)
    if not columns:
        print(f"  WARNING: no text column detected in {spec.path}")
        return
    print(f"  {spec.path.relative_to(ROOT)} columns={columns} rows={len(df):,}")
    for col in columns:
        for value in df[col].astype(str):
            yield value


def iter_txt_text(spec: SourceSpec) -> Iterable[str]:
    print(f"  {spec.path.relative_to(ROOT)} lines")
    with spec.path.open("r", encoding="utf-8", errors="replace") as f:
        yield from f


def discover_sources() -> list[SourceSpec]:
    sources: list[SourceSpec] = []

    offensive_master = PROCESSED_DIR / "offensive_master.csv"
    if offensive_master.exists():
        sources.append(SourceSpec(offensive_master, ("text",), "csv"))

    triplets = RAW_DIR / "tamil_abusive_triplets_for_contrastive.csv"
    if triplets.exists():
        sources.append(SourceSpec(triplets, ("anchor", "positive", "negative"), "csv"))

    # Sentiment drops are expected under Datasets/, and sentiment-named raw
    # files are also accepted. Labels are ignored; every detected text field
    # becomes unlabeled corpus text.
    sentiment_paths: set[Path] = set()
    if DATASETS_DIR.exists():
        for pattern in ("**/*.csv", "**/*.tsv", "**/*.txt"):
            sentiment_paths.update(DATASETS_DIR.glob(pattern))
    for pattern in ("**/*sentiment*.csv", "**/*sentiment*.tsv", "**/*sentiment*.txt"):
        sentiment_paths.update(RAW_DIR.glob(pattern))

    existing = {spec.path.resolve() for spec in sources}
    for path in sorted(sentiment_paths):
        if path.resolve() in existing:
            continue
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            sources.append(SourceSpec(path, None, "csv"))
        elif suffix == ".txt":
            sources.append(SourceSpec(path, None, "txt"))

    return sources


def iter_source_text(spec: SourceSpec) -> Iterable[str]:
    if spec.source_type == "txt":
        yield from iter_txt_text(spec)
    else:
        yield from iter_csv_text(spec)


def build_unlabeled_pool(corpus_path: Path) -> dict[str, int]:
    sources = discover_sources()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    seen: set[str] = set()

    print("Building normalized unlabeled pool from:")
    with corpus_path.open("w", encoding="utf-8", newline="\n") as fout:
        for spec in sources:
            source_total = 0
            source_kept = 0
            for raw_text in iter_source_text(spec):
                source_total += 1
                stats["raw_texts"] += 1
                clean = normalize_for_pool(raw_text)
                if not clean:
                    stats["empty_after_normalization"] += 1
                    continue
                if clean in seen:
                    stats["duplicates_after_normalization"] += 1
                    continue
                tokens = tokenize_for_fasttext(clean)
                if not tokens:
                    stats["no_trainable_tokens"] += 1
                    continue
                seen.add(clean)
                fout.write(clean + "\n")
                source_kept += 1
                stats["kept_sentences"] += 1
                stats["tokens"] += len(tokens)
            print(f"    kept {source_kept:,} / {source_total:,}")

    stats["sources"] = len(sources)
    return dict(stats)


def train_fasttext(
    corpus_path: Path,
    model_path: Path,
    *,
    vector_size: int,
    epochs: int,
    min_count: int,
    workers: int,
    min_n: int,
    max_n: int,
) -> FastText:
    corpus = LineSentenceCorpus(corpus_path)
    model = FastText(
        vector_size=vector_size,
        window=5,
        min_count=min_count,
        min_n=min_n,
        max_n=max_n,
        sg=1,
        negative=10,
        sample=1e-4,
        workers=workers,
        epochs=epochs,
        seed=13,
    )
    print("Building FastText vocabulary...")
    model.build_vocab(corpus_iterable=corpus)
    print(f"  vocab size: {len(model.wv):,}")
    print("Training FastText...")
    model.train(
        corpus_iterable=corpus,
        total_examples=model.corpus_count,
        total_words=model.corpus_total_words,
        epochs=model.epochs,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_facebook_model(model, str(model_path))
    print(f"Saved FastText binary to {model_path}")
    return model


def load_lexicon_queries(limit: int | None = None) -> list[str]:
    with LEXICON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    queries: list[str] = []
    for canon, rec in data["entries"].items():
        surfaces = [canon, *rec.get("variants", [])]
        for surface in surfaces:
            clean = normalize_for_pool(surface)
            if clean and " " not in clean and clean not in queries:
                queries.append(clean)
            if limit is not None and len(queries) >= limit:
                return queries
    return queries


def write_neighbor_report(
    model: FastText,
    report_path: Path,
    *,
    topn: int,
    query_limit: int | None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    queries = load_lexicon_queries(limit=query_limit)
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["query", "in_vocab", "neighbor", "score"])
        for query in queries:
            in_vocab = query in model.wv.key_to_index
            try:
                neighbors = model.wv.most_similar(query, topn=topn)
            except KeyError:
                writer.writerow([query, in_vocab, "<no-vector>", ""])
                continue
            for neighbor, score in neighbors:
                writer.writerow([query, in_vocab, neighbor, f"{score:.4f}"])
    print(f"Wrote neighbor sanity report to {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-size", type=int, default=200, choices=range(100, 301))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-n", type=int, default=3)
    parser.add_argument("--max-n", type=int, default=6)
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--neighbor-topn", type=int, default=10)
    parser.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help="Limit lexicon sanity-check queries; defaults to all single-token surfaces.",
    )
    parser.add_argument(
        "--reuse-model",
        action="store_true",
        help="Skip corpus build/training and only regenerate the neighbor report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reuse_model:
        print(f"Loading existing FastText binary from {args.model_path}")
        model = load_facebook_model(str(args.model_path))
        write_neighbor_report(
            model,
            args.report_path,
            topn=args.neighbor_topn,
            query_limit=args.query_limit,
        )
        return

    stats = build_unlabeled_pool(args.corpus_path)
    print("\nPool summary:")
    for key in (
        "sources",
        "raw_texts",
        "kept_sentences",
        "duplicates_after_normalization",
        "empty_after_normalization",
        "no_trainable_tokens",
        "tokens",
    ):
        print(f"  {key}: {stats.get(key, 0):,}")

    if stats.get("kept_sentences", 0) < 90_000:
        print(
            "  WARNING: normalized pool is below 90,000 sentences. "
            "No sentiment files were found under Datasets/ or data/raw/*sentiment*."
        )

    model = train_fasttext(
        args.corpus_path,
        args.model_path,
        vector_size=args.vector_size,
        epochs=args.epochs,
        min_count=args.min_count,
        workers=args.workers,
        min_n=args.min_n,
        max_n=args.max_n,
    )
    write_neighbor_report(
        model,
        args.report_path,
        topn=args.neighbor_topn,
        query_limit=args.query_limit,
    )


if __name__ == "__main__":
    main()
