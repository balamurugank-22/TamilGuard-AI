"""
Build sequence tagging assets and smoke-test the CharCNN+FastText+BiLSTM+CRF.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.sequence_tagger import (  # noqa: E402
    CharFastTextBiLSTMCRF,
    BIOTaggingDataset,
    bio_collate_fn,
    build_fasttext_embedding_matrix,
    build_vocabs,
    compute_class_weights,
    load_bio_sequences,
    save_vocab_bundle,
)

DEFAULT_TRAIN_PATH = ROOT / "data" / "processed" / "weak_bio_labels.csv"
DEFAULT_VOCAB_PATH = ROOT / "models" / "sequence_vocab.json"
DEFAULT_MATRIX_PATH = ROOT / "embeddings" / "fasttext_tamil_tanglish_vocab.pt"
DEFAULT_FASTTEXT_PATH = ROOT / "embeddings" / "fasttext_tamil_tanglish.bin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--embedding-matrix-path", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--fasttext-path", type=Path, default=DEFAULT_FASTTEXT_PATH)
    parser.add_argument("--min-word-freq", type=int, default=1)
    parser.add_argument("--min-char-freq", type=int, default=1)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-word-len", type=int, default=32)
    parser.add_argument("--skip-fasttext-matrix", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading BIO corpus: {args.train_path}")
    sequences = load_bio_sequences(args.train_path)
    print(f"  sequences: {len(sequences):,}")
    print(f"  token rows: {sum(len(seq.tokens) for seq in sequences):,}")

    print("Building word and character vocabularies...")
    bundle = build_vocabs(
        sequences,
        min_word_freq=args.min_word_freq,
        min_char_freq=args.min_char_freq,
        max_words=args.max_words,
    )
    class_weights = compute_class_weights(bundle["tag_counts"], bundle["tag_to_id"])
    bundle["class_weights"] = [float(x) for x in class_weights.tolist()]
    save_vocab_bundle(bundle, args.vocab_path)
    print(f"  word vocab: {len(bundle['word_to_id']):,}")
    print(f"  char vocab: {len(bundle['char_to_id']):,}")
    print(f"  tag counts: {bundle['tag_counts']}")
    print(f"  emission class weights: {bundle['class_weights']}")
    print(f"Saved vocab bundle: {args.vocab_path}")

    embedding_matrix = None
    if not args.skip_fasttext_matrix:
        print(f"Building FastText-aligned embedding matrix: {args.fasttext_path}")
        embedding_matrix = build_fasttext_embedding_matrix(
            bundle["word_to_id"],
            args.fasttext_path,
        )
        args.embedding_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embedding_matrix, args.embedding_matrix_path)
        print(
            f"Saved embedding matrix: {args.embedding_matrix_path} "
            f"shape={tuple(embedding_matrix.shape)}"
        )

    print("Smoke-testing Dataset/DataLoader and model forward pass...")
    dataset = BIOTaggingDataset(
        sequences[: min(len(sequences), 64)],
        bundle["word_to_id"],
        bundle["char_to_id"],
        bundle["tag_to_id"],
        max_word_len=args.max_word_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=bio_collate_fn,
    )
    batch = next(iter(loader))
    model = CharFastTextBiLSTMCRF(
        num_words=len(bundle["word_to_id"]),
        num_chars=len(bundle["char_to_id"]),
        num_tags=len(bundle["tag_to_id"]),
        fasttext_embeddings=embedding_matrix,
        emission_class_weights=class_weights,
    )
    loss = model(
        batch["word_ids"],
        batch["char_ids"],
        batch["mask"],
        labels=batch["labels"],
    )
    decoded = model(batch["word_ids"], batch["char_ids"], batch["mask"])
    print(f"  word_ids: {tuple(batch['word_ids'].shape)}")
    print(f"  char_ids: {tuple(batch['char_ids'].shape)}")
    print(f"  labels:   {tuple(batch['labels'].shape)}")
    print(f"  loss:     {float(loss.detach()):.4f}")
    print(f"  decoded:  {len(decoded)} sequences")
    print("All sequence-tagging assets are ready.")


if __name__ == "__main__":
    main()
