"""
Word+character BIO tagging components.

This module contains the data and model plumbing for:
  token rows -> word IDs + char IDs + BIO label IDs
  CharCNN(word) + FastText(word) -> BiLSTM -> CRF
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torchcrf import CRF

PAD_WORD = "<PAD>"
UNK_WORD = "<UNK>"
PAD_CHAR = "<PAD>"
UNK_CHAR = "<UNK>"

TAG_TO_ID: dict[str, int] = {
    "O": 0,
    "B-ABUSE": 1,
    "I-ABUSE": 2,
}
ID_TO_TAG: dict[int, str] = {v: k for k, v in TAG_TO_ID.items()}


@dataclass(frozen=True)
class BIOSequence:
    sent_id: str
    tokens: list[str]
    tags: list[str]


def normalize_token(token: str) -> str:
    return str(token).strip().lower()


def load_bio_sequences(
    path: str | Path,
    *,
    sent_id_col: str | None = None,
    token_col: str = "token",
    tag_col: str | None = None,
) -> list[BIOSequence]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if sent_id_col is None:
        sent_id_col = "gold_id" if "gold_id" in df.columns else "sent_id"
    if tag_col is None:
        for candidate in ("tag", "final_tag", "weak_tag"):
            if candidate in df.columns:
                tag_col = candidate
                break
    if tag_col is None:
        raise ValueError(
            f"{path} needs one of these tag columns: tag, final_tag, weak_tag"
        )

    required = {sent_id_col, token_col, tag_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    sequences: list[BIOSequence] = []
    for sent_id, group in df.groupby(sent_id_col, sort=False):
        tokens = [normalize_token(tok) for tok in group[token_col].tolist()]
        tags = group[tag_col].tolist()
        unknown_tags = sorted(set(tags) - set(TAG_TO_ID))
        if unknown_tags:
            raise ValueError(f"Unknown BIO tags in sent_id={sent_id}: {unknown_tags}")
        if len(tokens) != len(tags):
            raise ValueError(f"Token/tag length mismatch in sent_id={sent_id}")
        sequences.append(BIOSequence(str(sent_id), tokens, tags))
    return sequences


def build_vocabs(
    sequences: Iterable[BIOSequence],
    *,
    min_word_freq: int = 1,
    min_char_freq: int = 1,
    max_words: int | None = None,
) -> dict[str, Any]:
    word_counts: Counter[str] = Counter()
    char_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()

    for seq in sequences:
        word_counts.update(seq.tokens)
        tag_counts.update(seq.tags)
        for token in seq.tokens:
            char_counts.update(token)

    word_to_id = {PAD_WORD: 0, UNK_WORD: 1}
    for word, count in word_counts.most_common():
        if count < min_word_freq:
            continue
        if max_words is not None and len(word_to_id) >= max_words:
            break
        word_to_id[word] = len(word_to_id)

    char_to_id = {PAD_CHAR: 0, UNK_CHAR: 1}
    for char, count in char_counts.most_common():
        if count >= min_char_freq:
            char_to_id[char] = len(char_to_id)

    return {
        "word_to_id": word_to_id,
        "char_to_id": char_to_id,
        "tag_to_id": dict(TAG_TO_ID),
        "id_to_tag": {str(k): v for k, v in ID_TO_TAG.items()},
        "word_counts": dict(word_counts),
        "char_counts": dict(char_counts),
        "tag_counts": dict(tag_counts),
        "special_tokens": {
            "pad_word": PAD_WORD,
            "unk_word": UNK_WORD,
            "pad_char": PAD_CHAR,
            "unk_char": UNK_CHAR,
        },
    }


def save_vocab_bundle(bundle: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)


def load_vocab_bundle(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_class_weights(
    tag_counts: dict[str, int],
    tag_to_id: dict[str, int] | None = None,
    *,
    clamp_max: float = 20.0,
) -> torch.Tensor:
    tag_to_id = tag_to_id or TAG_TO_ID
    weights = torch.ones(len(tag_to_id), dtype=torch.float32)
    present: list[tuple[int, int]] = []
    for tag, idx in tag_to_id.items():
        count = int(tag_counts.get(tag, 0))
        if count > 0:
            present.append((idx, count))

    if not present:
        return weights

    counts = torch.tensor([count for _, count in present], dtype=torch.float32)
    present_weights = counts.sum() / (len(present) * counts)
    present_weights = present_weights / present_weights.mean()
    for (idx, _), weight in zip(present, present_weights):
        weights[idx] = weight
    return weights.clamp(max=clamp_max)


class BIOTaggingDataset(Dataset):
    def __init__(
        self,
        sequences: list[BIOSequence],
        word_to_id: dict[str, int],
        char_to_id: dict[str, int],
        tag_to_id: dict[str, int] | None = None,
        *,
        max_word_len: int = 32,
    ) -> None:
        self.sequences = sequences
        self.word_to_id = word_to_id
        self.char_to_id = char_to_id
        self.tag_to_id = tag_to_id or TAG_TO_ID
        self.max_word_len = max_word_len

    def __len__(self) -> int:
        return len(self.sequences)

    def _word_id(self, token: str) -> int:
        return self.word_to_id.get(token, self.word_to_id[UNK_WORD])

    def _char_ids(self, token: str) -> list[int]:
        unk = self.char_to_id[UNK_CHAR]
        ids = [self.char_to_id.get(ch, unk) for ch in token[: self.max_word_len]]
        return ids or [self.char_to_id[PAD_CHAR]]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq = self.sequences[idx]
        return {
            "sent_id": seq.sent_id,
            "tokens": seq.tokens,
            "word_ids": [self._word_id(tok) for tok in seq.tokens],
            "char_ids": [self._char_ids(tok) for tok in seq.tokens],
            "labels": [self.tag_to_id[tag] for tag in seq.tags],
        }


def bio_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_seq_len = max(len(item["word_ids"]) for item in batch)
    max_char_len = max(
        len(chars)
        for item in batch
        for chars in item["char_ids"]
    )

    word_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    char_ids = torch.zeros(batch_size, max_seq_len, max_char_len, dtype=torch.long)
    labels = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    for row, item in enumerate(batch):
        seq_len = len(item["word_ids"])
        lengths[row] = seq_len
        word_ids[row, :seq_len] = torch.tensor(item["word_ids"], dtype=torch.long)
        labels[row, :seq_len] = torch.tensor(item["labels"], dtype=torch.long)
        mask[row, :seq_len] = True
        for col, chars in enumerate(item["char_ids"]):
            char_ids[row, col, : len(chars)] = torch.tensor(chars, dtype=torch.long)

    return {
        "sent_ids": [item["sent_id"] for item in batch],
        "tokens": [item["tokens"] for item in batch],
        "word_ids": word_ids,
        "char_ids": char_ids,
        "labels": labels,
        "mask": mask,
        "lengths": lengths,
    }


def build_fasttext_embedding_matrix(
    word_to_id: dict[str, int],
    fasttext_path: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    from gensim.models.fasttext import load_facebook_model

    ft_model = load_facebook_model(str(fasttext_path))
    vector_size = int(ft_model.vector_size)
    matrix = torch.empty(len(word_to_id), vector_size, dtype=dtype)
    nn.init.normal_(matrix, mean=0.0, std=0.05)
    matrix[word_to_id[PAD_WORD]].zero_()

    for word, idx in word_to_id.items():
        if word in {PAD_WORD, UNK_WORD}:
            continue
        matrix[idx] = torch.tensor(ft_model.wv[word], dtype=dtype)
    return matrix


class CharCNN(nn.Module):
    def __init__(
        self,
        num_chars: int,
        *,
        char_dim: int = 50,
        out_channels: int = 50,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        padding_idx: int = 0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.char_embedding = nn.Embedding(
            num_chars,
            char_dim,
            padding_idx=padding_idx,
        )
        self.convs = nn.ModuleList(
            nn.Conv1d(
                in_channels=char_dim,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = out_channels * len(kernel_sizes)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, max_char_len = char_ids.shape
        flat = char_ids.reshape(batch_size * seq_len, max_char_len)
        emb = self.char_embedding(flat).transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            features = torch.relu(conv(emb))
            pooled = torch.max(features, dim=2).values
            conv_outputs.append(pooled)
        char_repr = torch.cat(conv_outputs, dim=1)
        char_repr = self.dropout(char_repr)
        return char_repr.reshape(batch_size, seq_len, self.output_dim)


class CharFastTextBiLSTMCRF(nn.Module):
    def __init__(
        self,
        *,
        num_words: int,
        num_chars: int,
        num_tags: int,
        word_embedding_dim: int = 200,
        fasttext_embeddings: torch.Tensor | None = None,
        freeze_word_embeddings: bool = False,
        char_dim: int = 50,
        char_out_channels: int = 50,
        char_kernel_sizes: tuple[int, ...] = (3, 4, 5),
        lstm_hidden_size: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        pad_word_id: int = 0,
        pad_char_id: int = 0,
        emission_class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if fasttext_embeddings is not None:
            word_embedding_dim = int(fasttext_embeddings.shape[1])
            self.word_embedding = nn.Embedding.from_pretrained(
                fasttext_embeddings,
                freeze=freeze_word_embeddings,
                padding_idx=pad_word_id,
            )
        else:
            self.word_embedding = nn.Embedding(
                num_words,
                word_embedding_dim,
                padding_idx=pad_word_id,
            )

        self.char_cnn = CharCNN(
            num_chars,
            char_dim=char_dim,
            out_channels=char_out_channels,
            kernel_sizes=char_kernel_sizes,
            padding_idx=pad_char_id,
            dropout=dropout,
        )
        lstm_input = word_embedding_dim + self.char_cnn.output_dim
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=lstm_hidden_size // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.emission = nn.Linear(lstm_hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

        if emission_class_weights is None:
            self.register_buffer("emission_log_weights", None)
        else:
            weights = emission_class_weights.float().clamp_min(1e-6).log()
            self.register_buffer("emission_log_weights", weights)

    def emissions(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        word_repr = self.word_embedding(word_ids)
        char_repr = self.char_cnn(char_ids)
        x = self.dropout(torch.cat([word_repr, char_repr], dim=-1))
        lstm_out, _ = self.lstm(x)
        return self.emission(self.dropout(lstm_out))

    def neg_log_likelihood(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        emissions = self.emissions(word_ids, char_ids, mask)
        if self.emission_log_weights is not None:
            emissions = emissions + self.emission_log_weights.view(1, 1, -1)
        log_likelihood = self.crf(
            emissions,
            labels,
            mask=mask.bool(),
            reduction="mean",
        )
        return -log_likelihood

    def decode(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[list[int]]:
        emissions = self.emissions(word_ids, char_ids, mask)
        return self.crf.decode(emissions, mask=mask.bool())

    def forward(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | list[list[int]]:
        if labels is not None:
            return self.neg_log_likelihood(word_ids, char_ids, labels, mask)
        return self.decode(word_ids, char_ids, mask)
