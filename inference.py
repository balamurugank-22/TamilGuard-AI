"""
inference.py — end-to-end abusive-language detection pipeline.

Usage (as a library)
--------------------
    from inference import load_model, predict

    bundle = load_model()          # one-time startup; ~1-2 s on CPU
    result = predict("நீ ஒரு thevdiya da", bundle)

    print(result.safe)             # False
    print(result.flagged_words)    # ['thevdiya']
    print(result.categories)       # ['sexual']

Pipeline stages
---------------
  1. normalize_text()         — NFC, HTML decode, URL/mention strip, repeat-collapse
  2. tag_tokens()             — script-aware tokenization → List[Token]
  3. Lexicon.match_token()    — exact + suffix-strip + multi-word matching
  4. Model inference          — word_ids + char_ids → BiLSTM-CRF → BIO tags
  5. Merge (OR)               — token flagged by *either* model or lexicon → B-ABUSE
  6. Aggregate                — build InferenceResult with spans, categories, etc.

Lexicon override hook
---------------------
The lexicon always runs as an independent, parallel pass.  Its detections are
OR-ed with the model's detections: any token flagged by *either* source is
marked unsafe.  Per-span `.source` attribution records which system(s) caught
it ("model", "lexicon", or "both").  This gives a high-precision safety net for
known slurs that the model may miss (e.g. spelling-variant FNs from eval).

Set `lexicon_override=False` in `load_model()` to disable this hook and run
model-only inference.
"""

from __future__ import annotations

import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from normalize import normalize_text, tag_tokens, Token            # noqa: E402
from models.sequence_tagger import (                               # noqa: E402
    CharFastTextBiLSTMCRF,
    ID_TO_TAG,
    UNK_WORD,
    UNK_CHAR,
    PAD_CHAR,
    normalize_token as _norm_tok,
)
from scripts.generate_weak_bio import Lexicon, _nfc, Match         # noqa: E402

# ---------------------------------------------------------------------------
# Default paths (can be overridden in load_model())
# ---------------------------------------------------------------------------
_DEFAULT_CHECKPOINT = ROOT / "models" / "checkpoints" / "sequence_tagger" / "best.pt"
_DEFAULT_LEXICON    = ROOT / "lexicon" / "abusive_lexicon.json"


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlaggedSpan:
    """A single token that was flagged as abusive by model, lexicon, or both."""

    token: str
    """The original surface token (as it appears in the normalized text)."""

    source: str
    """Which system flagged this token: ``'model'``, ``'lexicon'``, or ``'both'``."""

    canon: str = ""
    """Canonical lexicon form (empty when flagged by model only)."""

    category: str = ""
    """Lexicon category, e.g. ``'profanity'``, ``'sexual'``, ``'slur'`` (empty when model-only)."""

    severity: str = ""
    """Lexicon severity: ``'high'``, ``'medium'``, ``'low'``, or ``''``."""

    match_type: str = ""
    """How the lexicon matched: ``'exact'``, ``'suffix_strip'``, or ``''``."""


@dataclass
class InferenceResult:
    """Structured output from :func:`predict`."""

    text: str
    """Input text after normalization."""

    safe: bool
    """``True`` if no token was flagged by model or lexicon."""

    flagged_spans: list[FlaggedSpan]
    """One :class:`FlaggedSpan` per flagged surface token (order matches ``tokens``)."""

    flagged_words: list[str]
    """Unique surface tokens that were flagged (deduped, order-preserving)."""

    categories: list[str]
    """Unique non-empty lexicon categories found across all flagged spans."""

    tokens: list[str]
    """All surface tokens (after normalization + tokenization)."""

    model_tags: list[str]
    """Per-token BIO tags from the model (``'O'``, ``'B-ABUSE'``, ``'I-ABUSE'``)."""

    lexicon_tags: list[str]
    """Per-token BIO tags from the lexicon-only pass."""

    merged_tags: list[str]
    """Final merged BIO tags (OR of model and lexicon)."""


@dataclass
class ModelBundle:
    """
    Holds all loaded artefacts for inference.  Create once with
    :func:`load_model` and reuse across calls to :func:`predict`.
    """

    model: CharFastTextBiLSTMCRF
    word_to_id: dict[str, int]
    char_to_id: dict[str, int]
    max_word_len: int
    device: torch.device
    lexicon: Lexicon | None
    lexicon_override: bool
    checkpoint_meta: dict[str, Any]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str | Path = _DEFAULT_CHECKPOINT,
    lexicon_path: str | Path | None = _DEFAULT_LEXICON,
    *,
    device: str | torch.device | None = None,
    lexicon_override: bool = True,
    verbose: bool = True,
) -> ModelBundle:
    """
    Load the trained model and (optionally) the lexicon.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``best.pt`` checkpoint saved by ``train_sequence_tagger.py``.
    lexicon_path:
        Path to ``abusive_lexicon.json``.  Pass ``None`` to skip the lexicon.
    device:
        ``'cuda'``, ``'cpu'``, or a :class:`torch.device`.  Defaults to CUDA
        if available, otherwise CPU.
    lexicon_override:
        If ``True`` (default), the lexicon override hook is active: tokens
        matched by the lexicon are *always* flagged regardless of model output.
    verbose:
        Print loading messages to stdout.

    Returns
    -------
    ModelBundle
        Ready-to-use bundle; pass to :func:`predict`.
    """
    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if verbose:
        print(f"[inference] Loading checkpoint: {checkpoint_path}  (device={device})")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab_bundle: dict[str, Any] = ckpt["vocab_bundle"]
    args: dict[str, Any] = ckpt["args"]
    state = ckpt["model_state_dict"]

    embedding_matrix: torch.Tensor = state["word_embedding.weight"].detach().clone()
    class_weights = torch.tensor(vocab_bundle.get("class_weights", [1.0, 1.0, 1.0]))
    if args.get("no_class_weights", False):
        class_weights = None

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
        emission_class_weights=class_weights,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[inference] Model loaded  ({n_params:,} parameters, epoch={ckpt.get('epoch')})")

    lexicon: Lexicon | None = None
    if lexicon_path is not None and lexicon_override:
        if verbose:
            print(f"[inference] Loading lexicon: {Path(lexicon_path).name}")
        lexicon = Lexicon(Path(lexicon_path))

    return ModelBundle(
        model=model,
        word_to_id=vocab_bundle["word_to_id"],
        char_to_id=vocab_bundle["char_to_id"],
        max_word_len=int(args.get("max_word_len", 32)),
        device=device,
        lexicon=lexicon,
        lexicon_override=lexicon_override,
        checkpoint_meta={"epoch": ckpt.get("epoch"), "path": str(checkpoint_path)},
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _token_to_lexicon_token(surface: str) -> Token:
    """Convert a surface string to a :class:`Token` for lexicon matching."""
    pieces = tag_tokens(surface)
    if not pieces:
        return Token(surface, "SYMBOL")
    if len(pieces) == 1:
        return pieces[0]
    # Mixed-script token: prefer Tamil, then Latin
    for script in ("TAMIL", "LATIN"):
        if any(p.script == script for p in pieces):
            return Token(surface, script)
    return Token(surface, "SYMBOL")


def _lexicon_predict(
    tokens: list[str],
    lexicon: Lexicon,
    sensitivity: str = "standard",
) -> tuple[list[str], list[Match | None]]:
    """
    Run the full lexicon matching pass over a token list with configurable sensitivity.

    Returns
    -------
    (bio_tags, matches)
        ``bio_tags``: per-token ``'O'`` / ``'B-ABUSE'`` / ``'I-ABUSE'``
        ``matches``:  per-token :class:`Match` or ``None``
    """
    n = len(tokens)
    labels: list[str] = ["O"] * n
    matches: list[Match | None] = [None] * n

    # Pass 1: multi-word (2-token window)
    i = 0
    while i < n - 1:
        phrase_key = _nfc((tokens[i] + " " + tokens[i + 1]).lower())
        hit = lexicon.multi.get(phrase_key)
        if hit:
            labels[i]     = "B-ABUSE"
            labels[i + 1] = "I-ABUSE"
            matches[i]    = hit
            matches[i + 1] = hit
            i += 2
            continue
        i += 1

    # Pass 2: single-token exact + suffix-strip + fuzzy Levenshtein
    for idx, surface in enumerate(tokens):
        if labels[idx] != "O":
            continue
        tok = _token_to_lexicon_token(surface)
        hit = lexicon.match_token(tok, sensitivity=sensitivity)
        
        if hit:
            # On standard/strict, ONLY auto-flag high and medium severity words.
            # Low severity is left to the neural model context.
            # On maximum sensitivity, we override and auto-flag EVERYTHING.
            if sensitivity != "maximum" and hit.severity == "low":
                continue
                
            labels[idx]  = "B-ABUSE"
            matches[idx] = hit

    return labels, matches


def _build_tensor_inputs(
    surface_tokens: list[str],
    word_to_id: dict[str, int],
    char_to_id: dict[str, int],
    max_word_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert surface tokens to word_ids / char_ids / mask tensors for a
    single sentence.  Returns (word_ids, char_ids, mask) each with batch
    dimension of 1.
    """
    unk_word = word_to_id.get(UNK_WORD, 1)
    pad_char = char_to_id.get(PAD_CHAR, 0)
    unk_char = char_to_id.get(UNK_CHAR, 1)

    # Normalize tokens to lowercase (matching training preprocessing)
    norm_tokens = [_norm_tok(t) for t in surface_tokens]
    seq_len = len(norm_tokens)

    word_ids = torch.tensor(
        [word_to_id.get(tok, unk_word) for tok in norm_tokens],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)  # (1, seq_len)

    # Character IDs: (1, seq_len, max_char_len_in_this_sentence)
    char_seqs = []
    for tok in norm_tokens:
        ids = [char_to_id.get(ch, unk_char) for ch in tok[:max_word_len]]
        if not ids:
            ids = [pad_char]
        char_seqs.append(ids)

    max_char_len = max(len(cs) for cs in char_seqs)
    char_tensor = torch.full((seq_len, max_char_len), pad_char, dtype=torch.long, device=device)
    for i, cs in enumerate(char_seqs):
        char_tensor[i, : len(cs)] = torch.tensor(cs, dtype=torch.long, device=device)
    char_ids = char_tensor.unsqueeze(0)  # (1, seq_len, max_char_len)

    mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)

    return word_ids, char_ids, mask


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------

def predict(
    text: str,
    bundle: ModelBundle,
    *,
    apply_leetspeak: bool = False,
    sensitivity: str = "standard",
) -> InferenceResult:
    """
    Run the full detection pipeline on a single raw text string.

    Parameters
    ----------
    text:
        Raw input (social media post, comment, etc.).
    bundle:
        A :class:`ModelBundle` returned by :func:`load_model`.
    apply_leetspeak:
        Apply leet/symbol folding before tokenization. Automatically True in
        strict and maximum sensitivity modes.
    sensitivity:
        Sensitivity mode: ``'standard'``, ``'strict'``, or ``'maximum'``.

    Returns
    -------
    InferenceResult
    """
    if sensitivity in ("strict", "maximum"):
        apply_leetspeak = True

    # ── 1. Normalize ──────────────────────────────────────────────────────────
    clean = normalize_text(
        text,
        apply_leetspeak=apply_leetspeak,
        remove_urls=True,
        remove_mentions=True,
        collapse_repeats=True,
    )

    # ── 2. Tokenize ───────────────────────────────────────────────────────────
    tok_objects = tag_tokens(clean)
    surface_tokens = [t.text for t in tok_objects]

    if not surface_tokens:
        return InferenceResult(
            text=clean,
            safe=True,
            flagged_spans=[],
            flagged_words=[],
            categories=[],
            tokens=[],
            model_tags=[],
            lexicon_tags=[],
            merged_tags=[],
        )

    # ── 3. Model inference ────────────────────────────────────────────────────
    with torch.no_grad():
        word_ids, char_ids, mask = _build_tensor_inputs(
            surface_tokens,
            bundle.word_to_id,
            bundle.char_to_id,
            bundle.max_word_len,
            bundle.device,
        )
        path = bundle.model(word_ids, char_ids, mask)  # returns list[list[int]]
        model_tags = [ID_TO_TAG[idx] for idx in path[0]]

    # ── 4. Lexicon override pass ──────────────────────────────────────────────
    lexicon_tags: list[str]
    lex_matches: list[Match | None]

    if bundle.lexicon is not None and bundle.lexicon_override:
        lexicon_tags, lex_matches = _lexicon_predict(surface_tokens, bundle.lexicon, sensitivity=sensitivity)
    else:
        lexicon_tags = ["O"] * len(surface_tokens)
        lex_matches = [None] * len(surface_tokens)

    # ── 5. Merge (logical OR) ─────────────────────────────────────────────────
    abuse_bio = {"B-ABUSE", "I-ABUSE"}
    merged_tags: list[str] = []
    for m_tag, l_tag in zip(model_tags, lexicon_tags):
        if m_tag in abuse_bio or l_tag in abuse_bio:
            # Preserve I-ABUSE when both agree; otherwise collapse to B-ABUSE
            if m_tag == "I-ABUSE" and l_tag == "I-ABUSE":
                merged_tags.append("I-ABUSE")
            elif m_tag == "I-ABUSE" or l_tag == "I-ABUSE":
                # If one says I-ABUSE the other O: conservative → B-ABUSE
                merged_tags.append("B-ABUSE")
            else:
                merged_tags.append("B-ABUSE")
        else:
            merged_tags.append("O")

    # ── 6. Build FlaggedSpan objects ──────────────────────────────────────────
    flagged_spans: list[FlaggedSpan] = []
    seen_words: dict[str, None] = {}     # ordered set via dict
    seen_cats: dict[str, None] = {}

    for surface, m_tag, l_tag, lx_match in zip(
        surface_tokens, model_tags, lexicon_tags, lex_matches
    ):
        if m_tag not in abuse_bio and l_tag not in abuse_bio:
            continue

        by_model = m_tag in abuse_bio
        by_lex   = l_tag in abuse_bio

        if by_model and by_lex:
            source = "both"
        elif by_model:
            source = "model"
        else:
            source = "lexicon"

        canon      = lx_match.canon      if lx_match else ""
        category   = lx_match.category   if lx_match else ""
        severity   = lx_match.severity   if lx_match else ""
        match_type = lx_match.match_type if lx_match else ""

        flagged_spans.append(FlaggedSpan(
            token=surface,
            source=source,
            canon=canon,
            category=category,
            severity=severity,
            match_type=match_type,
        ))
        seen_words[surface] = None
        if category:
            seen_cats[category] = None

    # ── 7. Aggregate ──────────────────────────────────────────────────────────
    return InferenceResult(
        text=clean,
        safe=len(flagged_spans) == 0,
        flagged_spans=flagged_spans,
        flagged_words=list(seen_words),
        categories=list(seen_cats),
        tokens=surface_tokens,
        model_tags=model_tags,
        lexicon_tags=lexicon_tags,
        merged_tags=merged_tags,
    )


def predict_batch(
    texts: list[str],
    bundle: ModelBundle,
    *,
    apply_leetspeak: bool = False,
    sensitivity: str = "standard",
) -> list[InferenceResult]:
    """
    Run :func:`predict` over a list of texts.
    """
    return [predict(t, bundle, apply_leetspeak=apply_leetspeak, sensitivity=sensitivity) for t in texts]


# ---------------------------------------------------------------------------
# Result serialization helper
# ---------------------------------------------------------------------------

def result_to_dict(result: InferenceResult, *, full: bool = False) -> dict:
    """
    Serialize an :class:`InferenceResult` to a plain dict suitable for
    JSON encoding.

    Parameters
    ----------
    full:
        If ``True``, include ``tokens``, ``model_tags``, ``lexicon_tags``,
        and ``merged_tags`` in the output (useful for debugging).
    """
    out: dict = {
        "safe": result.safe,
        "flagged_words": result.flagged_words,
        "categories": result.categories,
        "flagged_spans": [
            {
                "token":      span.token,
                "source":     span.source,
                "canon":      span.canon,
                "category":   span.category,
                "severity":   span.severity,
                "match_type": span.match_type,
            }
            for span in result.flagged_spans
        ],
    }
    if full:
        out["text_normalized"] = result.text
        out["tokens"]          = result.tokens
        out["model_tags"]      = result.model_tags
        out["lexicon_tags"]    = result.lexicon_tags
        out["merged_tags"]     = result.merged_tags
    return out
