"""
Shared text-normalization utilities for the Tamil abusive-language
detection pipeline.

This module is the single source of truth for text cleaning. Import it
everywhere raw text is touched -- dataset building, lexicon construction,
model training, and inference -- so that preprocessing never drifts
between pipeline stages.

Usage
-----
    from normalize import normalize_text, tag_tokens

    clean = normalize_text(raw_text)
    tokens = tag_tokens(clean)   # [Token("தமிழ்", "TAMIL"), ...]

If calling from a script inside a project subdirectory (scripts/, models/,
lexicon/, eval/, embeddings/), add the project root to ``sys.path`` first::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from normalize import normalize_text, tag_tokens

Pipeline stages (each individually callable; composed by ``normalize_text``):
    1. Unicode NFC normalization                 -> nfc_normalize
    2. HTML entity decoding                       -> decode_html_entities
    3. URL stripping / replacement               -> strip_urls
    4. @mention stripping / replacement          -> strip_mentions
    5. Repeated-character collapsing             -> collapse_repeated_chars
    6. Leetspeak / symbol substitution (opt-in)  -> normalize_leetspeak
    7. Script-level tokenization + tagging       -> tag_tokens

Note on leetspeak normalization
--------------------------------
``normalize_leetspeak`` is a conservative, dictionary-free heuristic
designed for lexicon matching (catching "b1tch", "a55", "sh!t" etc.).
It is opt-in via ``normalize_text(apply_leetspeak=True)`` because it is
lossy: e.g. a genuine film title like "Thalapathy64" could be partially
mis-folded when it contains both letters and digits. Apply it to a
matching/scoring copy, not to text you store or display verbatim.
"""

from __future__ import annotations

import html
import unicodedata
from dataclasses import dataclass

import regex

__all__ = [
    "Token",
    "collapse_repeated_chars",
    "decode_html_entities",
    "nfc_normalize",
    "normalize_and_tag",
    "normalize_leetspeak",
    "normalize_text",
    "strip_mentions",
    "strip_urls",
    "tag_tokens",
]


# ---------------------------------------------------------------------------
# 1. Unicode NFC normalization
# ---------------------------------------------------------------------------

def nfc_normalize(text: str) -> str:
    """Normalize text to Unicode NFC.

    Ensures that visually-identical strings encoded with different (but
    canonically equivalent) code-point sequences compare and hash the same
    way during dedup, lexicon lookup, and tokenization.
    """
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------------------
# 2. HTML entity decoding
# ---------------------------------------------------------------------------

def decode_html_entities(text: str) -> str:
    """Decode HTML entities: ``&amp;``, ``&lt;``, ``&#2965;``, etc."""
    return html.unescape(text)


# ---------------------------------------------------------------------------
# 3. URL / @mention handling
# ---------------------------------------------------------------------------

_URL_RE = regex.compile(r"(?:https?://|www\.)\S+", regex.IGNORECASE)
_MENTION_RE = regex.compile(r"(?<![\w@])@\w+")


def strip_urls(text: str, replacement: str = "") -> str:
    """Remove (or replace) http(s):// and www. URLs."""
    return _URL_RE.sub(replacement, text)


def strip_mentions(text: str, replacement: str = "") -> str:
    """Remove (or replace) @mentions (e.g. '@some_user')."""
    return _MENTION_RE.sub(replacement, text)


# ---------------------------------------------------------------------------
# 4. Repeated-character collapsing
# ---------------------------------------------------------------------------

def collapse_repeated_chars(text: str, max_repeat: int = 2) -> str:
    """Collapse runs of 3+ identical characters down to ``max_repeat``.

    Works on any Unicode character (Tamil, Latin, punctuation, emoji).
    Runs shorter than 3 are never touched, so legitimate doubled letters
    like "book" or "keep" survive the default setting.

    Examples::

        collapse_repeated_chars("sooooo nice!!!!!")  -> "soo nice!!"
        collapse_repeated_chars("தமிழ்!!!!!!!")       -> "தமிழ்!!"
    """
    if max_repeat < 1:
        raise ValueError("max_repeat must be >= 1")
    pattern = regex.compile(r"(.)\1{" + str(max_repeat) + r",}")
    return pattern.sub(lambda m: m.group(1) * max_repeat, text)


# ---------------------------------------------------------------------------
# 5. Leetspeak / symbol substitution normalization
# ---------------------------------------------------------------------------

# Digit ↔ letter substitutions.
_DIGIT_LEET: dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "9": "g",
}

# Symbol ↔ letter substitutions.
_SYMBOL_LEET: dict[str, str] = {
    "@": "a",
    "$": "s",
    "!": "i",
    "+": "t",
    "|": "l",
    "*": "",
}

_WORD_SPLIT_RE = regex.compile(r"(\s+)")
# Punctuation/symbol de-stuffing regex (e.g. p*u*n*d*a*i or s.u.n.n.i or p_u_n_d_a_i)
_DESTUFF_PUNCT_RE = regex.compile(
    r"\b([\p{L}\d](?:[\*\._\-\+~^/][\p{L}\d]){2,})\b",
    regex.UNICODE,
)
# Spaced single letters de-stuffing regex (e.g. "t h e v d i y a")
_DESTUFF_SPACE_RE = regex.compile(
    r"(?:\b[\p{L}]\s+){2,}\b[\p{L}]\b",
    regex.UNICODE,
)


def destuff_obfuscated_words(text: str) -> str:
    """
    Remove deliberate punctuation or space stuffing inside obfuscated words.
    e.g.
      'p*u*n*d*a*i' -> 'pundai'
      's.u.n.n.i'   -> 'sunni'
      'k_i_r_u_k_u' -> 'kiruku'
      't h e v d i' -> 'thevdi'
    """
    if not text:
        return ""

    def _clean_punct_match(m):
        raw = m.group(0)
        # Strip internal punctuation
        return regex.sub(r"[\*\._\-\+~^/]", "", raw)

    text = _DESTUFF_PUNCT_RE.sub(_clean_punct_match, text)

    def _clean_space_match(m):
        raw = m.group(0)
        return regex.sub(r"\s+", "", raw)

    text = _DESTUFF_SPACE_RE.sub(_clean_space_match, text)
    return text


def _normalize_leet_word(word: str) -> str:
    if not word or word.isspace():
        return word

    letters = sum(ch.isalpha() for ch in word)
    digit_hits = sum(ch in _DIGIT_LEET for ch in word)
    sym_hits = sum(ch in _SYMBOL_LEET for ch in word)

    chars = list(word)

    # If the token contains letters or leet symbols, substitute
    if letters or (digit_hits + sym_hits >= 2):
        for i, ch in enumerate(chars):
            if ch in _DIGIT_LEET:
                # Protect pure year numbers like 2024 or standalone 100
                if letters > 0 or digit_hits >= 3:
                    chars[i] = _DIGIT_LEET[ch]
            elif ch in _SYMBOL_LEET:
                chars[i] = _SYMBOL_LEET[ch]

    return "".join(chars)


def normalize_leetspeak(text: str) -> str:
    """Fold common leetspeak / symbol obfuscation and stuffed delimiters back to plain letters."""
    destuffed = destuff_obfuscated_words(text)
    return "".join(_normalize_leet_word(w) for w in _WORD_SPLIT_RE.split(destuffed))


# ---------------------------------------------------------------------------
# 6. Script tagging
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    """A contiguous run of characters sharing the same script category."""

    text: str
    script: str  # "TAMIL" | "LATIN" | "DIGIT" | "OTHER_SCRIPT" | "SYMBOL"


_TOKEN_RE = regex.compile(
    r"(?P<TAMIL>\p{Script=Tamil}+)"
    + r"|(?P<LATIN>\p{Script=Latin}+)"
    + r"|(?P<DIGIT>\p{Nd}+)"
    + r"|(?P<OTHER_SCRIPT>\p{L}+)"
    + r"|(?P<SPACE>\s+)"
    + r"|(?P<SYMBOL>[^\s\p{L}\p{Nd}]+)"
)


def tag_tokens(text: str, include_spaces: bool = False) -> list[Token]:
    """Tokenize ``text`` and tag each run with its dominant script.

    Consecutive characters of the same category are grouped into a single
    Token, switching on any script boundary (even without whitespace)::

        tag_tokens("தமிழ்123 nice!!")
        # [Token("தமிழ்", "TAMIL"), Token("123", "DIGIT"),
        #  Token("nice", "LATIN"), Token("!!", "SYMBOL")]

    Script categories:
    - ``TAMIL``        – Tamil Unicode block (U+0B80–U+0BFF), incl. matras
    - ``LATIN``        – Latin-script letters (covers ASCII + Extended Latin)
    - ``DIGIT``        – Unicode decimal digits (``\\p{Nd}``)
    - ``OTHER_SCRIPT`` – Any other alphabetic script (Devanagari, etc.)
    - ``SYMBOL``       – Punctuation, emoji, and everything else
    """
    tokens: list[Token] = []
    for m in _TOKEN_RE.finditer(text):
        script = m.lastgroup
        if script is None:
            continue  # should not happen; all alternatives are named
        if script == "SPACE" and not include_spaces:
            continue
        tokens.append(Token(text=m.group(), script=script))
    return tokens


# ---------------------------------------------------------------------------
# Composed normalization pipeline
# ---------------------------------------------------------------------------

def normalize_text(
    text: str | None,
    *,
    nfc: bool = True,
    decode_html: bool = True,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    collapse_repeats: bool = True,
    max_repeat: int = 2,
    apply_leetspeak: bool = False,
    url_replacement: str = "",
    mention_replacement: str = "",
) -> str:
    """Run the full normalization pipeline over ``text``.

    Accepts ``None`` safely (returns ``""``).

    Steps run in order:
        1. HTML entity decoding  (ensures real chars before NFC)
        2. NFC normalization
        3. URL removal / replacement
        4. @mention removal / replacement
        5. Repeated-character collapsing
        6. Leetspeak / symbol folding  (only when ``apply_leetspeak=True``)
        7. Whitespace collapse + strip

    Parameters
    ----------
    text:
        Raw input string (or None).
    nfc:
        Apply Unicode NFC normalization.
    decode_html:
        Decode HTML entities before any other step.
    remove_urls:
        Remove http(s)://… and www.… URLs.
    remove_mentions:
        Remove @mention tokens.
    collapse_repeats:
        Collapse character runs of 3+ to ``max_repeat``.
    max_repeat:
        Maximum allowed consecutive identical characters.
    apply_leetspeak:
        Apply leet/symbol folding (opt-in; lossy).
    url_replacement:
        Replacement string for removed URLs (default: empty → delete).
    mention_replacement:
        Replacement string for removed mentions (default: empty → delete).
    """
    if text is None:
        return ""

    text = str(text)

    if decode_html:
        text = decode_html_entities(text)
    if nfc:
        text = nfc_normalize(text)
    if remove_urls:
        text = strip_urls(text, url_replacement)
    if remove_mentions:
        text = strip_mentions(text, mention_replacement)
    if collapse_repeats:
        text = collapse_repeated_chars(text, max_repeat=max_repeat)
    if apply_leetspeak:
        text = normalize_leetspeak(text)

    return regex.sub(r"\s+", " ", text).strip()


def normalize_and_tag(
    text: str | None,
    *,
    nfc: bool = True,
    decode_html: bool = True,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    collapse_repeats: bool = True,
    max_repeat: int = 2,
    apply_leetspeak: bool = False,
    url_replacement: str = "",
    mention_replacement: str = "",
) -> tuple[str, list[Token]]:
    """Convenience wrapper: normalize ``text`` then script-tag the result.

    Returns a ``(clean_text, tokens)`` pair. Accepts the same keyword
    arguments as ``normalize_text``.
    """
    clean = normalize_text(
        text,
        nfc=nfc,
        decode_html=decode_html,
        remove_urls=remove_urls,
        remove_mentions=remove_mentions,
        collapse_repeats=collapse_repeats,
        max_repeat=max_repeat,
        apply_leetspeak=apply_leetspeak,
        url_replacement=url_replacement,
        mention_replacement=mention_replacement,
    )
    return clean, tag_tokens(clean)


# ---------------------------------------------------------------------------
# Quick self-test / demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _SAMPLES = [
        "Check this out http://example.com/video &amp; @some_user!!! Soooo goooood 😍😍😍",
        "இது ரொம்ப நல்லா இருக்கு!!!! பாருங்கள்.....",
        "sh!t this m0vie is s0000 b4d @loser",
        "Mixed: தமிழ்123 nice!! @user http://x.com seen it",
        "Thalapathy 64 trailer 200k likes waiting daaaaa",
        "@Suriya_offl super #NGKTrailer https://youtu.be/abc123 check",
        "Love &lt;3 &amp; respect for u &amp; @actor vijay",
        "b1tch a55hole sh!t 100k views",
        "சூப்பர்!!!!!!!!!!! Semma mass daaaaa",
    ]

    for raw in _SAMPLES:
        clean, tokens = normalize_and_tag(raw)
        print("RAW   :", raw)
        print("CLEAN :", clean)
        print("LEET  :", normalize_text(raw, apply_leetspeak=True))
        print("TOKENS:", tokens)
        print()
