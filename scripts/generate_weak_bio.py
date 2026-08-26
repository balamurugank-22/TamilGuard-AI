"""
Generate weak BIO labels for offensive sentences in offensive_master.csv.

Pipeline
--------
For every sentence whose label is in the "offensive" set:

  1. Tokenise with ``normalize.tag_tokens`` — produces Token(text, script)
     tuples; SYMBOL / DIGIT tokens are always O.

  2. Multi-word matching (sliding 2-token window) — catches the one
     multi-word lexicon phrase ("உயிரோட விடமாட்டேன்") before single-token
     matching consumes those positions.

  3. Exact NFC-lowercased lookup — the fastest path; covers all canonical
     forms and every pre-computed variant in the lexicon (275 keys).

  4. Suffix-strip matching — catches morphologically inflected forms not
     explicitly listed as variants:
       • Tamil (agglutinative): progressively remove 1-4 Unicode code-points
         from the right and re-check the lookup (e.g.
         "தேவிடியாவை" → strip 2 → "தேவிடியா" → match).
       • Tanglish (Latin): try an ordered list of known Romanised Tamil
         suffixes (-kku, -nga, -gala, -ingala, -la, -ya, …) from longest to
         shortest (e.g. "pundaikku" → strip "kku" → "pundai" → match).

  3. (No per-token Levenshtein pass — Tamil pronoun/verb roots are too
     close in edit-distance to slur terms for safe fuzzy matching without
     a morphological analyser. The variant clusters in the lexicon already
     encode every known spelling variation; no additional fuzz is needed.)

  5. Sentences where every token remains O are flagged ``needs_review=True``;
     this is the primary queue for human annotators.

BIO scheme
----------
  B-ABUSE  — first (or only) token of a matched abusive span
  I-ABUSE  — continuation token of the same span (multi-word entries only)
  O        — no match

Output columns
--------------
  sent_id        int   row index within the offensive slice
  sentence_label str   original sentence-level label
  token          str   surface token
  tag            str   B-ABUSE | I-ABUSE | O
  canon          str   matched canonical (empty for O)
  category       str   lexical category (empty for O)
  severity       str   high | medium | low (empty for O)
  match_type     str   exact | suffix_strip | '' for O
  needs_review   bool  True iff whole sentence has zero matches
"""

from __future__ import annotations

import csv
import json
import sys
import unicodedata
from pathlib import Path
from typing import NamedTuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from normalize import tag_tokens, Token  # type: ignore[import-untyped]

LEXICON_PATH  = ROOT / "lexicon"  / "abusive_lexicon.json"
MASTER_PATH   = ROOT / "data" / "processed" / "offensive_master.csv"
OUT_PATH      = ROOT / "data" / "processed" / "weak_bio_labels.csv"

# ─── Offensive labels ────────────────────────────────────────────────────────
OFFENSIVE_LABELS: list[str] = [
    "Offensive_Untargetede",
    "Offensive_Targeted_Insult_Group",
    "Offensive_Targeted_Insult_Individual",
    "Offensive_Targeted_Insult_Other",
    "Misandry",
    "Misogyny",
    "Xenophobia",
    "Homophobia",
    "Transphobic",
    "abusive",
]

# ─── Suffix lists ────────────────────────────────────────────────────────────
# Tanglish Romanised-Tamil suffixes, ordered longest-first so we always
# strip the longest matching suffix.
_TANGLISH_SUFFIXES: list[str] = [
    "ingala", "unga", "kinga", "inga",   # plural + address
    "kulla", "kulla",                    # locative variants
    "gala",                              # plural
    "ukku", "kku",                       # dative
    "oda",                               # comitative
    "ula", "la",                         # locative
    "nga",                               # plural / address
    "dra", "tra",                        # verb suffix
    "ya", "ai",                          # accusative
    "ra",                                # person/verb
    "da",                                # informal address / verbal
    "ku",                                # dative (short)
]

_MIN_STEM_LATIN  = 4   # minimum Latin characters after suffix removal
_MIN_STEM_TAMIL  = 4   # minimum Tamil code-points after suffix removal
_MAX_STRIP_TAMIL = 5   # maximum code-points to try stripping from a Tamil token


# ─── Match result ─────────────────────────────────────────────────────────────
class Match(NamedTuple):
    canon:      str
    category:   str
    severity:   str
    match_type: str


# ─── Lookup builder ───────────────────────────────────────────────────────────
def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


class Lexicon:
    """Pre-compiled lookup structures for fast multi-strategy matching."""

    def __init__(self, path: Path) -> None:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        entries: dict[str, dict] = data["entries"]

        # exact lookup: normalised token → Match
        self.exact: dict[str, Match] = {}
        # multi-word lookup: normalised phrase → Match
        self.multi: dict[str, Match] = {}
        for canon, rec in entries.items():
            cat: str = rec["category"]
            sev: str = rec["severity"]
            m = Match(canon, cat, sev, "exact")

            for surface in [canon] + rec["variants"]:
                key = _nfc(surface.lower())
                if " " in key:
                    self.multi[key] = m
                elif key not in self.exact:
                    self.exact[key] = m

        print(f"  Exact lookup:     {len(self.exact):>5} keys")
        print(f"  Multi-word:       {len(self.multi):>5} phrases")

    # ── per-token matching ────────────────────────────────────────────────────

    def _match_exact(self, norm: str) -> Match | None:
        hit = self.exact.get(norm)
        return hit

    def _match_suffix_latin(self, norm: str) -> Match | None:
        for suf in _TANGLISH_SUFFIXES:
            if norm.endswith(suf):
                stem = norm[: -len(suf)]
                if len(stem) >= _MIN_STEM_LATIN:
                    hit = self.exact.get(stem)
                    if hit:
                        return Match(hit.canon, hit.category, hit.severity,
                                     "suffix_strip")
        return None

    def _match_suffix_tamil(self, norm: str) -> Match | None:
        cps = list(norm)  # list of Unicode code-points (NFC)
        for strip_n in range(1, _MAX_STRIP_TAMIL + 1):
            if len(cps) - strip_n < _MIN_STEM_TAMIL:
                break
            stem = "".join(cps[:-strip_n])
            hit = self.exact.get(stem)
            if hit:
                return Match(hit.canon, hit.category, hit.severity,
                             "suffix_strip")
        return None

    # ── fuzzy Levenshtein matching ────────────────────────────────────────────

    def _match_fuzzy(self, norm: str, max_dist: int = 1, min_len: int = 5) -> Match | None:
        if len(norm) < min_len:
            return None
        # Innocent guard list to avoid false positives on common words
        innocent_guards = {
            "naan", "avan", "aval", "pola", "pottu", "enna", "illa", "irukku",
            "aana", "vara", "romba", "nalla", "super", "mass", "oru", "padam",
            "trailer", "video", "tamil", "ungal", "engal", "naanga", "neenga",
        }
        if norm in innocent_guards:
            return None

        for key, hit in self.exact.items():
            if len(key) < min_len:
                continue
            # Fast length filter
            if abs(len(norm) - len(key)) > max_dist:
                continue
            # Check Levenshtein distance
            d = _fast_edit_distance(norm, key, max_dist)
            if d <= max_dist:
                return Match(hit.canon, hit.category, hit.severity, "fuzzy_levenshtein")
        return None

    # ── per-token matching ────────────────────────────────────────────────────

    def match_token(self, token: Token, sensitivity: str = "standard") -> Match | None:
        """
        Match token against lexicon with configurable sensitivity:
          - 'standard': exact match + suffix-strip
          - 'strict':   exact + suffix-strip + Levenshtein distance <= 1 (len >= 5)
          - 'maximum':  exact + suffix-strip + Levenshtein distance <= 2 (len >= 6) / <= 1 (len >= 4) + substring scan
        """
        if token.script not in ("TAMIL", "LATIN"):
            return None
        norm = _nfc(token.text.lower())
        if len(norm) < 2:
            return None

        # 1. Exact match
        m = self._match_exact(norm)
        if m:
            return m

        # 2. Suffix stripping
        if token.script == "LATIN":
            m_suf = self._match_suffix_latin(norm)
        else:
            m_suf = self._match_suffix_tamil(norm)
        if m_suf:
            return m_suf

        # 3. Fuzzy Levenshtein matching (if strict or maximum sensitivity)
        if sensitivity == "strict":
            return self._match_fuzzy(norm, max_dist=1, min_len=5)
        elif sensitivity == "maximum":
            # Distance 1 for short/med words, distance 2 for longer words
            max_d = 2 if len(norm) >= 6 else 1
            fuzz = self._match_fuzzy(norm, max_dist=max_d, min_len=4)
            if fuzz:
                return fuzz

            # Substring containment for high-severity root words (len >= 4)
            if len(norm) >= 5:
                for key, hit in self.exact.items():
                    if len(key) >= 4 and hit.severity in ("high", "medium"):
                        if key in norm:
                            return Match(hit.canon, hit.category, hit.severity, "substring_containment")

        return None


def _fast_edit_distance(s1: str, s2: str, max_d: int = 2) -> int:
    """Fast bounded Levenshtein edit distance."""
    if abs(len(s1) - len(s2)) > max_d:
        return max_d + 1
    if s1 == s2:
        return 0
    dp = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        new_dp = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            new_dp[j + 1] = min(dp[j + 1] + 1, new_dp[j] + 1, dp[j] + cost)
        dp = new_dp
        if min(dp) > max_d:
            return max_d + 1
    return dp[-1]


# ─── BIO tagging ─────────────────────────────────────────────────────────────

class BIORow:
    canon:          str
    category:       str
    match_type:     str
    needs_review:   bool
    sent_id:        int
    sentence_label: str
    severity:       str
    tag:            str
    token:          str
    __slots__ = (
        "canon", "category", "match_type", "needs_review",
        "sent_id", "sentence_label", "severity", "tag", "token",
    )

    def __init__(
        self,
        sent_id: int,
        sentence_label: str,
        token: str,
        tag: str,
        match: Match | None,
        needs_review: bool,
    ) -> None:
        self.sent_id        = sent_id
        self.sentence_label = sentence_label
        self.token          = token
        self.tag            = tag
        self.canon          = match.canon      if match else ""
        self.category       = match.category   if match else ""
        self.severity       = match.severity   if match else ""
        self.match_type     = match.match_type if match else ""
        self.needs_review   = needs_review

    def as_tuple(self) -> tuple[int, str, str, str, str, str, str, str, bool]:
        return (
            self.sent_id, self.sentence_label, self.token, self.tag,
            self.canon, self.category, self.severity,
            self.match_type, self.needs_review,
        )


def tag_sentence(
    sent_id: int,
    text: str,
    sentence_label: str,
    lex: Lexicon,
) -> list[BIORow]:
    """Return BIO-tagged rows for one offensive sentence."""
    tokens = tag_tokens(_nfc(text))
    n = len(tokens)
    assignments: list[Match | None] = [None] * n
    is_b: list[bool] = [False] * n  # True → B-ABUSE; False → I-ABUSE when not None

    # Pass 1: multi-word matching (sliding 2-token window)
    i = 0
    while i < n - 1:
        phrase_key = _nfc(
            (tokens[i].text + " " + tokens[i + 1].text).lower()
        )
        hit = lex.multi.get(phrase_key)
        if hit:
            assignments[i]     = hit
            assignments[i + 1] = hit
            is_b[i]     = True   # B-ABUSE
            is_b[i + 1] = False  # I-ABUSE
            i += 2
            continue
        i += 1

    # Pass 2: single-token matching where not yet assigned
    for i, tok in enumerate(tokens):
        if assignments[i] is None:
            hit = lex.match_token(tok)
            # Only auto-label single tokens as B-ABUSE if they are high or medium severity.
            # Low severity words (like "dai") should be left to context, unless part of a phrase.
            if hit and hit.severity in ("high", "medium"):
                assignments[i] = hit
                is_b[i] = True

    # Build output rows
    any_match = any(a is not None for a in assignments)
    needs_review = not any_match

    rows: list[BIORow] = []
    for i, tok in enumerate(tokens):
        m = assignments[i]
        if m is None:
            tag = "O"
        elif is_b[i]:
            tag = "B-ABUSE"
        else:
            tag = "I-ABUSE"
        rows.append(BIORow(sent_id, sentence_label, tok.text, tag, m, needs_review))

    return rows


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading lexicon …")
    lex = Lexicon(LEXICON_PATH)

    print("\nLoading offensive_master.csv …")
    df = pd.read_csv(MASTER_PATH, dtype=str).dropna(subset=["text", "label"])
    off = df[df["label"].isin(OFFENSIVE_LABELS)].reset_index(drop=True)
    print(f"  Offensive sentences: {len(off):,}")

    # ── counters ────────────────────────────────────────────────────────────
    sent_matched   = 0
    sent_review    = 0
    tag_counts: dict[str, int]       = {"B-ABUSE": 0, "I-ABUSE": 0, "O": 0}
    match_counts: dict[str, int]     = {"exact": 0, "suffix_strip": 0, "fuzzy": 0}
    cat_counts: dict[str, int]       = {}

    print("\nTagging …")
    FIELDNAMES = [
        "sent_id", "sentence_label", "token", "tag",
        "canon", "category", "severity", "match_type", "needs_review",
    ]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8", newline="") as fout:
        # QUOTE_NONNUMERIC ensures string cells like "nan" (a common
        # Tamil/Tanglish word for "I/myself") are always quoted and not
        # silently coerced to NaN by downstream pandas reads.
        writer = csv.writer(fout, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(FIELDNAMES)

        for sent_id, (_, row) in enumerate(off.iterrows()):
            rows = tag_sentence(sent_id, str(row["text"]), str(row["label"]), lex)

            # Update counters before writing
            has_match = any(r.tag != "O" for r in rows)
            if has_match:
                sent_matched += 1
            else:
                sent_review += 1

            for r in rows:
                tag_counts[r.tag] = tag_counts.get(r.tag, 0) + 1
                if r.match_type:
                    match_counts[r.match_type] = match_counts.get(r.match_type, 0) + 1
                if r.category:
                    cat_counts[r.category] = cat_counts.get(r.category, 0) + 1
                writer.writerow(r.as_tuple())

            if (sent_id + 1) % 1000 == 0:
                print(f"  … {sent_id + 1:,} / {len(off):,}", end="\r", flush=True)

    print(f"\nWrote {OUT_PATH}")

    # ── summary ─────────────────────────────────────────────────────────────
    total_sents = len(off)
    total_toks  = sum(tag_counts.values())

    print("\n" + "=" * 56)
    print("SENTENCE-LEVEL SUMMARY")
    print("=" * 56)
    print(f"  Total offensive sentences  : {total_sents:>7,}")
    print(f"  ≥1 lexicon match           : {sent_matched:>7,}  "
          f"({sent_matched/total_sents:.1%})")
    print(f"  No match → needs_review    : {sent_review:>7,}  "
          f"({sent_review/total_sents:.1%})")

    print("\nTOKEN-LEVEL SUMMARY")
    print("=" * 56)
    print(f"  Total tokens               : {total_toks:>7,}")
    for tag in ("B-ABUSE", "I-ABUSE", "O"):
        n = tag_counts.get(tag, 0)
        print(f"  {tag:<10s}                : {n:>7,}  ({n/total_toks:.2%})")

    print("\nMATCH-TYPE BREAKDOWN (B/I-ABUSE tokens)")
    print("=" * 56)
    for mt in ("exact", "suffix_strip", "fuzzy"):
        n = match_counts.get(mt, 0)
        print(f"  {mt:<15s}             : {n:>6,}")

    print("\nCATEGORY BREAKDOWN (matched tokens)")
    print("=" * 56)
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<22s}       : {n:>6,}")

    # ── spot-check: a few needs_review examples ──────────────────────────────
    print("\nSPOT-CHECK — first 8 needs_review sentences:")
    ex_df = pd.read_csv(OUT_PATH, dtype=str, keep_default_na=False)
    review_sids: list[str] = (
        ex_df[ex_df["needs_review"] == "True"]["sent_id"]
        .drop_duplicates()
        .head(8)
        .tolist()
    )
    for sid_str in review_sids:
        sid_int = int(sid_str)
        orig = str(off.loc[sid_int, "text"])[:90] if sid_int < len(off) else "?"
        lbl  = str(off.loc[sid_int, "label"])     if sid_int < len(off) else "?"
        print(f"  [{lbl[:35]}] {orig}")


if __name__ == "__main__":
    main()
