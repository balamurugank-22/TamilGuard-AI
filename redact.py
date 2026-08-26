"""
redact.py — Smart Redaction & Auto-Censoring Engine for Tamil & Tanglish.

Provides flexible content moderation actions on detected abusive spans:
  1. partial:  th***ya / தே***யா (retains edge graphemes for readability)
  2. tag:      [REDACTED: SEXUAL] / [REDACTED: SLUR] / [REDACTED: ABUSE]
  3. block:    ████████ or ********
  4. polite:   constructive, respectful alternative substitutions
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from inference import InferenceResult, FlaggedSpan, ModelBundle, predict
from normalize import normalize_leetspeak, destuff_obfuscated_words


# ---------------------------------------------------------------------------
# Polite Replacements Dictionary (Tamil & Tanglish)
# ---------------------------------------------------------------------------

POLITE_REPLACEMENTS: dict[str, str] = {
    # Tanglish insults -> Polite alternatives
    "kirukku": "purindhukolladha",
    "kirukkan": "purindhukolladhavar",
    "kirukki": "purindhukolladhavar",
    "paithiyam": "aarvamadhiga aal",
    "kena": "amarikkaiyaana nanbar",
    "kenapunda": "nanbar",
    "muttal": "arivuruvalar",
    "loosu": "konjam mayangiye aal",
    "porukki": "thavaraana aal",
    "naaye": "nanbare",
    "naai": "nanbar",
    "panni": "nanbar",
    "eruma": "periyavare",
    "thevidiya": "[thavaraana varthai]",
    "thevdiya": "[thavaraana varthai]",
    "thevudiya": "[thavaraana varthai]",
    "thevdia": "[thavaraana varthai]",
    "pundai": "[inappropriate]",
    "punda": "[inappropriate]",
    "pundamavan": "[inappropriate]",
    "oombi": "[inappropriate]",
    "poolu": "[inappropriate]",
    "sunni": "[inappropriate]",
    "koodhi": "[inappropriate]",
    "gommale": "nanbare",
    "otha": "thavaru",
    "baadu": "nanbare",
    "potta": "nanbare",
    "mayiru": "mukkiyamatra vishayam",
    "mukkiri": "nanbare",
    "vetti": "velaiyillaadha",

    # Tamil Unicode insults -> Polite alternatives
    "கிறுக்கு": "புரிந்துகொள்ளாத",
    "கிறrules": "புரிந்துகொள்ளாதவர்",
    "கிறுகன்": "புரிந்துகொள்ளாதவர்",
    "கிறுக்கி": "புரிந்துகொள்ளாதவர்",
    "பைத்தியம்": "சிந்தனைக்குரியவர்",
    "முட்டாள்": "அறிவார்ந்தவர்",
    "நாயே": "நண்பரே",
    "நாய்": "நண்பர்",
    "பன்றி": "நண்பர்",
    "எருமை": "பெரியவரே",
    "பொறுக்கி": "தவறானவர்",
    "தேவிடியா": "[தவறான சொல்]",
    "தேவிடியாபயலே": "[தவறான சொல்]",
    "புண்டை": "[முறையற்ற சொல்]",
    "ஊம்பி": "[முறையற்ற சொல்]",
    "சுண்ணி": "[முறையற்ற சொல்]",
    "கூதி": "[முறையற்ற சொல்]",
    "மயிரு": "முக்கியமற்ற விஷயம்",
    "வெட்டி": "நேரமுள்ளவர்",
    "ஒத்தா": "தவறு",
    "சாவு": "நல்வாழ்வு",
    "கொன்றுவிடுவேன்": "எச்சரிக்கிறேன்",
}


# ---------------------------------------------------------------------------
# Unicode Grapheme Cluster Splitter
# ---------------------------------------------------------------------------

def split_graphemes(text: str) -> list[str]:
    """
    Split text into user-perceived characters (grapheme clusters).
    Handles Tamil Unicode combining vowel signs (Pulli, Virama, Matras)
    so letters like 'தே', 'வி', 'டி' are not sliced mid-character.
    """
    graphemes: list[str] = []
    current: list[str] = []

    for char in text:
        cat = unicodedata.category(char)
        # Mn: Nonspacing_Mark (e.g. Tamil vowel signs, pulli ், ா, ி, etc.)
        # Mc: Spacing_Combining_Mark (e.g. ொ, ோ, ௌ)
        # Me: Enclosing_Mark
        if cat in ("Mn", "Mc", "Me") and current:
            current.append(char)
        else:
            if current:
                graphemes.append("".join(current))
            current = [char]

    if current:
        graphemes.append("".join(current))

    return graphemes


# ---------------------------------------------------------------------------
# Core Masking Functions
# ---------------------------------------------------------------------------

def mask_partial(token: str, mask_char: str = "*") -> str:
    """
    Partial masking: preserves the start and end grapheme, masks the middle.
    e.g.
      'thevdiya' -> 'th***ya' (length 8 -> 2 start, 2 end, 4 masked)
      'kirukku'  -> 'k****u'
      'தேவிடியா' -> 'தே**யா'
      'bad'      -> 'b*d'
      'no'       -> '**'
    """
    graphemes = split_graphemes(token)
    n = len(graphemes)

    if n <= 2:
        return mask_char * n
    if n <= 4:
        # Keep 1st and last
        return graphemes[0] + (mask_char * (n - 2)) + graphemes[-1]
    if n <= 6:
        # Keep 1st and last
        return graphemes[0] + (mask_char * (n - 2)) + graphemes[-1]
    
    # For longer words (>= 7), keep 2 leading and 2 trailing if possible
    return "".join(graphemes[:2]) + (mask_char * (n - 4)) + "".join(graphemes[-2:])


def mask_tag(token: str, category: str = "", severity: str = "") -> str:
    """Tag redaction: e.g. [REDACTED: SEXUAL], [REDACTED: SLUR], or [REDACTED]."""
    if category and category.strip():
        tag_name = category.strip().upper()
    elif severity and severity.strip():
        tag_name = severity.strip().upper()
    else:
        tag_name = "ABUSE"
    return f"[REDACTED: {tag_name}]"


def mask_block(token: str, mask_char: str = "█") -> str:
    """Full block redaction: replaces all graphemes with mask_char."""
    graphemes = split_graphemes(token)
    return mask_char * max(len(graphemes), len(token))


def get_polite_replacement(token: str, canon: str = "") -> str:
    """Retrieve a constructive/polite replacement for an abusive token."""
    clean_tok = token.lower().strip()
    clean_canon = canon.lower().strip() if canon else ""

    # 1. Direct surface match
    if clean_tok in POLITE_REPLACEMENTS:
        return POLITE_REPLACEMENTS[clean_tok]

    # 2. Canonical form match
    if clean_canon and clean_canon in POLITE_REPLACEMENTS:
        return POLITE_REPLACEMENTS[clean_canon]

    # 3. Substring match
    for key, repl in POLITE_REPLACEMENTS.items():
        if key in clean_tok or (clean_canon and key in clean_canon):
            return repl

    # 4. Graceful fallback
    return "[nanbar / thavaru]"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class RedactedSpan:
    """Details about a specific token that was redacted."""
    original_token: str
    redacted_token: str
    mode: str
    category: str = ""
    severity: str = ""
    source: str = ""
    canon: str = ""


@dataclass
class CensorResult:
    """Structured result returned by the redactor."""
    original: str
    censored: str
    safe: bool
    mode: str
    redacted_count: int
    redacted_spans: list[RedactedSpan]
    polite_suggestion: str | None = None
    categories: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main Redaction Engine
# ---------------------------------------------------------------------------

class Redactor:
    """
    Smart Content Redactor for Tamil & Tanglish abusive spans.
    """

    @classmethod
    def apply(
        cls,
        text: str,
        inference_result: InferenceResult,
        mode: Literal["partial", "tag", "block", "polite"] = "partial",
        mask_char: str | None = None,
        severity_threshold: Literal["all", "low", "medium", "high"] = "all",
        allowed_categories: list[str] | None = None,
    ) -> CensorResult:
        """
        Apply smart censoring to text based on the detected InferenceResult.
        """
        if inference_result.safe or not inference_result.flagged_spans:
            return CensorResult(
                original=text,
                censored=text,
                safe=True,
                mode=mode,
                redacted_count=0,
                redacted_spans=[],
                polite_suggestion=text,
                categories=[],
            )

        # Severity filter ranking
        sev_rank = {"low": 1, "medium": 2, "high": 3, "": 1}
        min_rank = sev_rank.get(severity_threshold, 1)

        # Map flagged tokens to spans for lookup
        flagged_map: dict[str, FlaggedSpan] = {}
        for span in inference_result.flagged_spans:
            s_rank = sev_rank.get(span.severity.lower(), 1)
            if severity_threshold != "all" and s_rank < min_rank:
                continue
            if allowed_categories and span.category and span.category not in allowed_categories:
                continue
            flagged_map[span.token.lower()] = span

        # Tokenize by whitespace while preserving punctuation and spacing
        parts = re.split(r'(\s+)', text)
        censored_parts = []
        polite_parts = []
        redacted_spans: list[RedactedSpan] = []
        seen_redacted_tokens = set()

        default_char = "*" if mode == "partial" else "█"
        char = mask_char if mask_char is not None else default_char

        for part in parts:
            if not part or part.isspace():
                censored_parts.append(part)
                polite_parts.append(part)
                continue

            # Strip leading/trailing punctuation for lookup
            m = re.match(r'^([^\w\u0B80-\u0BFF]*)([\w\u0B80-\u0BFF]+)([^\w\u0B80-\u0BFF]*)$', part, flags=re.UNICODE)
            if m:
                prefix, core, suffix = m.groups()
                core_lower = core.lower()
                norm_core = normalize_leetspeak(core_lower)
                destuffed_core = destuff_obfuscated_words(core_lower)

                matched_span = (
                    flagged_map.get(core_lower)
                    or flagged_map.get(norm_core)
                    or flagged_map.get(destuffed_core)
                )

                # Also try matching inside flagged_map if multi-character overlap
                if not matched_span:
                    for f_word, f_span in flagged_map.items():
                        if f_word in (core_lower, norm_core, destuffed_core) or f_word in core_lower or f_word in norm_core:
                            matched_span = f_span
                            break

                if matched_span:
                    # Apply selected mode
                    if mode == "partial":
                        masked_core = mask_partial(core, mask_char=char)
                    elif mode == "tag":
                        masked_core = mask_tag(core, matched_span.category, matched_span.severity)
                    elif mode == "block":
                        masked_core = mask_block(core, mask_char=char)
                    elif mode == "polite":
                        masked_core = get_polite_replacement(core, matched_span.canon)
                    else:
                        masked_core = mask_partial(core, mask_char=char)

                    censored_parts.append(f"{prefix}{masked_core}{suffix}")
                    polite_core = get_polite_replacement(core, matched_span.canon)
                    polite_parts.append(f"{prefix}{polite_core}{suffix}")

                    red_span = RedactedSpan(
                        original_token=core,
                        redacted_token=masked_core,
                        mode=mode,
                        category=matched_span.category,
                        severity=matched_span.severity,
                        source=matched_span.source,
                        canon=matched_span.canon,
                    )
                    redacted_spans.append(red_span)
                    seen_redacted_tokens.add(core_lower)
                    continue

            # If no core match, check direct substring containment & de-obfuscation
            low_part = part.lower()
            norm_part = normalize_leetspeak(low_part)
            destuffed_part = destuff_obfuscated_words(low_part)

            matched_span = (
                flagged_map.get(low_part)
                or flagged_map.get(norm_part)
                or flagged_map.get(destuffed_part)
            )
            if not matched_span:
                for f_word, f_span in flagged_map.items():
                    if f_word in (low_part, norm_part, destuffed_part) or f_word in low_part or f_word in norm_part:
                        matched_span = f_span
                        break

            if matched_span:
                if mode == "partial":
                    masked = mask_partial(part, mask_char=char)
                elif mode == "tag":
                    masked = mask_tag(part, matched_span.category, matched_span.severity)
                elif mode == "block":
                    masked = mask_block(part, mask_char=char)
                elif mode == "polite":
                    masked = get_polite_replacement(part, matched_span.canon)
                else:
                    masked = mask_partial(part, mask_char=char)

                censored_parts.append(masked)
                polite_parts.append(get_polite_replacement(part, matched_span.canon))
                redacted_spans.append(
                    RedactedSpan(
                        original_token=part,
                        redacted_token=masked,
                        mode=mode,
                        category=matched_span.category,
                        severity=matched_span.severity,
                        source=matched_span.source,
                        canon=matched_span.canon,
                    )
                )
            else:
                censored_parts.append(part)
                polite_parts.append(part)

        censored_text = "".join(censored_parts)
        polite_suggestion = "".join(polite_parts)

        return CensorResult(
            original=text,
            censored=censored_text,
            safe=len(redacted_spans) == 0,
            mode=mode,
            redacted_count=len(redacted_spans),
            redacted_spans=redacted_spans,
            polite_suggestion=polite_suggestion if len(redacted_spans) > 0 else None,
            categories=inference_result.categories,
        )


def censor_text(
    text: str,
    bundle: ModelBundle,
    mode: Literal["partial", "tag", "block", "polite"] = "partial",
    mask_char: str | None = None,
    severity_threshold: Literal["all", "low", "medium", "high"] = "all",
    allowed_categories: list[str] | None = None,
    sensitivity: str = "standard",
) -> CensorResult:
    """
    Convenience function: runs inference and applies censoring in one call.
    """
    inf_res = predict(text, bundle, sensitivity=sensitivity)
    return Redactor.apply(
        text=text,
        inference_result=inf_res,
        mode=mode,
        mask_char=mask_char,
        severity_threshold=severity_threshold,
        allowed_categories=allowed_categories,
    )


def censor_result_to_dict(res: CensorResult) -> dict:
    """Serialize a CensorResult to a JSON-compatible dict."""
    return {
        "original": res.original,
        "censored": res.censored,
        "safe": res.safe,
        "mode": res.mode,
        "redacted_count": res.redacted_count,
        "categories": res.categories,
        "polite_suggestion": res.polite_suggestion,
        "redacted_spans": [
            {
                "original_token": s.original_token,
                "redacted_token": s.redacted_token,
                "mode": s.mode,
                "category": s.category,
                "severity": s.severity,
                "source": s.source,
                "canon": s.canon,
            }
            for s in res.redacted_spans
        ],
    }
