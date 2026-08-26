"""
test_redactor.py — Quick test script for redact.py
"""

import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redact import Redactor, split_graphemes, mask_partial, mask_tag, mask_block, get_polite_replacement
from inference import FlaggedSpan, InferenceResult

def test_unit():
    # 1. Test grapheme splitting for Tamil
    tamil_word = "தேவிடியா"
    graphemes = split_graphemes(tamil_word)
    print("Tamil graphemes for தேவிடியா:", graphemes)
    assert graphemes == ["தே", "வி", "டி", "யா"], f"Got {graphemes}"

    # 2. Test partial masking
    m1 = mask_partial("thevdiya", "*")
    print("thevdiya partial:", m1)
    assert m1.startswith("th") and m1.endswith("ya") and "*" in m1

    m2 = mask_partial("தேவிடியா", "*")
    print("தேவிடியா partial:", m2)
    assert m2 == "தே**யா", f"Expected தே**யா but got {m2}"

    # 3. Test Redactor.apply with mock InferenceResult
    mock_inf = InferenceResult(
        text="ne oru thevdiya da kirukkan",
        safe=False,
        flagged_spans=[
            FlaggedSpan(token="thevdiya", source="both", canon="thevidiya", category="sexual", severity="high"),
            FlaggedSpan(token="kirukkan", source="lexicon", canon="kirukku", category="profanity", severity="medium"),
        ],
        flagged_words=["thevdiya", "kirukkan"],
        categories=["sexual", "profanity"],
        tokens=["ne", "oru", "thevdiya", "da", "kirukkan"],
        model_tags=["O", "O", "B-ABUSE", "O", "B-ABUSE"],
        lexicon_tags=["O", "O", "B-ABUSE", "O", "B-ABUSE"],
        merged_tags=["O", "O", "B-ABUSE", "O", "B-ABUSE"],
    )

    res_partial = Redactor.apply("ne oru thevdiya da kirukkan", mock_inf, mode="partial")
    print("Partial Result:", res_partial.censored)
    assert "thevdiya" not in res_partial.censored

    res_tag = Redactor.apply("ne oru thevdiya da kirukkan", mock_inf, mode="tag")
    print("Tag Result:", res_tag.censored)
    assert "[REDACTED: SEXUAL]" in res_tag.censored

    res_block = Redactor.apply("ne oru thevdiya da kirukkan", mock_inf, mode="block")
    print("Block Result:", res_block.censored)
    assert "█" in res_block.censored

    res_polite = Redactor.apply("ne oru thevdiya da kirukkan", mock_inf, mode="polite")
    print("Polite Result:", res_polite.censored)
    print("Polite Suggestion:", res_polite.polite_suggestion)
    assert "purindhukolladhavar" in res_polite.polite_suggestion

    print("All unit tests passed successfully!")

if __name__ == "__main__":
    test_unit()
