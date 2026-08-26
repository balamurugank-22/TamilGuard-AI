"""
test_sensitivity.py — Comprehensive tests for Leetspeak/Punctuation De-obfuscation,
Fuzzy Levenshtein matching, and multi-level Sensitivity modes.
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

from normalize import normalize_leetspeak, destuff_obfuscated_words, normalize_text
from scripts.generate_weak_bio import Lexicon, Token
from inference import load_model, predict, ModelBundle
from serve import build_parser, get_bundle
from redact import censor_text

def run_tests():
    print("==================================================================")
    print("  TEST 1: De-obfuscation & Leetspeak Normalizer")
    print("==================================================================")
    
    # 1.1 Punctuation de-stuffing
    destuff_1 = destuff_obfuscated_words("p*u*n*d*a*i")
    print(f"p*u*n*d*a*i -> {destuff_1}")
    assert "pundai" in destuff_1

    destuff_2 = destuff_obfuscated_words("s.u.n.n.i")
    print(f"s.u.n.n.i -> {destuff_2}")
    assert "sunni" in destuff_2

    # 1.2 Leetspeak decoding
    leet_1 = normalize_leetspeak("th3vd!y@")
    print(f"th3vd!y@ -> {leet_1}")
    assert "thevdiya" in leet_1

    leet_2 = normalize_leetspeak("k1rukk@n")
    print(f"k1rukk@n -> {leet_2}")
    assert "kirukkan" in leet_2

    print("✓ Test 1 passed!\n")

    print("==================================================================")
    print("  TEST 2: Lexicon Fuzzy Levenshtein Matching")
    print("==================================================================")
    lex = Lexicon(ROOT / "lexicon" / "abusive_lexicon.json")

    # Standard vs Strict vs Maximum
    tok_typo = Token("thevdya", "LATIN") # 1 deletion edit from thevdiya
    
    match_standard = lex.match_token(tok_typo, sensitivity="standard")
    match_strict   = lex.match_token(tok_typo, sensitivity="strict")
    match_maximum  = lex.match_token(tok_typo, sensitivity="maximum")

    print(f"thevdya (Standard): {match_standard}")
    print(f"thevdya (Strict)  : {match_strict}")
    print(f"thevdya (Maximum) : {match_maximum}")

    assert match_strict is not None, "Expected strict mode to catch fuzzy typo 'thevdya'"
    assert match_maximum is not None, "Expected maximum mode to catch fuzzy typo 'thevdya'"
    print("✓ Test 2 passed!\n")

    print("==================================================================")
    print("  TEST 3: End-to-End Prediction with Sensitivity Modes")
    print("==================================================================")
    parser = build_parser()
    args = parser.parse_args([])
    bundle = get_bundle(args)

    # Obfuscated input
    obfuscated_text = "ne oru th3vd!y@ da"
    res_standard = predict(obfuscated_text, bundle, sensitivity="standard")
    res_strict   = predict(obfuscated_text, bundle, sensitivity="strict")

    print(f"'{obfuscated_text}' Standard Safe: {res_standard.safe}, Spans: {res_standard.flagged_words}")
    print(f"'{obfuscated_text}' Strict Safe  : {res_strict.safe}, Spans: {res_strict.flagged_words}")
    assert not res_strict.safe, "Expected strict mode to de-obfuscate and flag 'th3vd!y@'"

    # Punctuated input
    punct_text = "ivan oru p*u*n*d*a*i da"
    res_punct = predict(punct_text, bundle, sensitivity="strict")
    print(f"'{punct_text}' Strict Safe: {res_punct.safe}, Spans: {res_punct.flagged_words}")
    assert not res_punct.safe, "Expected strict mode to catch punctuated 'p*u*n*d*a*i'"

    # Censoring test
    censor_res = censor_text(obfuscated_text, bundle, mode="partial", sensitivity="strict")
    print(f"Censored: {censor_res.censored}")
    print(f"Polite  : {censor_res.polite_suggestion}")
    assert "th3vd!y@" not in censor_res.censored or "*" in censor_res.censored

    print("✓ Test 3 passed!\n")
    print("🎉 ALL SENSITIVITY & DE-OBFUSCATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
