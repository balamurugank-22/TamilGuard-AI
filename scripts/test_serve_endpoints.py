"""
test_serve_endpoints.py — Test Flask /predict and /censor endpoints using Flask TestClient
"""

import sys
import json
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
from serve import get_bundle, build_parser
from flask import Flask, request, jsonify
from inference import predict, result_to_dict
from redact import Redactor, censor_result_to_dict

def test_endpoints():
    print("[test] Loading ModelBundle...")
    parser = build_parser()
    args = parser.parse_args([])
    bundle = get_bundle(args)
    print(f"[test] Bundle loaded successfully on device: {bundle.device}")

    # Test sentence
    test_text = "நீ ஒரு thevdiya da kirukkan"

    # 1. Test Inference
    print("\n--- 1. Testing Predict ---")
    inf_res = predict(test_text, bundle)
    print(f"Safe: {inf_res.safe}")
    print(f"Flagged Words: {inf_res.flagged_words}")
    print(f"Categories: {inf_res.categories}")
    assert not inf_res.safe, "Expected text to be unsafe"
    assert "thevdiya" in inf_res.flagged_words

    # 2. Test Redactor modes
    print("\n--- 2. Testing Redactor Modes ---")
    for mode in ["partial", "tag", "block", "polite"]:
        c_res = Redactor.apply(test_text, inf_res, mode=mode)
        out_dict = censor_result_to_dict(c_res)
        print(f"Mode [{mode}]: {out_dict['censored']}")
        if out_dict.get("polite_suggestion"):
            print(f"  Polite Suggestion: {out_dict['polite_suggestion']}")
        assert out_dict["redacted_count"] >= 1

    print("\n[test] All serve & censor tests passed successfully!")

if __name__ == "__main__":
    test_endpoints()
