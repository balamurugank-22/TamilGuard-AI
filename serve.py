"""
serve.py — CLI demo and HTTP endpoint for the abusive-language detector.

CLI usage
---------
    # Single text argument
    python serve.py "நீ ஒரு thevdiya da"

    # Read from stdin (one sentence per line)
    python serve.py --stdin

    # Batch from a text file
    python serve.py --file input.txt

    # Debug: show per-token tags
    python serve.py --full "some text here"

    # Compact JSON output (no pretty-print)
    python serve.py --compact "some text here"

HTTP server usage (Flask)
-------------------------
    python serve.py --web
    python serve.py --web --port 8080

    POST /predict
        Body  : {"text": "..."}
        Query : ?full=true   (include per-token debug fields)

    GET /health   → {"status": "ok"}

Endpoints return JSON:
    {
      "safe": false,
      "flagged_words": ["thevdiya"],
      "categories": ["sexual"],
      "flagged_spans": [
        {
          "token": "thevdiya",
          "source": "both",
          "canon": "thevdiya",
          "category": "sexual",
          "severity": "high",
          "match_type": "exact"
        }
      ]
    }

With ?full=true or --full, also includes:
    "text_normalized", "tokens", "model_tags", "lexicon_tags", "merged_tags"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# --- Memory Optimization for Render Free Tier ---
import torch
# Limit PyTorch to 1 thread to massively reduce memory footprint
torch.set_num_threads(1)
# ------------------------------------------------

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for Tamil)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from inference import load_model, predict, result_to_dict, ModelBundle  # noqa: E402
from redact import Redactor, censor_text, censor_result_to_dict           # noqa: E402

# ---------------------------------------------------------------------------
# Shared bundle (loaded once, reused)
# ---------------------------------------------------------------------------

_BUNDLE: ModelBundle | None = None


def get_bundle(args: argparse.Namespace) -> ModelBundle:
    """Load (or return cached) ModelBundle."""
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = load_model(
            checkpoint_path=args.checkpoint,
            lexicon_path=args.lexicon if not args.no_lexicon else None,
            device=args.device,
            lexicon_override=not args.no_lexicon,
            verbose=True,
        )
    return _BUNDLE


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _render(
    text: str,
    bundle: ModelBundle,
    full: bool,
    compact: bool,
    censor: bool = False,
    censor_mode: str = "partial",
    mask_char: str | None = None,
    sensitivity: str = "standard",
) -> str:
    """Run inference (or censoring) and format result as JSON string."""
    t0 = time.perf_counter()
    if censor:
        inf_res = predict(text, bundle, sensitivity=sensitivity)
        censor_res = Redactor.apply(
            text=text,
            inference_result=inf_res,
            mode=censor_mode,
            mask_char=mask_char,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        out = censor_result_to_dict(censor_res)
    else:
        result = predict(text, bundle, sensitivity=sensitivity)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        out = result_to_dict(result, full=full)

    out["_ms"] = round(elapsed_ms, 1)
    indent = None if compact else 2
    return json.dumps(out, ensure_ascii=False, indent=indent)


def run_cli(args: argparse.Namespace) -> None:
    bundle = get_bundle(args)
    censor = getattr(args, "censor", False)
    censor_mode = getattr(args, "censor_mode", "partial")
    mask_char = getattr(args, "mask_char", None)
    sensitivity = getattr(args, "sensitivity", "standard")

    if args.text:
        # Single-text mode
        print(_render(args.text, bundle, args.full, args.compact, censor=censor, censor_mode=censor_mode, mask_char=mask_char, sensitivity=sensitivity))

    elif args.file:
        # File batch mode
        path = Path(args.file)
        lines = [l.rstrip("\n") for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for line in lines:
            print(_render(line, bundle, args.full, args.compact, censor=censor, censor_mode=censor_mode, mask_char=mask_char, sensitivity=sensitivity))

    else:
        # stdin mode
        print("[inference] Reading from stdin — one sentence per line (Ctrl-D to finish)",
              file=sys.stderr)
        for line in sys.stdin:
            line = line.rstrip("\n")
            if line.strip():
                print(_render(line, bundle, args.full, args.compact, censor=censor, censor_mode=censor_mode, mask_char=mask_char, sensitivity=sensitivity))


# ---------------------------------------------------------------------------
# Flask web server
# ---------------------------------------------------------------------------

def create_app(bundle: ModelBundle | None = None, args: argparse.Namespace | None = None):
    """Flask application factory.

    Can be used in two ways:
      1. Directly via `run_web(args)` → dev server with Werkzeug
      2. Via Gunicorn: `gunicorn serve:app` → production

    When called without arguments (Gunicorn mode), uses default paths.
    """
    try:
        from flask import Flask, request, jsonify, send_from_directory
    except ImportError:
        print(
            "[serve] Flask is not installed. Run:  pip install flask\n"
            "Or start without --web for the CLI mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Load model if not provided ────────────────────────────────────────
    if bundle is None:
        if args is None:
            args = build_parser().parse_args([])
        bundle = get_bundle(args)

    # ── Determine if built frontend exists ────────────────────────────────
    FRONTEND_DIST = ROOT / "frontend" / "dist"
    has_frontend = (FRONTEND_DIST / "index.html").exists()

    if has_frontend:
        app = Flask(__name__, static_folder=str(FRONTEND_DIST), static_url_path="")
    else:
        app = Flask(__name__)

    # ── CORS ──────────────────────────────────────────────────────────────
    try:
        from flask_cors import CORS
        CORS(app)
    except ImportError:
        # Fallback manual CORS if flask-cors not installed
        @app.after_request
        def add_cors(response):
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            return response

    # ── disable Flask's default logging to keep output clean ──────────────
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    # ── /health ───────────────────────────────────────────────────────────
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "device": str(bundle.device)})


    # ── /predict  POST ────────────────────────────────────────────────────
    @app.route("/predict", methods=["POST", "OPTIONS"])
    def predict_endpoint():
        if request.method == "OPTIONS":
            # Pre-flight
            resp = app.make_default_options_response()
            return _cors(resp)

        payload = request.get_json(silent=True)
        if not payload or "text" not in payload:
            return jsonify({"error": "body must be JSON with a 'text' field"}), 400

        text: str = str(payload["text"])
        full: bool = request.args.get("full", "").lower() in ("1", "true", "yes")
        sensitivity: str = str(payload.get("sensitivity", request.args.get("sensitivity", "standard"))).lower()

        t0 = time.perf_counter()
        result = predict(text, bundle, sensitivity=sensitivity)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        out = result_to_dict(result, full=full)
        out["_ms"] = elapsed_ms
        return jsonify(out)

    # ── /predict_batch  POST ──────────────────────────────────────────────
    @app.route("/predict_batch", methods=["POST", "OPTIONS"])
    def predict_batch_endpoint():
        if request.method == "OPTIONS":
            resp = app.make_default_options_response()
            return _cors(resp)

        payload = request.get_json(silent=True)
        if not payload or "texts" not in payload:
            return jsonify({"error": "body must be JSON with a 'texts' list"}), 400

        texts = payload["texts"]
        if not isinstance(texts, list):
            return jsonify({"error": "'texts' must be a list of strings"}), 400

        full: bool = request.args.get("full", "").lower() in ("1", "true", "yes")
        sensitivity: str = str(payload.get("sensitivity", request.args.get("sensitivity", "standard"))).lower()

        t0 = time.perf_counter()
        results = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                results.append({"safe": True, "flagged_words": []})
                continue
            
            res = predict(text, bundle, sensitivity=sensitivity)
            results.append(result_to_dict(res, full=full))

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return jsonify({"results": results, "_ms": elapsed_ms})

    # ── /predict  GET (convenience for browser testing) ───────────────────
    @app.route("/predict", methods=["GET"])
    def predict_get():
        text = request.args.get("text", "")
        if not text:
            return jsonify({"error": "Provide ?text=<your text>"}), 400
        full: bool = request.args.get("full", "").lower() in ("1", "true", "yes")
        sensitivity: str = request.args.get("sensitivity", "standard").lower()

        t0 = time.perf_counter()
        result = predict(text, bundle, sensitivity=sensitivity)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        out = result_to_dict(result, full=full)
        out["_ms"] = elapsed_ms
        return jsonify(out)

    # ── /censor  POST ─────────────────────────────────────────────────────
    @app.route("/censor", methods=["POST", "OPTIONS"])
    def censor_endpoint():
        if request.method == "OPTIONS":
            resp = app.make_default_options_response()
            return _cors(resp)

        payload = request.get_json(silent=True)
        if not payload or "text" not in payload:
            return jsonify({"error": "body must be JSON with a 'text' field"}), 400

        text: str = str(payload["text"])
        mode: str = str(payload.get("mode", "partial")).lower()
        mask_char: str | None = payload.get("mask_char")
        severity_threshold: str = str(payload.get("severity_threshold", "all")).lower()
        allowed_categories = payload.get("allowed_categories")
        sensitivity: str = str(payload.get("sensitivity", request.args.get("sensitivity", "standard"))).lower()

        t0 = time.perf_counter()
        inf_res = predict(text, bundle, sensitivity=sensitivity)
        censor_res = Redactor.apply(
            text=text,
            inference_result=inf_res,
            mode=mode,
            mask_char=mask_char,
            severity_threshold=severity_threshold,
            allowed_categories=allowed_categories,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        out = censor_result_to_dict(censor_res)
        out["_ms"] = elapsed_ms
        return jsonify(out)

    # ── /censor  GET (convenience) ────────────────────────────────────────
    @app.route("/censor", methods=["GET"])
    def censor_get():
        text = request.args.get("text", "")
        if not text:
            return jsonify({"error": "Provide ?text=<your text>"}), 400
        mode = request.args.get("mode", "partial").lower()
        mask_char = request.args.get("mask_char")
        severity_threshold = request.args.get("severity_threshold", "all").lower()
        sensitivity = request.args.get("sensitivity", "standard").lower()

        t0 = time.perf_counter()
        inf_res = predict(text, bundle, sensitivity=sensitivity)
        censor_res = Redactor.apply(
            text=text,
            inference_result=inf_res,
            mode=mode,
            mask_char=mask_char,
            severity_threshold=severity_threshold,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        out = censor_result_to_dict(censor_res)
        out["_ms"] = elapsed_ms
        return jsonify(out)

    # ── / — serve React frontend or fallback HTML demo ─────────────────────
    if has_frontend:
        @app.route("/", methods=["GET"])
        def serve_index():
            return send_from_directory(str(FRONTEND_DIST), "index.html")

        # Catch-all for client-side routing (React Router, etc.)
        @app.errorhandler(404)
        def fallback(e):
            return send_from_directory(str(FRONTEND_DIST), "index.html")
    else:
        @app.route("/", methods=["GET"])
        def demo_ui():
            html = _demo_html()
            return html, 200, {"Content-Type": "text/html; charset=utf-8"}

    return app


def run_web(args: argparse.Namespace) -> None:
    """Start the Flask dev server (for `python serve.py --web`)."""
    bundle = get_bundle(args)
    app = create_app(bundle=bundle, args=args)

    host = args.host
    port = args.port
    print(f"[inference] Server listening on http://{host}:{port}/")
    print(f"[inference]   POST /predict   body: {{\"text\": \"...\"}}")
    print(f"[inference]   GET  /predict?text=...  (browser shortcut)")
    print(f"[inference]   GET  /           (interactive demo UI)")
    app.run(host=host, port=port, debug=False, use_reloader=False)


# ---------------------------------------------------------------------------
# Minimal demo HTML (self-contained, no build step)
# ---------------------------------------------------------------------------

def _demo_html() -> str:
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Tamil Abusive Language Detector</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #0d0f17;
      --surface:   #161a27;
      --surface2:  #1e2436;
      --border:    #2a3150;
      --accent:    #6366f1;
      --accent-hi: #818cf8;
      --safe:      #22c55e;
      --unsafe:    #f43f5e;
      --warn:      #f59e0b;
      --text:      #e2e8f0;
      --muted:     #64748b;
      --card-r:    16px;
    }

    body {
      font-family: 'Inter', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 40px 16px 80px;
    }

    header {
      text-align: center;
      margin-bottom: 40px;
    }
    header h1 {
      font-size: 2rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent-hi) 0%, #a78bfa 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      letter-spacing: -0.5px;
    }
    header p {
      color: var(--muted);
      margin-top: 8px;
      font-size: 0.9rem;
    }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--card-r);
      padding: 28px;
      width: 100%;
      max-width: 720px;
      margin-bottom: 20px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }

    label {
      display: block;
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 10px;
    }

    textarea {
      width: 100%;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 10px;
      color: var(--text);
      font-family: inherit;
      font-size: 1rem;
      padding: 14px 16px;
      resize: vertical;
      min-height: 100px;
      outline: none;
      transition: border-color 0.2s;
    }
    textarea:focus { border-color: var(--accent); }
    textarea::placeholder { color: var(--muted); }

    .controls {
      display: flex;
      gap: 12px;
      margin-top: 16px;
      flex-wrap: wrap;
      align-items: center;
    }

    button#analyze-btn {
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 10px;
      padding: 12px 28px;
      font-family: inherit;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s, transform 0.1s;
      letter-spacing: 0.02em;
    }
    button#analyze-btn:hover  { background: var(--accent-hi); }
    button#analyze-btn:active { transform: scale(0.97); }
    button#analyze-btn:disabled { opacity: 0.5; cursor: not-allowed; }

    .toggle-label {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.85rem;
      color: var(--muted);
      cursor: pointer;
      user-select: none;
    }
    .toggle-label input[type=checkbox] { accent-color: var(--accent); width: 16px; height: 16px; }

    .examples {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 14px;
    }
    .examples span { font-size: 0.78rem; color: var(--muted); align-self: center; }
    .example-chip {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 5px 14px;
      font-size: 0.8rem;
      cursor: pointer;
      transition: border-color 0.15s, color 0.15s;
      color: var(--text);
    }
    .example-chip:hover { border-color: var(--accent); color: var(--accent-hi); }

    /* Result card */
    #result-card { display: none; }

    .verdict {
      display: flex;
      align-items: center;
      gap: 14px;
      margin-bottom: 20px;
    }
    .verdict-badge {
      font-size: 1.4rem;
      font-weight: 700;
      padding: 6px 20px;
      border-radius: 99px;
      letter-spacing: 0.05em;
    }
    .verdict-badge.safe   { background: rgba(34,197,94,0.15); color: var(--safe); }
    .verdict-badge.unsafe { background: rgba(244,63,94,0.15);  color: var(--unsafe); }

    .verdict-meta { font-size: 0.8rem; color: var(--muted); }
    .verdict-meta strong { color: var(--text); }

    .section-title {
      font-size: 0.78rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 10px;
    }

    .tag-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 18px;
    }
    .tag {
      padding: 4px 12px;
      border-radius: 99px;
      font-size: 0.82rem;
      font-weight: 500;
    }
    .tag.word      { background: rgba(244,63,94,0.15);  color: var(--unsafe); border: 1px solid rgba(244,63,94,0.3); }
    .tag.cat       { background: rgba(245,158,11,0.15); color: var(--warn);   border: 1px solid rgba(245,158,11,0.3); }
    .tag.source-both    { background: rgba(99,102,241,0.15); color: var(--accent-hi); border: 1px solid rgba(99,102,241,0.3); }
    .tag.source-model   { background: rgba(99,102,241,0.1);  color: #a78bfa;          border: 1px solid rgba(167,139,250,0.3); }
    .tag.source-lexicon { background: rgba(34,197,94,0.1);   color: var(--safe);      border: 1px solid rgba(34,197,94,0.3); }

    /* Token stream */
    .token-stream {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 20px;
      padding: 16px;
      background: var(--surface2);
      border-radius: 10px;
      border: 1px solid var(--border);
    }
    .tok {
      padding: 3px 10px;
      border-radius: 6px;
      font-size: 0.88rem;
      font-family: 'Inter', monospace;
      background: var(--surface);
      border: 1px solid var(--border);
      transition: transform 0.15s;
      position: relative;
    }
    .tok:hover { transform: translateY(-2px); }
    .tok.flagged {
      background: rgba(244,63,94,0.18);
      border-color: rgba(244,63,94,0.45);
      color: #fda4af;
      font-weight: 600;
    }
    .tok .tip {
      display: none;
      position: absolute;
      bottom: calc(100% + 6px);
      left: 50%;
      transform: translateX(-50%);
      background: #1e2436;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px 10px;
      font-size: 0.72rem;
      white-space: nowrap;
      color: var(--text);
      z-index: 10;
      box-shadow: 0 4px 16px rgba(0,0,0,0.5);
      pointer-events: none;
    }
    .tok:hover .tip { display: block; }

    /* Span table */
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
    }
    th {
      text-align: left;
      color: var(--muted);
      font-weight: 500;
      padding: 6px 10px;
      border-bottom: 1px solid var(--border);
    }
    td {
      padding: 8px 10px;
      border-bottom: 1px solid rgba(42,49,80,0.5);
      vertical-align: middle;
    }
    tr:last-child td { border-bottom: none; }

    .pill {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 99px;
      font-size: 0.75rem;
      font-weight: 600;
    }
    .pill.sev-high   { background: rgba(244,63,94,0.2);  color: #f43f5e; }
    .pill.sev-medium { background: rgba(245,158,11,0.2); color: #f59e0b; }
    .pill.sev-low    { background: rgba(34,197,94,0.2);  color: #22c55e; }
    .pill.src-both    { background: rgba(99,102,241,0.2); color: #818cf8; }
    .pill.src-model   { background: rgba(167,139,250,0.2); color: #a78bfa; }
    .pill.src-lexicon { background: rgba(34,197,94,0.2);  color: #22c55e; }

    /* JSON view */
    details { margin-top: 16px; }
    summary {
      cursor: pointer;
      font-size: 0.8rem;
      color: var(--muted);
      user-select: none;
      padding: 6px 0;
    }
    summary:hover { color: var(--accent-hi); }
    pre {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
      font-size: 0.78rem;
      overflow-x: auto;
      color: #a5b4fc;
      margin-top: 8px;
      line-height: 1.5;
    }

    .spinner {
      display: none;
      width: 20px; height: 20px;
      border: 2px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.6s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    footer {
      margin-top: 40px;
      font-size: 0.75rem;
      color: var(--muted);
      text-align: center;
    }
  </style>
</head>
<body>

<header>
  <h1>⚡ Tamil Abusive Language Detector</h1>
  <p>BiLSTM-CRF sequence tagger · lexicon override · BIO span detection</p>
</header>

<!-- Input card -->
<div class="card">
  <label for="text-input">Input Text</label>
  <textarea id="text-input" placeholder="Type a Tamil / Tanglish / English sentence…"></textarea>

  <div class="controls">
    <button id="analyze-btn" onclick="analyze()">Analyze</button>
    <div class="spinner" id="spinner"></div>
    <label class="toggle-label">
      <input type="checkbox" id="full-toggle" /> Show token-level debug
    </label>
  </div>

  <div class="examples">
    <span>Try:</span>
    <div class="example-chip" onclick="setExample(this)">நீ ஒரு thevdiya da</div>
    <div class="example-chip" onclick="setExample(this)">super movie semma mass da</div>
    <div class="example-chip" onclick="setExample(this)">ASHIT THEVUDIYA Pula Breast umbi</div>
    <div class="example-chip" onclick="setExample(this)">avan nalla payan da</div>
  </div>
</div>

<!-- Result card -->
<div class="card" id="result-card">

  <div class="verdict" id="verdict-row"></div>

  <div id="words-section"></div>
  <div id="cats-section"></div>
  <div id="token-stream-section"></div>
  <div id="spans-section"></div>
  <details id="json-section">
    <summary>Raw JSON response</summary>
    <pre id="json-pre"></pre>
  </details>
</div>

<footer>CharCNN + FastText + BiLSTM + CRF · Lexicon override · Tamil/Tanglish/English</footer>

<script>
  const API = '/predict';

  function setExample(el) {
    document.getElementById('text-input').value = el.textContent;
  }

  function tag(cls, text) {
    return `<span class="tag ${cls}">${esc(text)}</span>`;
  }
  function pill(cls, text) {
    return text ? `<span class="pill ${cls}">${esc(text)}</span>` : '';
  }
  function esc(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  async function analyze() {
    const text = document.getElementById('text-input').value.trim();
    if (!text) return;

    const full = document.getElementById('full-toggle').checked;
    const btn  = document.getElementById('analyze-btn');
    const spin = document.getElementById('spinner');

    btn.disabled = true;
    spin.style.display = 'block';

    let data;
    try {
      const url = full ? `${API}?full=true` : API;
      const resp = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text})
      });
      data = await resp.json();
    } catch(e) {
      alert('Request failed: ' + e);
      return;
    } finally {
      btn.disabled = false;
      spin.style.display = 'none';
    }

    renderResult(data);
  }

  // also trigger on Ctrl+Enter
  document.getElementById('text-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) analyze();
  });

  function renderResult(d) {
    const rc = document.getElementById('result-card');
    rc.style.display = 'block';

    // Verdict
    const safe = d.safe;
    const badge = safe
      ? `<span class="verdict-badge safe">✓ SAFE</span>`
      : `<span class="verdict-badge unsafe">✗ UNSAFE</span>`;
    document.getElementById('verdict-row').innerHTML =
      `${badge}<div class="verdict-meta">
         <strong>${d.flagged_spans ? d.flagged_spans.length : 0}</strong> flagged token(s) · <strong>${d._ms} ms</strong>
       </div>`;

    // Flagged words
    const wSec = document.getElementById('words-section');
    if (d.flagged_words && d.flagged_words.length) {
      wSec.innerHTML = `<div class="section-title">Flagged Words</div>
        <div class="tag-row">${d.flagged_words.map(w => tag('word', w)).join('')}</div>`;
    } else {
      wSec.innerHTML = '';
    }

    // Categories
    const cSec = document.getElementById('cats-section');
    if (d.categories && d.categories.length) {
      cSec.innerHTML = `<div class="section-title">Categories</div>
        <div class="tag-row">${d.categories.map(c => tag('cat', c)).join('')}</div>`;
    } else {
      cSec.innerHTML = '';
    }

    // Token stream (only in full/debug mode or always)
    const tSec = document.getElementById('token-stream-section');
    if (d.tokens) {
      const flags = new Set((d.flagged_words || []));
      const spanMap = {};
      (d.flagged_spans || []).forEach(s => { spanMap[s.token] = s; });

      const toks = d.tokens.map(tok => {
        const flagged = flags.has(tok);
        const sp = spanMap[tok];
        let tip = '';
        if (sp) {
          tip = `<span class="tip">source: ${sp.source}` +
            (sp.canon ? ` · canon: ${sp.canon}` : '') +
            (sp.category ? ` · ${sp.category}` : '') +
            (sp.severity ? ` · ${sp.severity}` : '') + `</span>`;
        }
        return `<span class="tok${flagged?' flagged':''}">${esc(tok)}${tip}</span>`;
      }).join('');
      tSec.innerHTML = `<div class="section-title">Token Stream <span style="color:#3b4870;font-weight:400">(hover flagged for detail)</span></div>
        <div class="token-stream">${toks || '<em style="color:var(--muted)">No tokens</em>'}</div>`;
    } else {
      tSec.innerHTML = '';
    }

    // Span table
    const sSec = document.getElementById('spans-section');
    if (d.flagged_spans && d.flagged_spans.length) {
      const rows = d.flagged_spans.map(s => `
        <tr>
          <td><strong>${esc(s.token)}</strong></td>
          <td>${pill('src-'+s.source, s.source)}</td>
          <td>${esc(s.canon || '—')}</td>
          <td>${esc(s.category || '—')}</td>
          <td>${s.severity ? pill('sev-'+s.severity, s.severity) : '—'}</td>
          <td>${esc(s.match_type || '—')}</td>
        </tr>`).join('');
      sSec.innerHTML = `<div class="section-title">Flagged Spans</div>
        <table>
          <thead><tr>
            <th>Token</th><th>Source</th><th>Canon</th>
            <th>Category</th><th>Severity</th><th>Match</th>
          </tr></thead>
          <tbody>${rows}</tbody>
        </table>`;
    } else {
      sSec.innerHTML = '';
    }

    // JSON
    document.getElementById('json-pre').textContent = JSON.stringify(d, null, 2);
  }
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tamil abusive-language detector — CLI and web server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── input modes ──────────────────────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "text", nargs="?", metavar="TEXT",
        help="Text to analyze (single sentence).",
    )
    input_group.add_argument(
        "--stdin", action="store_true",
        help="Read sentences from stdin, one per line.",
    )
    input_group.add_argument(
        "--file", metavar="PATH",
        help="Read sentences from a text file, one per line.",
    )

    # ── server mode ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--web", action="store_true",
        help="Start Flask HTTP server instead of running CLI.",
    )
    parser.add_argument("--host", default="0.0.0.0", metavar="HOST",
                        help="Server bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, metavar="PORT",
                        help="Server bind port (default: 5000)")

    # ── output options ────────────────────────────────────────────────────────
    parser.add_argument(
        "--full", action="store_true",
        help="Include per-token debug fields (tokens, model_tags, lexicon_tags, merged_tags).",
    )
    parser.add_argument(
        "--compact", action="store_true",
        help="Output compact JSON (no pretty-print).",
    )
    parser.add_argument(
        "--censor", action="store_true",
        help="Apply smart auto-censoring and masking to the input.",
    )
    parser.add_argument(
        "--censor-mode", choices=["partial", "tag", "block", "polite"], default="partial",
        help="Masking mode for --censor: partial | tag | block | polite (default: partial).",
    )
    parser.add_argument(
        "--mask-char", default=None,
        help="Character to use for masking (default: '*' for partial, '█' for block).",
    )
    parser.add_argument(
        "--sensitivity", choices=["standard", "strict", "maximum"], default="standard",
        help="Detection sensitivity level: standard | strict | maximum (default: standard).",
    )

    # ── model options ─────────────────────────────────────────────────────────
    _ckpt = ROOT / "models" / "checkpoints" / "sequence_tagger" / "best.pt"
    _lex  = ROOT / "lexicon" / "abusive_lexicon.json"
    parser.add_argument("--checkpoint", type=Path, default=_ckpt,
                        metavar="PATH", help="Model checkpoint (default: best.pt)")
    parser.add_argument("--lexicon", type=Path, default=_lex,
                        metavar="PATH", help="Lexicon JSON path")
    parser.add_argument("--no-lexicon", action="store_true",
                        help="Disable lexicon override hook (model-only inference).")
    parser.add_argument(
        "--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu",
        help="Compute device: 'cpu' or 'cuda' (default: auto-detect).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.web:
        run_web(args)
    elif args.text or args.stdin or args.file:
        run_cli(args)
    else:
        # No input given: show help + interactive prompt
        parser.print_help()
        print("\n[inference] No input given — enter a sentence to analyze:")
        try:
            while True:
                try:
                    line = input(">>> ").strip()
                except EOFError:
                    break
                if not line:
                    continue
                bundle = get_bundle(args)
                import time
                t0 = time.perf_counter()
                from inference import predict, result_to_dict
                result = predict(line, bundle)
                ms = round((time.perf_counter() - t0) * 1000, 1)
                out = result_to_dict(result, full=args.full)
                out["_ms"] = ms
                print(json.dumps(out, ensure_ascii=False, indent=2))
        except KeyboardInterrupt:
            pass


# ---------------------------------------------------------------------------
# Module-level app for Gunicorn / production WSGI servers
# Usage:  gunicorn serve:app --bind 0.0.0.0:5000 --workers 1
# ---------------------------------------------------------------------------
# Only create the app when this module is imported by a WSGI server (not when
# run directly via `python serve.py`, which goes through main() instead).
if __name__ == "__main__":
    main()
else:
    app = create_app()

