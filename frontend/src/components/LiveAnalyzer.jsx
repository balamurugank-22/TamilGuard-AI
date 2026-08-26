import React, { useState } from 'react';
import { Send, ShieldAlert, ShieldCheck, Trash2, Zap, Code2 } from 'lucide-react';
import { predictText } from '../services/api';
import PresetsBar from './PresetsBar';
import TokenVisualizer from './TokenVisualizer';
import SmartRedactor from './SmartRedactor';

const SENSITIVITY_OPTIONS = [
  {
    id: 'standard',
    label: 'Standard',
    hint: 'Neural + Exact Lexicon',
    color: 'indigo',
  },
  {
    id: 'strict',
    label: 'Strict',
    hint: 'Leetspeak + Fuzzy Match',
    color: 'amber',
  },
  {
    id: 'maximum',
    label: 'Maximum',
    hint: 'Zero Tolerance Mode',
    color: 'rose',
  },
];

const ACTIVE_COLOR = {
  indigo: 'border-indigo-500 bg-indigo-500/10 text-white',
  amber: 'border-amber-500  bg-amber-500/10  text-white',
  rose: 'border-rose-500   bg-rose-500/10   text-white',
};

export default function LiveAnalyzer() {
  const [inputText, setInputText] = useState('');
  const [sensitivity, setSensitivity] = useState('standard');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showJson, setShowJson] = useState(false);

  const handleAnalyze = async () => {
    if (!inputText.trim() || loading) return;
    setLoading(true);
    setResult(null);
    try {
      const data = await predictText(inputText, true, sensitivity);
      setResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => { setInputText(''); setResult(null); };

  const isAbusive = result && !result.safe;

  return (
    <div className="space-y-5">

      {/* Quick Test Presets */}
      <PresetsBar onSelectPreset={(p) => { setInputText(p.text); setResult(null); }} />

      {/* ── Input Card ──────────────────────────────────────────────── */}
      <div className="glass-panel rounded-2xl border border-zinc-800 p-5 space-y-4 animated-border hover:shadow-2xl hover:shadow-indigo-500/10 transition-all duration-300">

        {/* Header row */}
        <div className="flex items-center justify-between pb-2 border-b border-zinc-900">
          <span className="text-base font-bold text-zinc-200 uppercase tracking-wide">Type or paste a comment</span>
          {inputText && (
            <button
              onClick={handleClear}
              className="flex items-center gap-1.5 text-sm font-medium text-zinc-400 hover:text-rose-500 transition-colors cursor-pointer"
            >
              <Trash2 className="w-4 h-4" /> Clear
            </button>
          )}
        </div>

        {/* Textarea */}
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleAnalyze(); }}
          placeholder="Context matters. Give us a sentence…"
          rows={3}
          className="w-full bg-black border border-zinc-700 focus:border-blue-500 rounded-xl px-5 py-4 text-zinc-100 text-xl font-medium focus:outline-none focus:ring-4 focus:ring-blue-500/10 transition-all resize-none placeholder:text-zinc-500 shadow-inner"
        />

        {/* Sensitivity selector + Analyze button */}
        <div className="flex flex-col sm:flex-row sm:items-center gap-3">

          {/* Sensitivity pills */}
          <div className="flex items-center gap-1.5 flex-1">
            <Zap className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
            <span className="text-xs text-zinc-500 mr-1">Sensitivity:</span>
            {SENSITIVITY_OPTIONS.map((opt) => {
              const active = sensitivity === opt.id;
              return (
                <button
                  key={opt.id}
                  onClick={() => setSensitivity(opt.id)}
                  title={opt.hint}
                  className={`px-3 py-1 rounded-full text-xs font-semibold border transition-all cursor-pointer ${active
                    ? ACTIVE_COLOR[opt.color]
                    : 'border-zinc-800 text-zinc-500 hover:border-zinc-500 hover:text-zinc-200 bg-transparent'
                    }`}
                >
                  {opt.label}
                </button>
              );
            })}
          </div>

          {/* Analyze */}
          <button
            onClick={handleAnalyze}
            disabled={loading || !inputText.trim()}
            className="flex items-center justify-center gap-2 px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl text-lg shadow-md transition-all disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
          >
            {loading ? (
              <><div className="w-5 h-5 border-2 border-black/30 border-t-white rounded-full animate-spin" /> Analyzing…</>
            ) : (
              <><Send className="w-5 h-5" /> Analyze</>
            )}
          </button>
        </div>

        <p className="text-sm text-zinc-400 font-medium pt-2">
          Ctrl + Enter to analyze · Supports Tamil Unicode, Tanglish, English & obfuscated text
        </p>
      </div>

      {/* ── Result ──────────────────────────────────────────────────── */}
      {result && (
        <div className="space-y-4 animate-fade-in-up">

          {/* Status banner */}
          <div className={`rounded-2xl border p-5 flex flex-col sm:flex-row sm:items-center gap-4 transition-all ${result.safe
            ? 'border-emerald-500/40 bg-emerald-500/5 animated-border'
            : 'border-rose-500/40 bg-rose-500/5 animated-border'
            }`}>
            <div className={`p-3 rounded-xl shrink-0 ${result.safe
              ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/25'
              : 'bg-rose-500/15 text-rose-400 border border-rose-500/25'
              }`}>
              {result.safe ? <ShieldCheck className="w-7 h-7" /> : <ShieldAlert className="w-7 h-7" />}
            </div>

            <div className="flex-1 min-w-0">
              <div className="flex flex-wrap items-center gap-3 mb-2">
                <h2 className={`text-2xl font-black tracking-tight ${result.safe ? 'text-emerald-600' : 'text-rose-600'}`}>
                  {result.safe ? 'Safe' : 'Abusive Content Detected'}
                </h2>
                <span className="text-sm font-mono px-3 py-1 rounded-full bg-zinc-900 border border-zinc-800 text-zinc-400 font-medium">
                  {result._ms || result._client_ms || '–'}ms
                </span>
              </div>

              <p className="text-base text-zinc-400 font-medium">
                {result.safe
                  ? 'No abusive words detected.'
                  : `${result.flagged_words.length} flagged span${result.flagged_words.length !== 1 ? 's' : ''} detected.`}
              </p>

              {/* Category tags */}
              {isAbusive && result.categories.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-2">
                  {result.categories.map((cat, i) => (
                    <span key={i} className="px-2.5 py-0.5 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/25 text-xs font-semibold capitalize">
                      {cat}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Token Visualizer */}
          <TokenVisualizer result={result} />

          {/* Smart Redactor */}
          <SmartRedactor
            originalText={inputText}
            inferenceResult={result}
            onApplyText={(t) => setInputText(t)}
            sensitivity={sensitivity}
          />

          {/* Raw JSON (collapsed) */}
          <div className="rounded-xl border border-zinc-800 overflow-hidden bg-black shadow-sm">
            <button
              onClick={() => setShowJson(!showJson)}
              className="w-full flex items-center justify-between px-5 py-4 text-sm font-bold text-zinc-400 hover:text-zinc-100 hover:bg-zinc-950 transition-colors cursor-pointer"
            >
              <span className="flex items-center gap-2"><Code2 className="w-5 h-5 text-indigo-500" /> Raw JSON Response</span>
              <span className="font-mono text-indigo-500 font-semibold">{showJson ? '▲ Hide' : '▼ Show'}</span>
            </button>
            {showJson && (
              <pre className="px-5 pb-5 pt-2 text-sm font-mono text-indigo-700 overflow-x-auto leading-relaxed bg-black border-t border-zinc-900">
                {JSON.stringify(result, null, 2)}
              </pre>
            )}
          </div>

        </div>
      )}
    </div>
  );
}
