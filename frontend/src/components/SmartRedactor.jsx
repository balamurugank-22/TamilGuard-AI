import React, { useState, useEffect } from 'react';
import { 
  ShieldCheck, 
  ShieldAlert, 
  Copy, 
  Check, 
  Sparkles, 
  Eye, 
  EyeOff, 
  Sliders, 
  RefreshCw,
  Hash,
  Tag,
  Square,
  HeartHandshake
} from 'lucide-react';
import { censorText } from '../services/api';

export default function SmartRedactor({ originalText, inferenceResult, onApplyText, sensitivity = 'standard' }) {
  const [mode, setMode] = useState('partial'); // 'partial' | 'tag' | 'block' | 'polite'
  const [severityFilter, setSeverityFilter] = useState('all');
  const [loading, setLoading] = useState(false);
  const [censorResult, setCensorResult] = useState(null);
  const [copied, setCopied] = useState(false);
  const [copiedPolite, setCopiedPolite] = useState(false);

  // Fetch or compute censored output whenever text, mode, severity, or sensitivity changes
  useEffect(() => {
    if (!originalText || !originalText.trim()) return;

    let isMounted = true;
    setLoading(true);

    censorText(originalText, {
      mode,
      severity_threshold: severityFilter,
      sensitivity,
    }).then((data) => {
      if (isMounted) {
        setCensorResult(data);
        setLoading(false);
      }
    }).catch(() => {
      if (isMounted) setLoading(false);
    });

    return () => {
      isMounted = false;
    };
  }, [originalText, mode, severityFilter, sensitivity, inferenceResult]);

  const handleCopy = (text, setCopyState) => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopyState(true);
    setTimeout(() => setCopyState(false), 2000);
  };

  if (!originalText || !inferenceResult) return null;

  const isSafe = inferenceResult.safe;

  const modes = [
    {
      id: 'partial',
      label: 'Subtle Mask',
      desc: 'e.g. th***ya / தே***யா',
      icon: Hash,
      color: 'text-amber-400',
      border: 'hover:border-amber-500/50',
      activeBg: 'bg-amber-500/10 border-amber-500/40 text-amber-300',
    },
    {
      id: 'tag',
      label: 'Category Tag',
      desc: 'e.g. [REDACTED: SLUR]',
      icon: Tag,
      color: 'text-rose-400',
      border: 'hover:border-rose-500/50',
      activeBg: 'bg-rose-500/10 border-rose-500/40 text-rose-300',
    },
    {
      id: 'block',
      label: 'Full Block',
      desc: 'e.g. ████████',
      icon: Square,
      color: 'text-indigo-400',
      border: 'hover:border-indigo-500/50',
      activeBg: 'bg-indigo-500/10 border-indigo-500/40 text-indigo-300',
    },
    {
      id: 'polite',
      label: 'Polite Rephrase',
      desc: 'Respectful alternative',
      icon: HeartHandshake,
      color: 'text-emerald-400',
      border: 'hover:border-emerald-500/50',
      activeBg: 'bg-emerald-500/10 border-emerald-500/40 text-emerald-300',
    },
  ];

  return (
    <div className="glass-panel p-6 rounded-2xl border border-zinc-800 space-y-6 shadow-xl relative overflow-hidden">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-zinc-800/60">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-gradient-to-br from-indigo-500/20 to-purple-500/20 border border-indigo-500/40 text-indigo-500">
            <Sparkles className="w-5 h-5 text-indigo-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-base font-bold text-zinc-100 tracking-wide">
                Smart Redaction & Auto-Censoring Studio
              </h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                API: /censor
              </span>
            </div>
            <p className="text-xs text-zinc-500">
              Live token-level masking, category-aware redaction, and polite sentence reconstruction
            </p>
          </div>
        </div>

        {/* Severity Filter Dropdown */}
        <div className="flex items-center gap-2 self-start sm:self-auto">
          <span className="text-xs text-zinc-500 font-medium flex items-center gap-1">
            <Sliders className="w-3.5 h-3.5 text-zinc-500" />
            Threshold:
          </span>
          <select
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            className="bg-zinc-900 border border-zinc-800 rounded-lg px-2.5 py-1 text-xs text-zinc-200 focus:outline-none focus:border-indigo-500 cursor-pointer"
          >
            <option value="all">All Severities (Low, Med, High)</option>
            <option value="medium">Medium + High Only</option>
            <option value="high">High Severity Only</option>
          </select>
        </div>
      </div>

      {/* Mode Selector Buttons */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {modes.map((m) => {
          const Icon = m.icon;
          const isActive = mode === m.id;
          return (
            <button
              key={m.id}
              onClick={() => setMode(m.id)}
              className={`flex flex-col items-start p-3.5 rounded-xl border text-left transition-all cursor-pointer ${
                isActive
                  ? m.activeBg + ' shadow-lg shadow-black/40'
                  : 'bg-zinc-950/60 border-zinc-800 text-zinc-500 hover:bg-zinc-900 ' + m.border
              }`}
            >
              <div className="flex items-center gap-2 mb-1.5 w-full justify-between">
                <span className="font-semibold text-xs text-zinc-200 flex items-center gap-1.5">
                  <Icon className={`w-3.5 h-3.5 ${m.color}`} />
                  {m.label}
                </span>
                {isActive && (
                  <span className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
                )}
              </div>
              <span className="text-[11px] text-zinc-500 font-mono">
                {m.desc}
              </span>
            </button>
          );
        })}
      </div>

      {/* Output Comparison Display Card */}
      {isSafe ? (
        <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center gap-3 text-emerald-400 text-sm">
          <ShieldCheck className="w-5 h-5 flex-shrink-0" />
          <span>Text is already <strong>safe and clean</strong>. No censoring or redactions required.</span>
        </div>
      ) : (
        <div className="space-y-4">
          
          {/* Censored Result Box */}
          <div className="p-5 rounded-xl bg-black/90 border border-zinc-800 space-y-3 relative group">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-zinc-500">
                  Moderated / Censored Output ({mode.toUpperCase()})
                </span>
                {censorResult?.redacted_count > 0 && (
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-rose-500/20 text-rose-300 border border-rose-500/30">
                    {censorResult.redacted_count} token{censorResult.redacted_count > 1 ? 's' : ''} masked
                  </span>
                )}
              </div>

              <div className="flex items-center gap-2">
                {onApplyText && censorResult?.censored && (
                  <button
                    onClick={() => onApplyText(censorResult.censored)}
                    className="text-xs text-indigo-500 hover:text-indigo-300 px-2.5 py-1 rounded-md bg-indigo-500/10 border border-indigo-500/30 transition-colors flex items-center gap-1 cursor-pointer"
                  >
                    Apply to Input
                  </button>
                )}
                <button
                  onClick={() => handleCopy(censorResult?.censored, setCopied)}
                  className="text-xs text-zinc-300 hover:text-white px-2.5 py-1 rounded-md bg-zinc-900 border border-zinc-800 hover:border-zinc-9500 transition-all flex items-center gap-1.5 cursor-pointer"
                >
                  {copied ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                      <span className="text-emerald-400">Copied!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Rendered Text */}
            <div className="p-3.5 rounded-lg bg-zinc-950 border border-zinc-800/80 font-mono text-sm leading-relaxed text-zinc-200 break-words">
              {loading ? (
                <div className="flex items-center gap-2 text-zinc-500 py-1">
                  <RefreshCw className="w-4 h-4 animate-spin text-indigo-500" />
                  Generating redaction...
                </div>
              ) : (
                censorResult?.censored || originalText
              )}
            </div>
          </div>

          {/* Polite Rephrasing Card */}
          {censorResult?.polite_suggestion && censorResult.polite_suggestion !== originalText && (
            <div className="p-4 rounded-xl bg-gradient-to-r from-emerald-950/40 to-aurora-900 border border-emerald-500/30 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-emerald-400">
                  <HeartHandshake className="w-4 h-4" />
                  <span className="text-xs font-bold uppercase tracking-wider">
                    Polite / Constructive Rephrasing Suggestion
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  {onApplyText && (
                    <button
                      onClick={() => onApplyText(censorResult.polite_suggestion)}
                      className="text-xs text-emerald-400 hover:text-emerald-300 px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 transition-colors cursor-pointer"
                    >
                      Use Suggestion
                    </button>
                  )}
                  <button
                    onClick={() => handleCopy(censorResult.polite_suggestion, setCopiedPolite)}
                    className="text-xs text-zinc-300 hover:text-white px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 transition-colors flex items-center gap-1 cursor-pointer"
                  >
                    {copiedPolite ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                    {copiedPolite ? 'Copied' : 'Copy'}
                  </button>
                </div>
              </div>

              <p className="text-xs text-zinc-300 font-sans italic bg-black/60 p-2.5 rounded-lg border border-emerald-500/20">
                "{censorResult.polite_suggestion}"
              </p>
            </div>
          )}

          {/* Redacted Tokens Breakdown Table */}
          {censorResult?.redacted_spans?.length > 0 && (
            <div className="space-y-2">
              <span className="text-xs font-bold uppercase tracking-wider text-zinc-500">
                Redacted Spans Summary
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5">
                {censorResult.redacted_spans.map((span, idx) => (
                  <div
                    key={idx}
                    className="p-2.5 rounded-lg bg-zinc-950 border border-zinc-800 flex items-center justify-between text-xs font-mono"
                  >
                    <div className="flex items-center gap-2 truncate">
                      <span className="text-rose-400 line-through truncate max-w-[90px]">{span.original_token}</span>
                      <span className="text-zinc-500">→</span>
                      <span className="text-emerald-400 font-bold truncate max-w-[90px]">{span.redacted_token}</span>
                    </div>
                    {span.category && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-900 text-zinc-500 border border-zinc-800 ml-2 uppercase">
                        {span.category}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      )}

    </div>
  );
}
