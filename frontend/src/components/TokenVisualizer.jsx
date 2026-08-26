import React, { useState } from 'react';
import { ShieldAlert, ShieldCheck, Tag, Info, AlertTriangle } from 'lucide-react';
import SpanDetailsModal from './SpanDetailsModal';

export default function TokenVisualizer({ result }) {
  const [selectedSpan, setSelectedSpan] = useState(null);

  if (!result || !result.tokens || result.tokens.length === 0) {
    return null;
  }

  const { tokens, merged_tags = [], flagged_spans = [] } = result;

  // Map tokens to span metadata
  const spanMap = {};
  flagged_spans.forEach(span => {
    spanMap[span.token.toLowerCase()] = span;
  });

  return (
    <div className="glass-panel p-6 rounded-2xl border border-zinc-800 space-y-6">
      
      {/* Header Info */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-bold text-zinc-200 uppercase tracking-wider flex items-center gap-2">
            <Tag className="w-4 h-4 text-indigo-500" />
            Interactive Sequence Token Map
          </h3>
          <p className="text-xs text-zinc-500 mt-0.5">
            Click any highlighted token to inspect CharCNN embeddings, BIO transitions, and lexicon matching logic.
          </p>
        </div>
        <span className="text-xs font-mono text-zinc-500 bg-zinc-900 px-3 py-1 rounded-lg border border-zinc-800">
          {tokens.length} tokens
        </span>
      </div>

      {/* Visual Token Stream */}
      <div className="p-5 rounded-xl bg-black/80 border border-zinc-800 min-h-[100px] flex flex-wrap gap-2.5 items-center">
        {tokens.map((token, idx) => {
          const bio = merged_tags[idx] || 'O';
          const isFlagged = bio !== 'O';
          const spanData = isFlagged ? (spanMap[token.toLowerCase()] || { token, source: 'model' }) : null;

          return (
            <div key={idx} className="relative group">
              <button
                type="button"
                onClick={() => isFlagged && setSelectedSpan({ ...spanData, index: idx, bio })}
                className={`relative px-4 py-2.5 rounded-xl font-mono text-base transition-all duration-200 flex flex-col items-center gap-1 ${
                  isFlagged
                    ? 'bg-rose-500/15 text-rose-400 border-2 border-rose-500/60 hover:border-rose-500 shadow-sm shadow-rose-500/10 cursor-pointer'
                    : 'bg-black text-zinc-100 border border-zinc-700 hover:border-zinc-500 cursor-default'
                }`}
              >
                <span className="font-bold">{token}</span>
                <span className={`text-[10px] uppercase tracking-widest font-bold px-1.5 py-0.5 rounded ${
                  isFlagged ? 'bg-rose-500 text-white' : 'text-zinc-500'
                }`}>
                  {bio}
                </span>
              </button>

              {/* Hover Tooltip for Flagged Items */}
              {isFlagged && spanData && (
                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:flex flex-col items-center pointer-events-none z-30">
                  <div className="bg-zinc-900 text-white text-[11px] font-sans rounded-xl p-2.5 shadow-2xl border border-zinc-800 whitespace-nowrap space-y-1">
                    <div className="flex items-center gap-1.5 font-bold text-rose-500">
                      <AlertTriangle className="w-3.5 h-3.5" />
                      <span>{spanData.category || 'Abuse Flagged'}</span>
                    </div>
                    <div className="text-zinc-300 font-mono text-[10px]">
                      Source: <span className="text-indigo-500 font-bold">{spanData.source}</span>
                    </div>
                    {spanData.canon && (
                      <div className="text-zinc-500 text-[10px]">
                        Lemma: <span className="text-zinc-200">{spanData.canon}</span>
                      </div>
                    )}
                  </div>
                  <div className="w-2 h-2 bg-zinc-900 rotate-45 -mt-1 border-r border-b border-zinc-800"></div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Flagged Tokens Summary Table (if any) */}
      {flagged_spans.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-zinc-800 bg-zinc-950/60">
          <table className="w-full text-left text-xs">
            <thead className="bg-zinc-900/80 text-zinc-500 font-bold uppercase tracking-wider border-b border-zinc-800">
              <tr>
                <th className="px-4 py-3">Flagged Token</th>
                <th className="px-4 py-3">Detection Source</th>
                <th className="px-4 py-3">Canonical Lemma</th>
                <th className="px-4 py-3">Category</th>
                <th className="px-4 py-3">Severity</th>
                <th className="px-4 py-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800 font-mono">
              {flagged_spans.map((span, idx) => (
                <tr key={idx} className="hover:bg-zinc-900/50 transition-colors">
                  <td className="px-4 py-3 font-bold text-rose-500">{span.token}</td>
                  <td className="px-4 py-3 font-sans">
                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase ${
                      span.source === 'both' ? 'bg-indigo-500/20 text-indigo-500 border border-indigo-500/40' :
                      span.source === 'model' ? 'bg-purple-500/20 text-purple-500 border border-purple-500/40' :
                      'bg-emerald-500/20 text-emerald-500 border border-emerald-500/40'
                    }`}>
                      {span.source}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-zinc-300">{span.canon || '—'}</td>
                  <td className="px-4 py-3 font-sans text-amber-500 capitalize font-semibold">{span.category || 'General'}</td>
                  <td className="px-4 py-3 font-sans">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                      span.severity === 'high' ? 'bg-rose-500/20 text-rose-400' :
                      span.severity === 'medium' ? 'bg-amber-500/20 text-amber-400' :
                      'bg-emerald-500/20 text-emerald-400'
                    }`}>
                      {span.severity || 'high'}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right font-sans">
                    <button
                      onClick={() => setSelectedSpan({ ...span, index: idx })}
                      className="text-indigo-500 hover:text-purple-500 font-semibold hover:underline"
                    >
                      Inspect
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Span Details Modal */}
      {selectedSpan && (
        <SpanDetailsModal
          span={selectedSpan}
          tokenIndex={selectedSpan.index}
          bioTag={selectedSpan.bio}
          onClose={() => setSelectedSpan(null)}
        />
      )}

    </div>
  );
}
