import React, { useEffect, useCallback } from 'react';
import { X, ShieldAlert, Cpu, BookOpen, AlertTriangle, Layers, ArrowRight } from 'lucide-react';

export default function SpanDetailsModal({ span, tokenIndex, bioTag, onClose }) {
  // Close on Escape key
  const handleEscape = useCallback((e) => {
    if (e.key === 'Escape') onClose();
  }, [onClose]);

  useEffect(() => {
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [handleEscape]);

  if (!span) return null;

  const isModel = span.source === 'model' || span.source === 'both';
  const isLexicon = span.source === 'lexicon' || span.source === 'both';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fade-in-up">
      <div className="bg-zinc-950 border border-zinc-800 rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden animate-scaleUp">
        
        {/* Header */}
        <div className="p-5 border-b border-zinc-800 bg-gradient-to-r from-aurora-800 to-aurora-750 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-rose-500/20 text-rose-500 border border-rose-500/30">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-zinc-100 text-base">Token Inspection</h3>
              <p className="text-xs text-zinc-500 font-mono">Index #{tokenIndex} in tokenized sequence</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-white hover:bg-zinc-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body Content */}
        <div className="p-6 space-y-5">
          
          {/* Surface Word Spotlight */}
          <div className="flex items-center justify-between p-4 rounded-xl bg-black border border-zinc-800">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">Surface Token</span>
              <div className="text-2xl font-bold font-mono text-rose-500 mt-0.5">{span.token}</div>
            </div>
            <div className="text-right">
              <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">BIO Tag</span>
              <div className="text-sm font-bold font-mono px-2.5 py-1 rounded-lg bg-rose-500/20 text-rose-500 border border-rose-500/30 mt-1 inline-block">
                {bioTag || 'B-ABUSE'}
              </div>
            </div>
          </div>

          {/* Source Attribution Grid */}
          <div className="grid grid-cols-2 gap-3">
            
            {/* Neural System */}
            <div className={`p-3.5 rounded-xl border ${isModel ? 'bg-indigo-500/10 border-indigo-500/40' : 'bg-black border-zinc-800 opacity-50'}`}>
              <div className="flex items-center gap-2 mb-1">
                <Cpu className="w-4 h-4 text-indigo-500" />
                <span className="text-xs font-bold text-zinc-200">BiLSTM-CRF Neural</span>
              </div>
              <span className="text-[11px] text-zinc-500">
                {isModel ? 'Flagged via CharCNN + FastText contextual decoder' : 'Not triggered in neural pass'}
              </span>
            </div>

            {/* Lexicon Safety Net */}
            <div className={`p-3.5 rounded-xl border ${isLexicon ? 'bg-emerald-500/10 border-emerald-500/40' : 'bg-black border-zinc-800 opacity-50'}`}>
              <div className="flex items-center gap-2 mb-1">
                <BookOpen className="w-4 h-4 text-emerald-500" />
                <span className="text-xs font-bold text-zinc-200">Lexicon Safety Net</span>
              </div>
              <span className="text-[11px] text-zinc-500">
                {isLexicon ? `Match Type: ${span.match_type || 'exact'}` : 'No lexicon override'}
              </span>
            </div>

          </div>

          {/* Lexicon & Linguistic Properties */}
          <div className="space-y-2 bg-black/60 p-4 rounded-xl border border-zinc-800 text-xs">
            <div className="flex justify-between py-1 border-b border-zinc-800">
              <span className="text-zinc-500">Canonical Lemma</span>
              <span className="font-mono font-semibold text-zinc-200">{span.canon || '—'}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-zinc-800">
              <span className="text-zinc-500">Harm Category</span>
              <span className="font-semibold text-amber-500 capitalize">{span.category || 'General Abuse'}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-zinc-800">
              <span className="text-zinc-500">Severity Tier</span>
              <span className={`font-semibold capitalize px-2 py-0.5 rounded text-[11px] ${
                span.severity === 'high' ? 'bg-rose-500/20 text-rose-500' :
                span.severity === 'medium' ? 'bg-amber-500/20 text-amber-500' :
                'bg-emerald-500/20 text-emerald-500'
              }`}>
                {span.severity || 'high'}
              </span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-zinc-500">Detection Source Attribution</span>
              <span className="font-semibold text-zinc-200 uppercase tracking-wider text-[10px] bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
                {span.source}
              </span>
            </div>
          </div>

        </div>

        {/* Footer */}
        <div className="p-4 bg-zinc-900 border-t border-zinc-800 text-right">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-xl text-xs font-semibold transition-colors cursor-pointer"
          >
            Close Inspector
          </button>
        </div>

      </div>
    </div>
  );
}
