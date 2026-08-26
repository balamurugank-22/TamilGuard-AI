import React from 'react';
import { Cpu, Layers, BookOpen, ShieldCheck, Database, CheckCircle2, Award, Zap, GitBranch } from 'lucide-react';

export default function ArchitectureDiagram() {
  const steps = [
    {
      num: '01',
      title: 'Unicode & Script Normalizer',
      subtitle: 'normalize.py',
      desc: 'NFC canonicalization, HTML entity decode, repeated character collapsing, and script boundary tagging (Tamil, Latin, Digits, Symbols).',
      badge: 'Deterministic'
    },
    {
      num: '02',
      title: 'Dual Feature Extractor',
      subtitle: 'FastText + CharCNN',
      desc: 'Subword FastText (1.7GB model) generates semantic word vectors while CharCNN (1D multi-kernel CNN) captures spelling variants & agglutinative morphology.',
      badge: '200D + 150D'
    },
    {
      num: '03',
      title: 'Bidirectional LSTM',
      subtitle: 'models/sequence_tagger.py',
      desc: '256-hidden units recurrent contextual encoder models long-range dependencies and syntactic framing (insult vs benign teasing).',
      badge: 'Deep Context'
    },
    {
      num: '04',
      title: 'CRF Transition Decoder',
      subtitle: 'torchcrf (Viterbi)',
      desc: 'Linear-chain Conditional Random Field models label transition constraints (e.g. I-ABUSE cannot follow O directly).',
      badge: 'BIO Tagging'
    },
    {
      num: '05',
      title: 'Lexicon Override Hook',
      subtitle: 'abusive_lexicon.json',
      desc: 'Parallel safety net with dynamic suffix stripping (-kku, -la, -nga) and phonetic variants, ensuring 100% recall on confirmed slurs.',
      badge: 'Safety Net'
    }
  ];

  return (
    <div className="space-y-6">
      
      {/* Benchmark Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        
        <div className="glass-panel p-5 rounded-2xl border border-zinc-800">
          <div className="text-xs uppercase font-bold tracking-wider text-zinc-500">Gold Test Abuse F1</div>
          <div className="text-3xl font-black text-indigo-500 mt-1">96.79%</div>
          <div className="text-[11px] text-zinc-500 mt-1">Token-level micro F1 on 500 gold test sentences</div>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-zinc-800">
          <div className="text-xs uppercase font-bold tracking-wider text-zinc-500">Precision Rate</div>
          <div className="text-3xl font-black text-emerald-500 mt-1">100.0%</div>
          <div className="text-[11px] text-zinc-500 mt-1">0 False Positives on benign colloquialisms & teasing</div>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-zinc-800">
          <div className="text-xs uppercase font-bold tracking-wider text-zinc-500">Sentence Unsafe F1</div>
          <div className="text-3xl font-black text-purple-500 mt-1">96.59%</div>
          <div className="text-[11px] text-zinc-500 mt-1">Derived sentence-level safety classification</div>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-zinc-800">
          <div className="text-xs uppercase font-bold tracking-wider text-zinc-500">Average Latency</div>
          <div className="text-3xl font-black text-amber-500 mt-1">~18ms</div>
          <div className="text-[11px] text-zinc-500 mt-1">End-to-end CPU inference per sentence</div>
        </div>

      </div>

      {/* Visual Pipeline Stages */}
      <div className="glass-panel p-6 rounded-2xl border border-zinc-800 space-y-6">
        
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-zinc-200 uppercase tracking-wider flex items-center gap-2">
              <Layers className="w-4 h-4 text-indigo-500" />
              End-to-End Neural Architecture & Override Flow
            </h3>
            <p className="text-xs text-zinc-500 mt-0.5">
              How raw Tamil / Tanglish text flows through normalization, subword feature extraction, and sequence decoding.
            </p>
          </div>
          <span className="text-xs font-mono text-zinc-500 bg-zinc-900 px-3 py-1 rounded-lg border border-zinc-800">
            7.6M Parameters
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 relative">
          {steps.map((s, idx) => (
            <div key={idx} className="bg-zinc-950 p-4 rounded-xl border border-zinc-800 space-y-3 relative group hover:border-indigo-500/50 transition-colors">
              <div className="flex items-center justify-between">
                <span className="text-xs font-black font-mono text-indigo-500">{s.num}</span>
                <span className="text-[9px] font-bold px-2 py-0.5 rounded bg-zinc-900 text-zinc-300 border border-zinc-800">
                  {s.badge}
                </span>
              </div>
              <div>
                <h4 className="text-xs font-bold text-zinc-100">{s.title}</h4>
                <span className="text-[10px] font-mono text-indigo-500/80 block">{s.subtitle}</span>
              </div>
              <p className="text-[11px] text-zinc-500 leading-relaxed">
                {s.desc}
              </p>
            </div>
          ))}
        </div>

      </div>

      {/* Dataset & Embeddings Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        
        <div className="glass-panel p-6 rounded-2xl border border-zinc-800 space-y-3">
          <div className="flex items-center gap-2 text-sm font-bold text-zinc-200 uppercase tracking-wider">
            <Database className="w-4 h-4 text-emerald-500" />
            FastText Subword Embeddings
          </div>
          <p className="text-xs text-zinc-500 leading-relaxed">
            Trained on a domain-specific Tamil and Tanglish corpus. Character n-grams (3-6 chars) capture phonetic spellings (e.g. <code>thevdiya</code>, <code>thevidiya</code>, <code>thevudya</code>) with shared vector clusters.
          </p>
          <div className="flex flex-wrap gap-2 pt-1 font-mono text-xs text-zinc-300">
            <span className="px-2.5 py-1 rounded-lg bg-zinc-900 border border-zinc-800">Dim: 200</span>
            <span className="px-2.5 py-1 rounded-lg bg-zinc-900 border border-zinc-800">Subwords: 3-6</span>
            <span className="px-2.5 py-1 rounded-lg bg-zinc-900 border border-zinc-800">Size: 1.74 GB</span>
          </div>
        </div>

        <div className="glass-panel p-6 rounded-2xl border border-zinc-800 space-y-3">
          <div className="flex items-center gap-2 text-sm font-bold text-zinc-200 uppercase tracking-wider">
            <BookOpen className="w-4 h-4 text-amber-500" />
            Multi-Tier Abusive Lexicon
          </div>
          <p className="text-xs text-zinc-500 leading-relaxed">
            275 curated keys covering sexual slurs, insults, threats, and profanity with dynamic Tanglish suffix-stripping (<code>-kku</code>, <code>-la</code>, <code>-nga</code>, <code>-oda</code>) for morphologically inflected forms.
          </p>
          <div className="flex flex-wrap gap-2 pt-1 font-mono text-xs text-zinc-300">
            <span className="px-2.5 py-1 rounded-lg bg-zinc-900 border border-zinc-800">272 Exact</span>
            <span className="px-2.5 py-1 rounded-lg bg-zinc-900 border border-zinc-800">Multi-word Spans</span>
            <span className="px-2.5 py-1 rounded-lg bg-zinc-900 border border-zinc-800">Phonetic Clusters</span>
          </div>
        </div>

      </div>

    </div>
  );
}
