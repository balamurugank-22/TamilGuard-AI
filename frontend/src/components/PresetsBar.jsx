import React from 'react';
import { Sparkles, Tag } from 'lucide-react';
import { TEST_PRESETS } from '../data/testPresets';

export default function PresetsBar({ onSelectPreset }) {
  return (
    <div className="glass-panel p-4 rounded-2xl border border-zinc-800 mb-6">
      <div className="flex items-center gap-2 mb-3">
        <Sparkles className="w-4 h-4 text-indigo-500" />
        <h3 className="text-xs uppercase tracking-wider font-bold text-zinc-300">
          Benchmark & Test Presets
        </h3>
        <span className="text-[11px] text-zinc-500">
          (Click any chip to test edge cases, variants, or safe teasing)
        </span>
      </div>

      <div className="flex flex-wrap gap-2">
        {TEST_PRESETS.map((preset) => {
          const isSafe = preset.type === 'safe';
          return (
            <button
              key={preset.id}
              onClick={() => onSelectPreset(preset)}
              className={`group flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium border transition-all duration-200 text-left ${
                isSafe
                  ? 'bg-zinc-900/80 hover:bg-emerald-500/10 border-zinc-800 hover:border-emerald-500/50 text-zinc-300 hover:text-emerald-500'
                  : 'bg-zinc-900/80 hover:bg-rose-500/10 border-zinc-800 hover:border-rose-500/50 text-zinc-300 hover:text-rose-500'
              }`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  isSafe ? 'bg-emerald-500' : 'bg-rose-500'
                }`}
              />
              <span className="font-semibold">{preset.title}</span>
              <span className="text-[10px] text-zinc-500 group-hover:text-zinc-300 font-mono">
                [{preset.category}]
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
