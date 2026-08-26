import React, { useState } from 'react';
import { Play, Download, Trash2, Filter, ShieldCheck, ShieldAlert, FileText, CheckCircle2, XCircle } from 'lucide-react';
import { predictText } from '../services/api';
import { TEST_PRESETS } from '../data/testPresets';

export default function BatchTester() {
  const [textLines, setTextLines] = useState(
    TEST_PRESETS.map(p => p.text).join('\n')
  );
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState([]);
  const [filter, setFilter] = useState('all'); // all, safe, unsafe

  const handleRunBatch = async () => {
    const lines = textLines.split('\n').map(l => l.trim()).filter(Boolean);
    if (lines.length === 0 || running) return;

    setRunning(true);
    setProgress(0);
    const batchResults = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      try {
        const res = await predictText(line, false);
        batchResults.push({ line, ...res });
      } catch (err) {
        batchResults.push({ line, safe: true, flagged_words: [], categories: [], error: true });
      }
      setProgress(Math.round(((i + 1) / lines.length) * 100));
    }

    setResults(batchResults);
    setRunning(false);
  };

  const handleExportCSV = () => {
    if (results.length === 0) return;
    const headers = ['Sentence', 'Verdict', 'Flagged Words', 'Categories', 'Latency (ms)'];
    const rows = results.map(r => [
      `"${r.line.replace(/"/g, '""')}"`,
      r.safe ? 'SAFE' : 'UNSAFE',
      `"${(r.flagged_words || []).join('; ')}"`,
      `"${(r.categories || []).join('; ')}"`,
      r._ms || 0
    ]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `tamilguard_batch_evaluation_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const filteredResults = results.filter(r => {
    if (filter === 'safe') return r.safe;
    if (filter === 'unsafe') return !r.safe;
    return true;
  });

  const safeCount = results.filter(r => r.safe).length;
  const unsafeCount = results.filter(r => !r.safe).length;

  return (
    <div className="space-y-6">
      
      {/* Input Batch Box */}
      <div className="glass-panel p-6 rounded-2xl border border-zinc-800 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-zinc-200 uppercase tracking-wider flex items-center gap-2">
              <FileText className="w-4 h-4 text-indigo-500" />
              Batch Content Evaluation
            </h3>
            <p className="text-xs text-zinc-500 mt-0.5">
              Enter one sentence per line to evaluate in high-throughput batch mode.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setTextLines(TEST_PRESETS.map(p => p.text).join('\n'))}
              className="text-xs font-semibold px-3 py-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-zinc-300 transition-colors"
            >
              Load Presets
            </button>
            <button
              onClick={() => { setTextLines(''); setResults([]); }}
              className="text-xs text-zinc-500 hover:text-rose-400 p-1.5 transition-colors"
              title="Clear text"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        <textarea
          value={textLines}
          onChange={(e) => setTextLines(e.target.value)}
          placeholder="Enter sentences (one per line)..."
          rows={6}
          className="w-full bg-black/90 border border-zinc-800 focus:border-indigo-500 rounded-xl p-4 text-zinc-100 text-xs font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500/20 transition-all resize-y"
        />

        <div className="flex items-center justify-between">
          <span className="text-xs text-zinc-500 font-mono">
            {textLines.split('\n').filter(l => l.trim()).length} line(s) queued
          </span>

          <button
            onClick={handleRunBatch}
            disabled={running || !textLines.trim()}
            className="flex items-center gap-2 px-6 py-2.5 bg-gradient-to-r from-indigo-500 to-purple-500 hover:from-indigo-500/90 text-white font-semibold rounded-xl text-xs shadow-lg shadow-indigo-500/25 transition-all disabled:opacity-50 cursor-pointer"
          >
            <Play className="w-4 h-4 fill-current" />
            <span>{running ? `Processing (${progress}%)...` : 'Run Batch Analysis'}</span>
          </button>
        </div>

        {/* Progress Bar */}
        {running && (
          <div className="w-full bg-black rounded-full h-2 overflow-hidden border border-zinc-800">
            <div
              className="bg-gradient-to-r from-indigo-500 to-purple-500 h-full transition-all duration-200"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        )}
      </div>

      {/* Batch Results Table */}
      {results.length > 0 && (
        <div className="glass-panel p-6 rounded-2xl border border-zinc-800 space-y-4 animate-fade-in-up">
          
          {/* Summary Metric Header */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-zinc-800 pb-4">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5 text-xs font-bold font-mono px-3 py-1.5 rounded-xl bg-emerald-500/15 text-emerald-500 border border-emerald-500/30">
                <CheckCircle2 className="w-4 h-4" />
                <span>{safeCount} SAFE</span>
              </div>
              <div className="flex items-center gap-1.5 text-xs font-bold font-mono px-3 py-1.5 rounded-xl bg-rose-500/15 text-rose-500 border border-rose-500/30">
                <XCircle className="w-4 h-4" />
                <span>{unsafeCount} UNSAFE</span>
              </div>
              <span className="text-xs text-zinc-500 font-mono">
                Total: {results.length} sentences
              </span>
            </div>

            <div className="flex items-center gap-2">
              {/* Filter Tabs */}
              <div className="flex bg-black p-1 rounded-lg border border-zinc-800 text-xs">
                {['all', 'safe', 'unsafe'].map((f) => (
                  <button
                    key={f}
                    onClick={() => setFilter(f)}
                    className={`px-3 py-1 rounded-md capitalize font-semibold transition-all ${
                      filter === f ? 'bg-zinc-800 text-white' : 'text-zinc-500 hover:text-zinc-200'
                    }`}
                  >
                    {f}
                  </button>
                ))}
              </div>

              {/* Export CSV Button */}
              <button
                onClick={handleExportCSV}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-zinc-900 hover:bg-zinc-800 text-zinc-200 border border-zinc-800 rounded-lg text-xs font-semibold transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Export CSV</span>
              </button>
            </div>
          </div>

          {/* Results Table */}
          <div className="overflow-x-auto rounded-xl border border-zinc-800 bg-black/60">
            <table className="w-full text-left text-xs">
              <thead className="bg-zinc-900 text-zinc-500 font-bold uppercase tracking-wider border-b border-zinc-800">
                <tr>
                  <th className="px-4 py-3 w-16">Status</th>
                  <th className="px-4 py-3">Sentence Text</th>
                  <th className="px-4 py-3">Flagged Words</th>
                  <th className="px-4 py-3">Categories</th>
                  <th className="px-4 py-3 text-right">Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-aurora-750">
                {filteredResults.map((item, idx) => (
                  <tr key={idx} className="hover:bg-zinc-900/40 transition-colors">
                    <td className="px-4 py-3">
                      {item.safe ? (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/15 text-emerald-500 border border-emerald-500/30">
                          SAFE
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-500/15 text-rose-500 border border-rose-500/30">
                          UNSAFE
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 font-medium text-zinc-200 max-w-md truncate">
                      {item.line}
                    </td>
                    <td className="px-4 py-3 font-mono text-rose-500 font-semibold">
                      {(item.flagged_words || []).join(', ') || '—'}
                    </td>
                    <td className="px-4 py-3 capitalize text-amber-500">
                      {(item.categories || []).join(', ') || '—'}
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-zinc-500">
                      {item._ms || 0}ms
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

        </div>
      )}

    </div>
  );
}
