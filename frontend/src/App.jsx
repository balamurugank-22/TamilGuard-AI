import React, { useState } from 'react';
import Navbar from './components/Navbar';
import LiveAnalyzer from './components/LiveAnalyzer';
import BatchTester from './components/BatchTester';
import ArchitectureDiagram from './components/ArchitectureDiagram';

export default function App() {
  const [activeTab, setActiveTab] = useState('live');

  return (
    <div className="min-h-screen bg-black text-zinc-100 flex flex-col selection:bg-indigo-500">
      
      {/* Top Header & Navigation */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Tab View Routing */}
        {activeTab === 'live' && <LiveAnalyzer />}
        {activeTab === 'batch' && <BatchTester />}
        {activeTab === 'arch' && <ArchitectureDiagram />}

      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-800 bg-zinc-950/60 py-6 text-center text-xs text-zinc-500 font-mono">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>TamilGuard AI · CharFastText + BiLSTM-CRF + Lexicon Safety Net</span>
          <span className="text-zinc-500">Trained on Tamil & Tanglish Social Corpus</span>
        </div>
      </footer>

    </div>
  );
}
