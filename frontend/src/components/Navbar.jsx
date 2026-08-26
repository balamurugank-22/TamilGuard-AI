import React, { useState, useEffect } from 'react';
import { ShieldAlert, ShieldCheck, Activity, Settings, RefreshCw, Cpu, Database, CheckCircle2, XCircle, Menu, X } from 'lucide-react';
import { checkBackendHealth, getApiBase, setApiBase } from '../services/api';

export default function Navbar({ activeTab, setActiveTab }) {
  const [health, setHealth] = useState({ online: false, checking: true });
  const [showConfig, setShowConfig] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [endpointInput, setEndpointInput] = useState(getApiBase());

  const checkStatus = async () => {
    setHealth(prev => ({ ...prev, checking: true }));
    const status = await checkBackendHealth();
    setHealth({ ...status, checking: false });
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleSaveEndpoint = (e) => {
    e.preventDefault();
    setApiBase(endpointInput);
    setShowConfig(false);
    checkStatus();
  };

  const tabs = [
    { id: 'live', label: 'Live Console' },
    { id: 'batch', label: 'Batch Suite' },
    { id: 'arch', label: 'Architecture & Metrics' },
  ];

  const handleTabClick = (tabId) => {
    setActiveTab(tabId);
    setMobileMenuOpen(false);
  };

  return (
    <header className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Brand Logo & Model Tag */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center shadow-lg shadow-indigo-500/25">
              <ShieldAlert className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-2xl tracking-tight text-zinc-100">
                  TamilGuard AI
                </span>
                <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-500 border border-indigo-500/30 hidden sm:inline-block">
                  BiLSTM-CRF
                </span>
              </div>
              <p className="text-xs text-zinc-500 font-medium hidden sm:block">Neural Tamil & Tanglish Content Safety</p>
            </div>
          </div>

          {/* Desktop Navigation Tabs */}
          <nav className="hidden md:flex items-center gap-1 bg-zinc-900 p-1 rounded-xl border border-zinc-800">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => handleTabClick(tab.id)}
                className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-all cursor-pointer ${
                  activeTab === tab.id
                    ? 'bg-indigo-500 text-white shadow-md shadow-indigo-500/30'
                    : 'text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>

          {/* Health Status & Settings + Mobile Hamburger */}
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 text-xs">
              <span className="relative flex h-2 w-2">
                {health.online ? (
                  <>
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-500 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                  </>
                ) : (
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500"></span>
                )}
              </span>
              <span className="text-zinc-300 font-mono">
                {health.checking ? 'Pinging...' : health.online ? 'Backend Live' : 'Offline'}
              </span>
              <button
                onClick={checkStatus}
                title="Refresh Status"
                className="text-zinc-500 hover:text-zinc-200 ml-1 p-0.5 hover:bg-zinc-800 rounded cursor-pointer"
              >
                <RefreshCw className={`w-3 h-3 ${health.checking ? 'animate-spin' : ''}`} />
              </button>
            </div>

            <button
              onClick={() => setShowConfig(!showConfig)}
              className="p-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 transition-colors cursor-pointer hidden sm:block"
              title="API Endpoint Configuration"
            >
              <Settings className="w-4 h-4" />
            </button>

            {/* Mobile Hamburger */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 transition-colors cursor-pointer"
              aria-label="Toggle mobile menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>

        </div>
      </div>

      {/* Mobile Navigation Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-zinc-800 bg-zinc-950/95 backdrop-blur-lg animate-fade-in-up">
          <div className="px-4 py-3 space-y-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => handleTabClick(tab.id)}
                className={`w-full text-left px-4 py-3 rounded-xl text-sm font-semibold transition-all cursor-pointer ${
                  activeTab === tab.id
                    ? 'bg-indigo-500/15 text-indigo-400 border border-indigo-500/30'
                    : 'text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Mobile status bar */}
          <div className="px-4 py-3 border-t border-zinc-800 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs">
              <span className="relative flex h-2 w-2">
                {health.online ? (
                  <>
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-500 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                  </>
                ) : (
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500"></span>
                )}
              </span>
              <span className="text-zinc-300 font-mono">
                {health.online ? 'Backend Live' : 'Offline / Standalone'}
              </span>
            </div>
            <button
              onClick={() => { setShowConfig(!showConfig); setMobileMenuOpen(false); }}
              className="text-xs text-zinc-400 hover:text-zinc-200 flex items-center gap-1 cursor-pointer"
            >
              <Settings className="w-3.5 h-3.5" /> Configure
            </button>
          </div>
        </div>
      )}

      {/* Endpoint Configuration Dropdown */}
      {showConfig && (
        <div className="border-t border-zinc-800 bg-zinc-900/95 px-4 py-3">
          <form onSubmit={handleSaveEndpoint} className="max-w-7xl mx-auto flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
            <span className="text-xs font-semibold text-zinc-500 whitespace-nowrap">Inference Server URL:</span>
            <input
              type="text"
              value={endpointInput}
              onChange={(e) => setEndpointInput(e.target.value)}
              className="flex-1 bg-black border border-zinc-800 rounded-lg px-3 py-1.5 text-xs text-zinc-200 font-mono focus:outline-none focus:border-indigo-500"
              placeholder="http://localhost:5000"
            />
            <div className="flex gap-2">
              <button
                type="submit"
                className="px-3 py-1.5 bg-indigo-500 hover:bg-purple-500 text-white text-xs font-semibold rounded-lg transition-colors cursor-pointer"
              >
                Save & Connect
              </button>
              <button
                type="button"
                onClick={() => setShowConfig(false)}
                className="px-3 py-1.5 bg-zinc-800 text-zinc-300 hover:text-white text-xs font-semibold rounded-lg transition-colors cursor-pointer"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}
    </header>
  );
}
