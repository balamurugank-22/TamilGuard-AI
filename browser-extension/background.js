// background.js — TamilGuard Browser Extension Service Worker
// Handles API calls from content scripts (avoids CORS issues from content context)

const DEFAULT_API = 'http://localhost:5000';

// On install, set default settings
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({
    enabled: true,
    apiBase: DEFAULT_API,
    blurStrength: 8,     // px
    action: 'blur',      // 'blur' | 'hide' | 'replace'
    sensitivity: 'standard',
    totalBlurred: 0,
  });
  console.log('[TamilGuard] Extension installed. Defaults set.');
});

// Listen for messages from content scripts
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'ANALYZE_TEXT') {
    handleAnalyze(msg.text, msg.sensitivity, msg.apiBase)
      .then(result => sendResponse({ ok: true, result }))
      .catch(err => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (msg.type === 'ANALYZE_BATCH') {
    handleBatchAnalyze(msg.texts, msg.sensitivity, msg.apiBase)
      .then(result => sendResponse({ ok: true, result }))
      .catch(err => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (msg.type === 'INCREMENT_BLURRED') {
    chrome.storage.local.get(['totalBlurred'], (data) => {
      chrome.storage.local.set({ totalBlurred: (data.totalBlurred || 0) + msg.count });
    });
    return false;
  }
});

async function handleAnalyze(text, sensitivity, apiBase) {
  const base = apiBase || DEFAULT_API;
  const response = await fetch(`${base}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, sensitivity: sensitivity || 'standard', debug: false }),
  });
  if (!response.ok) throw new Error(`API error: ${response.status}`);
  return response.json();
}

async function handleBatchAnalyze(texts, sensitivity, apiBase) {
  const base = apiBase || DEFAULT_API;
  const response = await fetch(`${base}/predict_batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texts, sensitivity: sensitivity || 'standard' }),
  });
  if (!response.ok) throw new Error(`API error: ${response.status}`);
  return response.json();
}
