// popup.js — TamilGuard Extension Popup Logic

const $ = id => document.getElementById(id);

// ── Load stored settings ───────────────────────────────────────────────────────
chrome.storage.local.get(
  ['enabled', 'apiBase', 'blurStrength', 'action', 'sensitivity', 'totalBlurred'],
  (data) => {
    $('enabled-toggle').checked = data.enabled !== false;
    $('toggle-label').textContent = data.enabled !== false ? 'ON' : 'OFF';
    $('action-select').value = data.action || 'blur';
    $('sensitivity-select').value = data.sensitivity || 'standard';
    $('blur-slider').value = data.blurStrength || 8;
    $('api-input').value = data.apiBase || 'http://localhost:5000';
    $('blurred-count').textContent = data.totalBlurred || 0;
  }
);

// ── Check backend health ───────────────────────────────────────────────────────
async function checkHealth(apiBase) {
  const base = apiBase || $('api-input').value || 'http://localhost:5000';
  try {
    const res = await fetch(`${base}/health`, { signal: AbortSignal.timeout(3000) });
    const json = await res.json();
    const dot = $('status-dot');
    const txt = $('status-text');
    if (res.ok && json.status === 'ok') {
      dot.className = 'online';
      txt.textContent = 'API Online';
    } else {
      dot.className = 'offline';
      txt.textContent = 'API Error';
    }
  } catch {
    $('status-dot').className = 'offline';
    $('status-text').textContent = 'API Offline';
  }
}

chrome.storage.local.get(['apiBase'], d => checkHealth(d.apiBase));

// ── Toggle enabled/disabled ────────────────────────────────────────────────────
$('enabled-toggle').addEventListener('change', (e) => {
  const enabled = e.target.checked;
  $('toggle-label').textContent = enabled ? 'ON' : 'OFF';
  chrome.storage.local.set({ enabled });
  // Notify all active tabs
  chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
    if (tabs[0]) {
      chrome.tabs.sendMessage(tabs[0].id, { type: 'SET_ENABLED', enabled }).catch(() => {});
    }
  });
});

// ── Save button ────────────────────────────────────────────────────────────────
$('save-btn').addEventListener('click', () => {
  const settings = {
    action: $('action-select').value,
    sensitivity: $('sensitivity-select').value,
    blurStrength: parseInt($('blur-slider').value),
    apiBase: $('api-input').value.trim() || 'http://localhost:5000',
  };
  chrome.storage.local.set(settings, () => {
    const msg = $('saved-msg');
    msg.style.opacity = '1';
    setTimeout(() => { msg.style.opacity = '0'; }, 1500);
    checkHealth(settings.apiBase);
  });
});

// ── Live blur slider preview label ────────────────────────────────────────────
$('blur-slider').addEventListener('input', (e) => {
  // Could show live label here if desired
});
