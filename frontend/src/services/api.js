// API Service for communication with the Flask / backend inference server

const DEFAULT_API_BASE = 'http://localhost:5000';

export function getApiBase() {
  return localStorage.getItem('tamilguard_api_base') || DEFAULT_API_BASE;
}

export function setApiBase(url) {
  localStorage.setItem('tamilguard_api_base', url.replace(/\/+$/, ''));
}

export async function checkBackendHealth() {
  const base = getApiBase();
  try {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), 3000);
    const resp = await fetch(`${base}/health`, { signal: controller.signal });
    clearTimeout(id);
    if (!resp.ok) return { online: false, error: `HTTP ${resp.status}` };
    const data = await resp.json();
    return { online: true, data };
  } catch (err) {
    return { online: false, error: err.message };
  }
}

export async function predictText(text, full = true, sensitivity = 'standard') {
  const base = getApiBase();
  const url = `${base}/predict?full=${full}&sensitivity=${sensitivity}`;
  
  const startTime = performance.now();
  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text, sensitivity }),
    });

    if (!resp.ok) {
      throw new Error(`Server returned status ${resp.status}`);
    }

    const data = await resp.json();
    data._client_ms = Math.round(performance.now() - startTime);
    return data;
  } catch (err) {
    console.warn(`[TamilGuard] Backend request failed (${err.message}). Using fallback local emulator.`);
    return fallbackPredict(text, sensitivity);
  }
}

// Helper bounded Levenshtein distance for client-side fallback
function editDist(s1, s2) {
  if (Math.abs(s1.length - s2.length) > 2) return 3;
  const dp = Array.from({ length: s1.length + 1 }, (_, i) => [i]);
  for (let j = 1; j <= s2.length; j++) dp[0][j] = j;
  for (let i = 1; i <= s1.length; i++) {
    for (let j = 1; j <= s2.length; j++) {
      dp[i][j] = s1[i - 1] === s2[j - 1] 
        ? dp[i - 1][j - 1] 
        : Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1);
    }
  }
  return dp[s1.length][s2.length];
}

// Fallback local rule-based heuristic for demo continuity if server is offline
function fallbackPredict(text, sensitivity = 'standard') {
  let clean = text.trim();
  
  // De-stuff punctuation if strict or maximum
  if (sensitivity === 'strict' || sensitivity === 'maximum') {
    clean = clean.replace(/\b([\p{L}\d](?:[\*\._\-\+~^/][\p{L}\d]){2,})\b/gu, (m) => m.replace(/[\*\._\-\+~^/]/g, ''));
    // Leetspeak folding
    clean = clean.replace(/3/g, 'e').replace(/!/g, 'i').replace(/@/g, 'a').replace(/0/g, 'o').replace(/\$/g, 's').replace(/1/g, 'i');
  }

  const slurs = [
    { word: 'thevdiya', canon: 'thevidiya', cat: 'sexual', sev: 'high' },
    { word: 'thevidiya', canon: 'thevidiya', cat: 'sexual', sev: 'high' },
    { word: 'thevudiya', canon: 'thevidiya', cat: 'sexual', sev: 'high' },
    { word: 'thevdya', canon: 'thevidiya', cat: 'sexual', sev: 'high' },
    { word: 'oombi', canon: 'oombi', cat: 'sexual', sev: 'high' },
    { word: 'umbi', canon: 'oombi', cat: 'sexual', sev: 'high' },
    { word: 'kirukkan', canon: 'kiruku', cat: 'profanity', sev: 'medium' },
    { word: 'pundai', canon: 'pundai', cat: 'sexual', sev: 'high' },
    { word: 'pundaikku', canon: 'pundai', cat: 'sexual', sev: 'high' },
    { word: 'தேவிடியா', canon: 'thevidiya', cat: 'sexual', sev: 'high' },
    { word: 'உயிரோட விடமாட்டேன்', canon: 'death_threat', cat: 'threat', sev: 'high' }
  ];

  const words = clean.split(/\s+/).filter(Boolean);
  const flaggedSpans = [];
  const flaggedWords = [];
  const categories = new Set();
  const tokens = words;
  const modelTags = [];
  const lexiconTags = [];
  const mergedTags = [];

  for (const w of words) {
    const lw = w.toLowerCase().replace(/[^\p{L}\p{Nd}]/gu, '');
    let match = slurs.find(s => s.word === lw || lw.includes(s.word));

    // Fuzzy matching if strict or maximum
    if (!match && (sensitivity === 'strict' || sensitivity === 'maximum')) {
      const maxDist = sensitivity === 'maximum' ? 2 : 1;
      match = slurs.find(s => s.word.length >= 4 && editDist(lw, s.word) <= maxDist);
    }

    if (match) {
      flaggedSpans.push({
        token: w,
        source: 'lexicon',
        canon: match.canon,
        category: match.cat,
        severity: match.sev,
        match_type: 'exact',
      });
      flaggedWords.push(w);
      categories.add(match.cat);
      modelTags.push('O');
      lexiconTags.push('B-ABUSE');
      mergedTags.push('B-ABUSE');
    } else {
      modelTags.push('O');
      lexiconTags.push('O');
      mergedTags.push('O');
    }
  }

  return {
    safe: flaggedSpans.length === 0,
    flagged_words: [...new Set(flaggedWords)],
    categories: Array.from(categories),
    flagged_spans: flaggedSpans,
    text_normalized: clean,
    tokens,
    model_tags: modelTags,
    lexicon_tags: lexiconTags,
    merged_tags: mergedTags,
    _ms: 1.5,
    _offline_fallback: true,
  };
}

export async function censorText(text, options = {}) {
  const base = getApiBase();
  const url = `${base}/censor`;
  const startTime = performance.now();

  const { mode = 'partial', mask_char, severity_threshold = 'all', allowed_categories, sensitivity = 'standard' } = options;

  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        mode,
        mask_char,
        severity_threshold,
        allowed_categories,
        sensitivity,
      }),
    });

    if (!resp.ok) {
      throw new Error(`Server returned status ${resp.status}`);
    }

    const data = await resp.json();
    data._client_ms = Math.round(performance.now() - startTime);
    return data;
  } catch (err) {
    console.warn(`[TamilGuard] /censor backend request failed (${err.message}). Using fallback local emulator.`);
    return fallbackCensor(text, mode, sensitivity);
  }
}

// Fallback client-side censor emulator
function fallbackCensor(text, mode = 'partial', sensitivity = 'standard') {
  const pred = fallbackPredict(text, sensitivity);
  if (pred.safe) {
    return {
      original: text,
      censored: text,
      safe: true,
      mode,
      redacted_count: 0,
      redacted_spans: [],
      polite_suggestion: text,
      categories: [],
      _ms: 1.0,
      _offline_fallback: true,
    };
  }

  const politeDict = {
    'thevdiya': '[thavaraana varthai]',
    'thevudiya': '[thavaraana varthai]',
    'oombi': '[inappropriate]',
    'umbi': '[inappropriate]',
    'kirukkan': 'purindhukolladhavar',
    'pundai': '[inappropriate]',
    'pundaikku': '[inappropriate]',
    'தேவிடியா': '[தவறான சொல்]',
    'உயிரோட விடமாட்டேன்': 'எச்சரிக்கிறேன்',
  };

  const words = text.split(/\s+/);
  const censoredWords = [];
  const politeWords = [];
  const redactedSpans = [];

  for (const w of words) {
    const cleanWord = w.toLowerCase().replace(/[^\p{L}\p{Nd}]/gu, '');
    const matched = pred.flagged_spans.find(s => s.token.toLowerCase() === cleanWord || cleanWord.includes(s.token.toLowerCase()));
    
    if (matched) {
      let masked = w;
      if (mode === 'partial') {
        masked = w.length <= 4 ? w[0] + '*'.repeat(Math.max(1, w.length - 2)) + w[w.length - 1] : w.slice(0, 2) + '*'.repeat(Math.max(1, w.length - 4)) + w.slice(-2);
      } else if (mode === 'tag') {
        masked = `[REDACTED: ${(matched.category || 'ABUSE').toUpperCase()}]`;
      } else if (mode === 'block') {
        masked = '█'.repeat(w.length);
      } else if (mode === 'polite') {
        masked = politeDict[matched.token.toLowerCase()] || '[nanbar]';
      }
      censoredWords.push(masked);
      politeWords.push(politeDict[matched.token.toLowerCase()] || '[nanbar]');
      redactedSpans.push({
        original_token: w,
        redacted_token: masked,
        mode,
        category: matched.category,
        severity: matched.severity,
      });
    } else {
      censoredWords.push(w);
      politeWords.push(w);
    }
  }

  return {
    original: text,
    censored: censoredWords.join(' '),
    safe: false,
    mode,
    redacted_count: redactedSpans.length,
    redacted_spans: redactedSpans,
    polite_suggestion: politeWords.join(' '),
    categories: pred.categories,
    _ms: 1.2,
    _offline_fallback: true,
  };
}
