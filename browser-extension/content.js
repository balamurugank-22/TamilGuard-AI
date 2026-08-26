// content.js — TamilGuard Browser Extension Content Script (V2 - TreeWalker & Batch API)
// Scans visible text nodes directly, preventing layout breakage and false positives.

(function () {
  'use strict';

  // ─── Config ──────────────────────────────────────────────────────────────────
  let CONFIG = {
    enabled: true,
    apiBase: 'http://localhost:5000',
    blurStrength: 8,
    action: 'blur',
    sensitivity: 'standard',
  };

  chrome.storage.local.get(CONFIG, (stored) => {
    CONFIG = { ...CONFIG, ...stored };
    if (CONFIG.enabled) init();
  });

  chrome.storage.onChanged.addListener((changes) => {
    for (const [key, { newValue }] of Object.entries(changes)) {
      CONFIG[key] = newValue;
    }
    if (!CONFIG.enabled) removeAllBlurs();
  });

  // ─── State ────────────────────────────────────────────────────────────────────
  const processedNodes = new WeakSet();
  let pendingNodes = [];
  let isProcessing = false;
  let blurCount = 0;

  // ─── DOM Helpers ─────────────────────────────────────────────────────────────
  
  // Exclude hidden elements, script tags, style tags, inputs, etc.
  const EXCLUDED_TAGS = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEXTAREA', 'INPUT', 'SELECT', 'OPTION', 'CODE', 'PRE']);

  function shouldScanNode(node) {
    if (processedNodes.has(node)) return false;
    
    // Check parent visibility and tag
    const parent = node.parentElement;
    if (!parent) return false;
    if (EXCLUDED_TAGS.has(parent.tagName)) return false;
    
    // Don't scan inside our own blur wrappers
    if (parent.closest('.__tg_blur_wrapper')) {
      return false;
    }
    
    // Is the text meaningful?
    const text = node.nodeValue;
    if (!text || text.trim().length < 3) return false;
    if (!/[a-zA-Z\u0B80-\u0BFF]/.test(text)) return false; // Must contain letters
    
    return true;
  }

  // ─── TreeWalker Scanner ──────────────────────────────────────────────────────
  function scanDOM() {
    const walker = document.createTreeWalker(
      document.body,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode(node) {
          return shouldScanNode(node) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
        }
      }
    );

    let node;
    while ((node = walker.nextNode())) {
      pendingNodes.push(node);
      processedNodes.add(node);
    }
    
    if (pendingNodes.length > 0) {
      processBatch();
    }
  }

  // ─── Batch Processor ─────────────────────────────────────────────────────────
  async function processBatch() {
    if (isProcessing || pendingNodes.length === 0) return;
    isProcessing = true;

    // Grab a chunk of text nodes (up to 50 at once)
    const batch = pendingNodes.splice(0, 50);
    const texts = batch.map(n => n.nodeValue);

    try {
      const response = await new Promise((resolve, reject) => {
        chrome.runtime.sendMessage(
          { type: 'ANALYZE_BATCH', texts, sensitivity: CONFIG.sensitivity, apiBase: CONFIG.apiBase },
          (res) => {
            if (chrome.runtime.lastError) return reject(chrome.runtime.lastError);
            if (!res || !res.ok) return reject(new Error(res?.error || 'API Failed'));
            resolve(res.result);
          }
        );
      });

      if (response && response.results) {
        // Apply blur to abusive nodes
        response.results.forEach((res, index) => {
          if (!res.safe && res.flagged_words && res.flagged_words.length > 0) {
            applyBlurToTextNode(batch[index], res.flagged_words);
          }
        });
      }
    } catch (e) {
      // API error or timeout, silently ignore
    }

    isProcessing = false;
    
    // If more pending, continue processing
    if (pendingNodes.length > 0) {
      setTimeout(processBatch, 10);
    }
  }

  // ─── Precise Blur Logic (No layout breaking) ──────────────────────────────────
  function applyBlurToTextNode(textNode, flaggedWords) {
    const parent = textNode.parentElement;
    if (!parent) return;

    let originalText = textNode.nodeValue;
    
    // Create a DocumentFragment to replace the single text node
    const fragment = document.createDocumentFragment();
    
    if (CONFIG.action === 'hide') {
      // Just drop the flagged words entirely from this text node
      for (const word of flaggedWords) {
        const regex = new RegExp(word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
        originalText = originalText.replace(regex, '');
      }
      fragment.appendChild(document.createTextNode(originalText));
    } else if (CONFIG.action === 'replace') {
      // Replace with blocks
      let redacted = originalText;
      for (const word of flaggedWords) {
        const regex = new RegExp(word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
        redacted = redacted.replace(regex, '▓'.repeat(word.length));
      }
      fragment.appendChild(document.createTextNode(redacted));
    } else {
      // Action === 'blur'
      // We wrap the exact abusive words in a blurred span, leaving safe words alone.
      let currentIndex = 0;
      
      const matches = [];
      for (const word of flaggedWords) {
        const regex = new RegExp(word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
        let match;
        while ((match = regex.exec(originalText)) !== null) {
          matches.push({ start: match.index, end: match.index + match[0].length, text: match[0] });
        }
      }
      
      matches.sort((a, b) => a.start - b.start);
      const mergedMatches = [];
      for (const m of matches) {
        if (mergedMatches.length === 0) {
          mergedMatches.push(m);
        } else {
          const last = mergedMatches[mergedMatches.length - 1];
          if (m.start <= last.end) {
            last.end = Math.max(last.end, m.end);
            last.text = originalText.substring(last.start, last.end);
          } else {
            mergedMatches.push(m);
          }
        }
      }

      if (mergedMatches.length === 0) {
        fragment.appendChild(document.createTextNode(originalText));
      } else {
        mergedMatches.forEach(match => {
          if (match.start > currentIndex) {
            fragment.appendChild(document.createTextNode(originalText.substring(currentIndex, match.start)));
          }
          
          const wrapper = document.createElement('span');
          wrapper.className = '__tg_blur_wrapper';
          wrapper.dataset.tgBlurred = 'true';
          wrapper.dataset.tgOriginal = match.text;
          wrapper.style.cssText = 'position: relative; display: inline-block; cursor: pointer; border-bottom: 2px dotted #6366f1;';
          
          const blurLayer = document.createElement('span');
          blurLayer.className = '__tg_blur_content';
          blurLayer.textContent = match.text;
          blurLayer.style.cssText = `filter: blur(${CONFIG.blurStrength}px); transition: filter 0.3s;`;
          
          let revealed = false;
          wrapper.title = "TamilGuard: Abusive word. Click to reveal.";
          wrapper.addEventListener('click', (e) => {
            e.stopPropagation();
            e.preventDefault();
            revealed = !revealed;
            blurLayer.style.filter = revealed ? 'blur(0px)' : `blur(${CONFIG.blurStrength}px)`;
            wrapper.style.borderBottomColor = revealed ? 'transparent' : '#6366f1';
          });

          wrapper.appendChild(blurLayer);
          fragment.appendChild(wrapper);
          
          currentIndex = match.end;
        });

        if (currentIndex < originalText.length) {
          fragment.appendChild(document.createTextNode(originalText.substring(currentIndex)));
        }
      }
    }
    
    parent.replaceChild(fragment, textNode);
    blurCount++;
    chrome.runtime.sendMessage({ type: 'INCREMENT_BLURRED', count: 1 });
  }

  function removeAllBlurs() {
    document.querySelectorAll('.__tg_blur_wrapper').forEach(el => {
      const txt = document.createTextNode(el.dataset.tgOriginal);
      el.parentNode.replaceChild(txt, el);
    });
  }

  // ─── Observers & Scheduling ──────────────────────────────────────────────────
  let scanTimeout = null;
  function scheduleScan() {
    if (!CONFIG.enabled) return;
    if (scanTimeout) clearTimeout(scanTimeout);
    scanTimeout = setTimeout(scanDOM, 400); // Fast 400ms debounce
  }

  const observer = new MutationObserver(() => {
    scheduleScan();
  });

  // ─── Init ─────────────────────────────────────────────────────────────────────
  function init() {
    scanDOM();
    observer.observe(document.body, { childList: true, subtree: true, characterData: true });
    window.addEventListener('scroll', scheduleScan, { passive: true });
    console.log('[TamilGuard] TreeWalker scanner active.');
  }

})();
