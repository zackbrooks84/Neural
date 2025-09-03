/* ============================================================
 * Ember / Neural – index/main.js
 * Modes: Auto (API → Browser), API only, Browser (WebLLM) only
 * ============================================================ */

(() => {
  // -------------------------------
  // Constants & Local Storage Keys
  // -------------------------------
  const LS_HISTORY = 'chat-history';
  const LS_API = 'api-base';
  const LS_MODE = 'chat-mode';              // 'auto' | 'api' | 'webllm'
  const MAX_HISTORY = 200;
  const REQ_TIMEOUT_MS = 30_000;
  const RETRIES = 2;
  const RETRY_BACKOFF_MS = 800;

  // WebLLM (browser) model id – served via:
  // <script src="https://unpkg.com/@mlc-ai/web-llm@0.2.70/dist/index.min.js"></script>
  const WEBLLM_MODEL = 'Llama-3.1-8B-Instruct-q4f16_1-MLC';

  // -------------------------------
  // DOM
  // -------------------------------
  const $ = (id) => document.getElementById(id);
  const CHAT_BOX = $('chat-box');
  const MESSAGE = $('message');
  const SEND = $('send');
  const API_INPUT = $('api-base');
  const SAVE_API_BTN = $('save-api');
  const USE_WEB = $('use-web');
  const WEB_QUERY = $('web-query');
  const MODE_SELECT = $('mode-select'); // optional (Auto/API/Browser dropdown)

  // Optional status chip in the hero
  const API_DOT = $('api-dot');
  const API_STATUS = $('api-status');

  // -------------------------------
  // State
  // -------------------------------
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  let history = [];
  try { history = JSON.parse(localStorage.getItem(LS_HISTORY) || '[]'); }
  catch { history = []; }

  let apiBase = localStorage.getItem(LS_API);
  if (!apiBase) {
    apiBase = (['localhost', '127.0.0.1'].includes(location.hostname))
      ? location.origin
      : 'http://localhost:8000';
    localStorage.setItem(LS_API, apiBase);
  }
  if (API_INPUT) API_INPUT.value = apiBase;

  let modePref = localStorage.getItem(LS_MODE) || 'auto';   // 'auto' | 'api' | 'webllm'
  if (MODE_SELECT) MODE_SELECT.value = modePref;

  let webllmEngine = null; // set when initialized

  // -------------------------------
  // Helpers
  // -------------------------------
  function normalizeApiBase(v) {
    if (!v) return null;
    let s = String(v).trim();
    if (!/^https?:\/\//i.test(s)) s = `http://${s}`;
    try {
      const u = new URL(s);
      u.pathname = u.pathname.replace(/\/+$/, '');
      return u.origin;
    } catch { return null; }
  }

  function setStatusChip(kind) {
    // kind: 'ok' | 'offline' | 'unresponsive' | 'browser'
    if (!API_DOT || !API_STATUS) return;
    if (kind === 'browser') {
      API_DOT.style.background = '#22c55e';
      API_DOT.style.boxShadow = '0 0 10px rgba(34,197,94,.6)';
      API_STATUS.textContent = 'Browser LLM';
    } else if (kind === 'ok') {
      API_DOT.style.background = '#22c55e';
      API_DOT.style.boxShadow = '0 0 10px rgba(34,197,94,.6)';
      API_STATUS.textContent = 'Available';
    } else if (kind === 'offline') {
      API_DOT.style.background = '#ef4444';
      API_DOT.style.boxShadow = '0 0 10px rgba(239,68,68,.6)';
      API_STATUS.textContent = 'Offline';
    } else {
      API_DOT.style.background = '#f59e0b';
      API_DOT.style.boxShadow = '0 0 10px rgba(245,158,11,.6)';
      API_STATUS.textContent = 'Unresponsive';
    }
  }

  async function probeHealth(base) {
    // Skip probing if user forces browser-only mode
    if (modePref === 'webllm') { setStatusChip('browser'); return false; }
    try {
      const r = await fetch(`${base}/health`, { method: 'GET' });
      if (r.ok) { setStatusChip('ok'); return true; }
      setStatusChip('unresponsive'); return false;
    } catch {
      setStatusChip('offline'); return false;
    }
  }

  // Linkify bare URLs
  function linkify(text) {
    const esc = (s) =>
      s.replace(/&/g, '&amp;')
       .replace(/</g, '&lt;')
       .replace(/>/g, '&gt;');
    const urlRx = /\b(https?:\/\/[^\s<>()]+[^\s<>().,;!?"'])/g;
    return esc(String(text ?? '')).replace(urlRx, '<a href="$1" target="_blank" rel="noreferrer">$1</a>');
  }

  // Chat history
  function appendMessage(role, text, id) {
    const div = document.createElement('div');
    div.className = role;
    if (id) div.id = id;
    div.innerHTML = linkify(text);
    CHAT_BOX.appendChild(div);
    CHAT_BOX.scrollTop = CHAT_BOX.scrollHeight;
  }
  function renderHistory() {
    CHAT_BOX.innerHTML = '';
    for (const item of history) {
      appendMessage('user', item.user);
      appendMessage('assistant', item.assistant);
    }
    CHAT_BOX.scrollTop = CHAT_BOX.scrollHeight;
  }
  function pushHistory(user, assistant) {
    history.push({ user, assistant });
    if (history.length > MAX_HISTORY) history.splice(0, history.length - MAX_HISTORY);
    localStorage.setItem(LS_HISTORY, JSON.stringify(history));
  }
  renderHistory();

  // -------------------------------
  // WebLLM (browser) path
  // -------------------------------
  async function ensureWebLLM() {
    if (webllmEngine) return webllmEngine;
    if (!window.webllm) throw new Error('WebLLM script not loaded.');
    appendMessage('assistant pending', 'Loading browser model… (first time only)');
    webllmEngine = await webllm.CreateMLCEngine(WEBLLM_MODEL, {
      initProgressCallback: (p) => {
        const el = document.querySelector('.assistant.pending:last-child');
        if (el && p && typeof p.progress === 'number') {
          el.innerHTML = `Loading browser model… ${Math.round(p.progress * 100)}%`;
        }
      }
    });
    const pending = document.querySelector('.assistant.pending:last-child');
    if (pending) pending.outerHTML = `<div class="assistant">Browser model ready.</div>`;
    setStatusChip('browser');
    return webllmEngine;
  }

  async function webllmChat(message) {
    const engine = await ensureWebLLM();
    const msgs = [
      { role: 'system', content: 'You are Ember, concise, kind, and truthful.' },
      ...history.flatMap(h => ([
        { role: 'user', content: h.user },
        { role: 'assistant', content: h.assistant },
      ])),
      { role: 'user', content: message }
    ];
    const out = await engine.chat.completions.create({
      messages: msgs,
      temperature: 0.7,
      top_p: 0.95,
      stream: false,
    });
    return out?.choices?.[0]?.message?.content ?? '(no output)';
  }

  // -------------------------------
  // API path
  // -------------------------------
  async function fetchWithTimeout(url, options = {}, timeout = REQ_TIMEOUT_MS) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    try {
      return await fetch(url, { ...options, signal: controller.signal });
    } finally {
      clearTimeout(id);
    }
  }

  async function postChat(payload) {
    const url = `${apiBase}/chat`;
    let attempt = 0;
    while (true) {
      try {
        const r = await fetchWithTimeout(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!r.ok) {
          const txt = await r.text().catch(() => '');
          throw new Error(`HTTP ${r.status}${txt ? `: ${txt}` : ''}`);
        }
        return r.json();
      } catch (err) {
        if (attempt >= RETRIES) throw err;
        await sleep(RETRY_BACKOFF_MS * Math.pow(2, attempt));
        attempt += 1;
      }
    }
  }

  // -------------------------------
  // Send message
  // -------------------------------
  async function sendMessage() {
    const msg = (MESSAGE.value || '').trim();
    if (!msg) return;

    appendMessage('user', msg);
    MESSAGE.value = '';
    const placeholderId = `assist-${Date.now()}`;
    appendMessage('assistant pending', '…', placeholderId);

    try {
      let reply;

      if (modePref === 'webllm') {
        // Browser-only
        reply = await webllmChat(msg);

      } else if (modePref === 'api') {
        // API-only (do not fall back)
        const payload = { message: msg };
        if (USE_WEB?.checked) {
          payload.use_web = true;
          const wq = (WEB_QUERY?.value || '').trim();
          if (wq) payload.web_query = wq;
        }
        const data = await postChat(payload);
        reply = data?.reply ?? '(No reply)';

      } else {
        // Auto: try API; if not healthy, fall back to browser
        const apiOk = await probeHealth(apiBase);
        if (apiOk) {
          const payload = { message: msg };
          if (USE_WEB?.checked) {
            payload.use_web = true;
            const wq = (WEB_QUERY?.value || '').trim();
            if (wq) payload.web_query = wq;
          }
          const data = await postChat(payload);
          reply = data?.reply ?? '(No reply)';
        } else {
          reply = await webllmChat(msg);
        }
      }

      const el = document.getElementById(placeholderId);
      if (el) el.outerHTML = `<div class="assistant">${linkify(reply)}</div>`;
      pushHistory(msg, reply);
    } catch (err) {
      const el = document.getElementById(placeholderId);
      const txt = `Error: ${err?.message || err}`;
      if (el) el.outerHTML = `<div class="assistant">${linkify(txt)}</div>`;
      console.error(err);
    } finally {
      if (WEB_QUERY) WEB_QUERY.value = '';
    }
  }

  // -------------------------------
  // Events
  // -------------------------------
  if (SEND) SEND.addEventListener('click', sendMessage);
  if (MESSAGE) {
    MESSAGE.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }
  if (SAVE_API_BTN && API_INPUT) {
    SAVE_API_BTN.addEventListener('click', async () => {
      const normalized = normalizeApiBase(API_INPUT.value);
      if (!normalized) { alert('Invalid API base. Example: http://localhost:8000'); return; }
      apiBase = normalized;
      localStorage.setItem(LS_API, apiBase);
      await probeHealth(apiBase);
    });
  }
  if (MODE_SELECT) {
    MODE_SELECT.addEventListener('change', async () => {
      modePref = MODE_SELECT.value;
      localStorage.setItem(LS_MODE, modePref);
      if (modePref === 'webllm') {
        setStatusChip('browser');
        try { await ensureWebLLM(); } catch (e) { console.error(e); }
      } else {
        await probeHealth(apiBase);
      }
    });
  }

  // Initial status probe
  (async () => {
    if (modePref === 'webllm') {
      setStatusChip('browser');
    } else {
      await probeHealth(apiBase);
    }
  })();
})();