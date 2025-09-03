/* ============================================================
 * Ember / Neural – index/main.js (enhanced)
 * Keeps existing IDs and structure for compatibility.
 * ============================================================ */

(() => {
  // -------------------------------
  // Constants & Local Storage Keys
  // -------------------------------
  const LS_HISTORY = 'chat-history';
  const LS_API = 'api-base';
  const MAX_HISTORY = 200;      // cap stored turns (pairs of user/assistant)
  const REQ_TIMEOUT_MS = 30_000; // network timeout for chat requests
  const RETRIES = 2;            // retries on failure
  const RETRY_BACKOFF_MS = 800; // exponential backoff base

  // -------------------------------
  // Elements (lazy getters)
  // -------------------------------
  const $ = (id) => document.getElementById(id);
  const CHAT_BOX = $('chat-box');
  const MESSAGE = $('message');
  const SEND = $('send');
  const API_INPUT = $('api-base');
  const SAVE_API_BTN = $('save-api');
  const USE_WEB = $('use-web');
  const WEB_QUERY = $('web-query');

  // API status elements in hero (if present on page)
  const API_DOT = $('api-dot');
  const API_STATUS = $('api-status');

  // -------------------------------
  // Utility helpers
  // -------------------------------
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // Basic URL normalizer / validator
  function normalizeApiBase(v) {
    if (!v) return null;
    let s = String(v).trim();
    if (!/^https?:\/\//i.test(s)) s = `http://${s}`;
    try {
      const u = new URL(s);
      // drop trailing slashes
      u.pathname = u.pathname.replace(/\/+$/, '');
      return u.origin;
    } catch {
      return null;
    }
  }

  function getDefaultApiBase() {
    if (['localhost', '127.0.0.1'].includes(window.location.hostname)) {
      return window.location.origin;
    }
    return 'http://localhost:8000';
  }

  function setApiStatus(ok) {
    if (!API_DOT || !API_STATUS) return;
    if (ok === true) {
      API_DOT.style.background = '#22c55e'; // green
      API_DOT.style.boxShadow = '0 0 10px rgba(34,197,94,.6)';
      API_STATUS.textContent = 'Available';
    } else if (ok === false) {
      API_DOT.style.background = '#ef4444'; // red
      API_DOT.style.boxShadow = '0 0 10px rgba(239,68,68,.6)';
      API_STATUS.textContent = 'Offline';
    } else {
      API_DOT.style.background = '#f59e0b'; // amber
      API_DOT.style.boxShadow = '0 0 10px rgba(245,158,11,.6)';
      API_STATUS.textContent = 'Unresponsive';
    }
  }

  async function probeHealth(base) {
    try {
      const r = await fetch(`${base}/health`, { method: 'GET' });
      setApiStatus(r.ok ? true : null);
    } catch {
      setApiStatus(false);
    }
  }

  // Make bare URLs clickable, leave everything else as text
  function linkify(text) {
    const esc = (s) =>
      s.replace(/&/g, '&amp;')
       .replace(/</g, '&lt;')
       .replace(/>/g, '&gt;');

    const urlRx = /\b(https?:\/\/[^\s<>()]+[^\s<>().,;!?"'])/g;
    return esc(text).replace(urlRx, '<a href="$1" target="_blank" rel="noreferrer">$1</a>');
  }

  // Sticky scroll: only auto-scroll if the user is near the bottom
  function atBottom(box, threshold = 24) {
    return box.scrollHeight - box.scrollTop - box.clientHeight < threshold;
  }
  function scrollToBottom(box) {
    box.scrollTop = box.scrollHeight;
  }

  // -------------------------------
  // State (history + API base)
  // -------------------------------
  let history = [];
  try { history = JSON.parse(localStorage.getItem(LS_HISTORY) || '[]'); }
  catch { history = []; }

  let apiBase = localStorage.getItem(LS_API);
  if (!apiBase) {
    apiBase = getDefaultApiBase();
    localStorage.setItem(LS_API, apiBase);
  }

  // Initialize API input & probe
  if (API_INPUT) API_INPUT.value = apiBase;
  probeHealth(apiBase);

  if (SAVE_API_BTN && API_INPUT) {
    SAVE_API_BTN.addEventListener('click', () => {
      const normalized = normalizeApiBase(API_INPUT.value);
      if (!normalized) {
        alert('Invalid API base URL. Example: http://localhost:8000');
        return;
      }
      apiBase = normalized;
      localStorage.setItem(LS_API, apiBase);
      probeHealth(apiBase);
    });
  }

  // -------------------------------
  // Rendering
  // -------------------------------
  function appendMessage(role, text, id) {
    const nearBottom = atBottom(CHAT_BOX);
    const div = document.createElement('div');
    div.className = role;
    if (id) div.id = id;
    div.innerHTML = linkify(String(text ?? ''));
    CHAT_BOX.appendChild(div);
    if (nearBottom) scrollToBottom(CHAT_BOX);
  }

  function renderHistory() {
    CHAT_BOX.innerHTML = '';
    for (const item of history) {
      appendMessage('user', item.user);
      appendMessage('assistant', item.assistant);
    }
    scrollToBottom(CHAT_BOX);
  }

  function pushHistory(user, assistant) {
    history.push({ user, assistant });
    if (history.length > MAX_HISTORY) {
      history.splice(0, history.length - MAX_HISTORY);
    }
    localStorage.setItem(LS_HISTORY, JSON.stringify(history));
  }

  function clearHistory() {
    history = [];
    localStorage.setItem(LS_HISTORY, JSON.stringify(history));
    renderHistory();
  }

  // Optional: expose clear in console for quick debugging
  window.__clearChat = clearHistory;

  // Initial render
  renderHistory();

  // -------------------------------
  // Chat logic
  // -------------------------------
  let inFlightController = null;

  async function fetchWithTimeout(url, options = {}, timeout = REQ_TIMEOUT_MS) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    try {
      const r = await fetch(url, { ...options, signal: controller.signal });
      return r;
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
          signal: inFlightController?.signal,
        });
        if (!r.ok) {
          const txt = await r.text().catch(() => '');
          throw new Error(`HTTP ${r.status}${txt ? `: ${txt}` : ''}`);
        }
        return r.json();
      } catch (err) {
        // Abort -> bubble up immediately
        if (err?.name === 'AbortError') throw err;
        if (attempt >= RETRIES) throw err;
        await sleep(RETRY_BACKOFF_MS * Math.pow(2, attempt));
        attempt += 1;
      }
    }
  }

  async function sendMessage() {
    const msg = (MESSAGE.value || '').trim();
    if (!msg) return;

    // Build payload (respect existing optional web fields)
    const payload = { message: msg };
    if (USE_WEB?.checked) {
      payload.use_web = true;
      const wq = (WEB_QUERY?.value || '').trim();
      if (wq) payload.web_query = wq;
    }

    // Show user message & placeholder
    appendMessage('user', msg);
    MESSAGE.value = '';
    const placeholderId = `assist-${Date.now()}`;
    appendMessage('assistant pending', '…', placeholderId);

    // Cancel any in-flight request if user sends again quickly
    if (inFlightController) {
      try { inFlightController.abort(); } catch {}
    }
    inFlightController = new AbortController();

    try {
      const data = await postChat(payload);
      const reply = data?.reply ?? '(No reply)';
      const el = document.getElementById(placeholderId);
      if (el) el.outerHTML = `<div class="assistant">${linkify(reply)}</div>`;
      pushHistory(msg, reply);
    } catch (err) {
      const el = document.getElementById(placeholderId);
      const msg = (err?.name === 'AbortError')
        ? 'Request cancelled.'
        : `Error: ${err?.message || err}`;
      if (el) el.outerHTML = `<div class="assistant">${linkify(msg)}</div>`;
      console.error('Chat request failed', err);
    } finally {
      // Reset web query field (UX nicety)
      if (WEB_QUERY) WEB_QUERY.value = '';
      // Re-probe API health to update status chip
      probeHealth(apiBase);
    }
  }

  // Events
  if (SEND) SEND.addEventListener('click', sendMessage);
  if (MESSAGE) {
    MESSAGE.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }

  // -------------------------------
  // 3D Visualization (Three.js)
  // -------------------------------
  const sceneContainer = $('scene-container');

  if (sceneContainer) {
    let scene, camera, renderer, cube;
    const colorInput = $('cube-color');
    const speedInput = $('rotation-speed');
    const toggleRotationBtn = $('toggle-rotation');
    const toggleWireframeBtn = $('toggle-wireframe');
    const resetViewBtn = $('reset-view');

    let rotationSpeed = parseFloat(speedInput?.value || '0.01');
    let isRotating = true;

    function initScene() {
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0c0c17);

      const { clientWidth: width, clientHeight: height } = sceneContainer;
      // Ensure non-zero size (in case CSS failed to set height)
      if (!height || height < 40) {
        sceneContainer.style.height = '420px';
      }

      const w = sceneContainer.clientWidth;
      const h = sceneContainer.clientHeight;

      camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
      camera.position.set(2, 2, 5);

      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
      renderer.setSize(w, h, false);
      sceneContainer.appendChild(renderer.domElement);

      const geometry = new THREE.BoxGeometry();
      const material = new THREE.MeshStandardMaterial({
        color: colorInput?.value || '#ff0000',
        roughness: 0.35,
        metalness: 0.15,
      });
      cube = new THREE.Mesh(geometry, material);
      scene.add(cube);

      createStarField();

      const ambient = new THREE.AmbientLight(0xffffff, 0.65);
      scene.add(ambient);

      const directional = new THREE.DirectionalLight(0xffffff, 0.9);
      directional.position.set(5, 5, 5);
      scene.add(directional);

      animate();
    }

    function createStarField() {
      const starGeometry = new THREE.BufferGeometry();
      const starCount = 700;
      const positions = new Float32Array(starCount * 3);

      for (let i = 0; i < starCount; i++) {
        positions[i * 3 + 0] = THREE.MathUtils.randFloatSpread(200);
        positions[i * 3 + 1] = THREE.MathUtils.randFloatSpread(200);
        positions[i * 3 + 2] = THREE.MathUtils.randFloatSpread(200);
      }

      starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      const starMaterial = new THREE.PointsMaterial({ color: 0xffffff, size: 0.7 });
      const stars = new THREE.Points(starGeometry, starMaterial);
      scene.add(stars);
    }

    function animate() {
      requestAnimationFrame(animate);
      if (isRotating && cube) {
        cube.rotation.x += rotationSpeed;
        cube.rotation.y += rotationSpeed;
      }
      renderer.render(scene, camera);
    }

    // Responsive resize (ResizeObserver is more accurate than window resize)
    const ro = new ResizeObserver(() => {
      const w = sceneContainer.clientWidth;
      const h = sceneContainer.clientHeight;
      if (w && h) {
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      }
    });
    ro.observe(sceneContainer);

    // Controls
    colorInput?.addEventListener('input', () => {
      if (cube?.material) cube.material.color.set(colorInput.value);
    });

    speedInput?.addEventListener('input', () => {
      rotationSpeed = parseFloat(speedInput.value || '0') || 0;
    });

    toggleRotationBtn?.addEventListener('click', () => {
      isRotating = !isRotating;
      toggleRotationBtn.textContent = isRotating ? 'Pause Rotation' : 'Resume Rotation';
      toggleRotationBtn.setAttribute('aria-pressed', String(isRotating));
    });

    toggleWireframeBtn?.addEventListener('click', () => {
      if (cube?.material) cube.material.wireframe = !cube.material.wireframe;
    });

    resetViewBtn?.addEventListener('click', () => {
      camera.position.set(2, 2, 5);
      camera.lookAt(0, 0, 0);
    });

    // Init scene
    initScene();
  }
})();