/* ============================================================================
 * Neural Chat — main.js
 * Powers the upgraded docs/index.html:
 * - Chat with Markdown + sanitize + code highlighting
 * - Local history + settings (API base persisted to localStorage)
 * - Web/search & URL fetch toggles + anchor tag filters
 * - Three.js reactive avatar with hue/reactivity controls
 * 
 * Requires globals loaded in index.html:
 *   THREE, marked, DOMPurify, hljs
 * ========================================================================== */

/* --------------------------
 * DOM helpers & constants
 * -------------------------- */
const $ = (id) => document.getElementById(id);

const CHAT = $("chat");
const MESSAGE = $("message");
const SEND = $("send");
const USE_WEB = $("use-web");
const WEB_QUERY = $("web-query");
const URLS = $("urls");
const ANCHOR_TAGS = $("anchor-tags");

const SETTINGS = $("settings");
const BTN_SETTINGS = $("btn-settings");
const BTN_SETTINGS_CLOSE = $("settings-close");
const BTN_SETTINGS_SAVE = $("settings-save");
const BTN_SETTINGS_RESET = $("settings-reset");
const BTN_CLEAR = $("btn-clear");
const API_BASE_TEXT = $("api-base");
const SETTING_API = $("setting-api");

const REACTIVITY = $("reactivity");
const HUE = $("hue");

const LS_API = "ember.apiBase";
const LS_HISTORY = "ember.chatHistory";

/* --------------------------
 * API base management
 * -------------------------- */
function getApiBase() {
  let base = localStorage.getItem(LS_API);
  if (!base) {
    if (["localhost", "127.0.0.1"].includes(window.location.hostname)) {
      base = "http://localhost:8000";
    } else {
      base = "http://localhost:8000";
    }
    localStorage.setItem(LS_API, base);
  }
  return base;
}
function setApiBase(v) {
  localStorage.setItem(LS_API, v);
  API_BASE_TEXT.textContent = v;
}
setApiBase(getApiBase());
SETTING_API.value = getApiBase();

/* --------------------------
 * Local history
 * -------------------------- */
function saveHistory() {
  localStorage.setItem(LS_HISTORY, CHAT.innerHTML);
}
function loadHistory() {
  const html = localStorage.getItem(LS_HISTORY);
  if (html) CHAT.innerHTML = html;
}
function clearChat() {
  CHAT.innerHTML = "";
  saveHistory();
}

/* --------------------------
 * Markdown rendering
 * -------------------------- */
function renderMarkdown(md) {
  marked.setOptions({
    breaks: true,
    highlight(code, lang) {
      try {
        return hljs.highlight(code, { language: lang }).value;
      } catch {
        return hljs.highlightAuto(code).value;
      }
    },
  });
  const dirty = marked.parse(md || "");
  return DOMPurify.sanitize(dirty);
}

/* --------------------------
 * Message helpers
 * -------------------------- */
function messageBubble(sender, html, role = "assistant") {
  const wrap = document.createElement("div");
  wrap.className = "msg";
  const isUser = role === "user";
  wrap.innerHTML = `
    <div class="flex ${isUser ? "justify-end" : "justify-start"}">
      <div class="max-w-[92%] md:max-w-[75%] rounded-xl px-3 py-2 border ${
        isUser
          ? "bg-indigo-600/70 border-indigo-400/40"
          : "bg-gray-900/70 border-white/10"
      }">
        <div class="text-xs opacity-70 mb-1">${sender}</div>
        <div class="chat-md prose prose-invert text-sm">${html}</div>
      </div>
    </div>
  `;
  CHAT.appendChild(wrap);
  CHAT.scrollTop = CHAT.scrollHeight;
  saveHistory();
  if (!isUser) pulseAvatar();
}
function appendUser(text) {
  messageBubble("You", renderMarkdown(text), "user");
}
function appendAssistant(text) {
  messageBubble("Neural", renderMarkdown(text), "assistant");
}
function setThinking() {
  const wrap = document.createElement("div");
  wrap.className = "msg";
  wrap.id = "thinking";
  wrap.innerHTML = `
    <div class="flex justify-start">
      <div class="max-w-[92%] md:max-w-[75%] rounded-xl px-3 py-2 border bg-gray-900/70 border-white/10">
        <div class="text-xs opacity-70 mb-1">Neural</div>
        <div class="text-sm text-purple-300 animate-pulse">Thinking…</div>
      </div>
    </div>
  `;
  CHAT.appendChild(wrap);
  CHAT.scrollTop = CHAT.scrollHeight;
  saveHistory();
}
function replaceThinkingWith(text) {
  const t = $("thinking");
  const html = renderMarkdown(text);
  if (!t) {
    appendAssistant(text);
    return;
  }
  t.outerHTML = `
    <div class="msg">
      <div class="flex justify-start">
        <div class="max-w-[92%] md:max-w-[75%] rounded-xl px-3 py-2 border bg-gray-900/70 border-white/10">
          <div class="text-xs opacity-70 mb-1">Neural</div>
          <div class="chat-md prose prose-invert text-sm">${html}</div>
        </div>
      </div>
    </div>
  `;
  CHAT.scrollTop = CHAT.scrollHeight;
  saveHistory();
  pulseAvatar();
}

/* --------------------------
 * Chat send
 * -------------------------- */
async function sendMessage() {
  const msg = (MESSAGE.value || "").trim();
  if (!msg) return;

  appendUser(msg);
  MESSAGE.value = "";
  setThinking();

  const payload = {
    message: msg,
    use_web: USE_WEB.checked,
    web_query: (WEB_QUERY.value || "").trim() || null,
    urls: (URLS.value || "")
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean),
    anchor_tags: (ANCHOR_TAGS.value || "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean),
  };
  if (!payload.urls.length) delete payload.urls;
  if (!payload.anchor_tags.length) delete payload.anchor_tags;

  try {
    const res = await fetch(`${getApiBase()}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`HTTP ${res.status}: ${errText}`);
    }
    const data = await res.json();
    replaceThinkingWith(data.reply ?? "(No reply)");
  } catch (err) {
    replaceThinkingWith(`**Connection error**: ${String(err)}`);
  }
}

/* --------------------------
 * Events
 * -------------------------- */
SEND.addEventListener("click", sendMessage);
MESSAGE.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Ctrl/Cmd+K focus; Shift+C clear
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    MESSAGE.focus();
  }
  if (e.shiftKey && e.key.toLowerCase() === "c") {
    e.preventDefault();
    clearChat();
  }
});

// Settings drawer
function openSettings() {
  SETTINGS.classList.remove("hidden");
  SETTINGS.classList.add("flex");
}
function closeSettings() {
  SETTINGS.classList.add("hidden");
  SETTINGS.classList.remove("flex");
}
BTN_SETTINGS.addEventListener("click", openSettings);
BTN_SETTINGS_CLOSE.addEventListener("click", closeSettings);
BTN_SETTINGS_SAVE.addEventListener("click", () => {
  const val = (SETTING_API.value || "").trim() || "http://localhost:8000";
  setApiBase(val);
  closeSettings();
});
BTN_SETTINGS_RESET.addEventListener("click", () => {
  setApiBase("http://localhost:8000");
  SETTING_API.value = getApiBase();
});

BTN_CLEAR.addEventListener("click", clearChat);

// Init footer API text + load history
API_BASE_TEXT.textContent = getApiBase();
loadHistory();

/* --------------------------
 * Three.js avatar
 * -------------------------- */
const avatarContainer = $("avatar-canvas");
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
avatarContainer.appendChild(renderer.domElement);

function resizeAvatar() {
  const w = avatarContainer.clientWidth;
  const h = avatarContainer.clientHeight;
  renderer.setSize(w, h, false);
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
resizeAvatar();
window.addEventListener("resize", resizeAvatar);

// Shader uniforms
const uniforms = {
  time: { value: 0.0 },
  hue: { value: (parseFloat(HUE.value) % 360) / 360 },
  amp: { value: parseFloat(REACTIVITY.value) / 100 },
};

const geo = new THREE.SphereGeometry(1, 96, 96);
const mat = new THREE.ShaderMaterial({
  uniforms,
  vertexShader: `
    uniform float time;
    uniform float amp;
    varying vec3 vPos;
    varying vec2 vUv;
    void main(){
      vUv = uv;
      vec3 p = position;
      float n = sin(p.x*3.2 + time*1.2)*0.02 + cos(p.y*4.1 + time*0.8)*0.02;
      p += normal * n * amp;
      vPos = p;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
    }
  `,
  fragmentShader: `
    uniform float time;
    uniform float hue;
    varying vec2 vUv;
    float sat = 0.75;
    float val = 0.9;
    vec3 hsv2rgb(vec3 c){
      vec3 p = abs(fract(c.xxx + vec3(0., 2./6., 4./6.)) * 6. - 3.);
      vec3 rgb = c.z * mix(vec3(1.), clamp(p - 1., 0., 1.), c.y);
      return rgb;
    }
    void main() {
      float r = 0.5 + 0.5 * sin(time*0.8 + vUv.x*8.);
      float mixv = mix(0.35, 1.0, r);
      vec3 col = hsv2rgb(vec3(hue, sat, val * mixv));
      gl_FragColor = vec4(col, 0.95);
    }
  `,
  transparent: true,
});
const sphere = new THREE.Mesh(geo, mat);
scene.add(sphere);
const light = new THREE.DirectionalLight(0xffffff, 0.8);
light.position.set(2, 2, 2);
scene.add(light);
camera.position.z = 2.6;

// Controls
REACTIVITY.addEventListener("input", (e) => {
  uniforms.amp.value = parseFloat(e.target.value) / 100;
});
HUE.addEventListener("input", (e) => {
  uniforms.hue.value = (parseFloat(e.target.value) % 360) / 360;
});

function tickAvatar() {
  requestAnimationFrame(tickAvatar);
  uniforms.time.value += 0.016;
  sphere.rotation.y += 0.0025;
  renderer.render(scene, camera);
}
tickAvatar();

// Small "pulse" animation when assistant replies
function pulseAvatar() {
  const base = uniforms.amp.value;
  let t = 0;
  const id = setInterval(() => {
    t += 1;
    uniforms.amp.value = base + Math.sin(t / 2) * 0.15;
    if (t > 18) {
      uniforms.amp.value = base;
      clearInterval(id);
    }
  }, 300);
}