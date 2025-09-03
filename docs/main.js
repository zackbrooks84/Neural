const HISTORY_KEY = 'chat-history';
const API_KEY = 'api-base';
let history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
let apiBase = localStorage.getItem(API_KEY);
if (!apiBase) {
  if (["localhost", "127.0.0.1"].includes(window.location.hostname)) {
    apiBase = window.location.origin;
  } else {
    apiBase = "http://localhost:8000";
  }
  localStorage.setItem(API_KEY, apiBase);
}

function renderHistory() {
  const box = document.getElementById('chat-box');
  box.innerHTML = '';
  history.forEach(item => {
    appendMessage('user', item.user);
    appendMessage('assistant', item.assistant);
  });
  box.scrollTop = box.scrollHeight;
}

const apiInput = document.getElementById('api-base');
const saveApiBtn = document.getElementById('save-api');
if (apiInput && saveApiBtn) {
  apiInput.value = apiBase;
  saveApiBtn.addEventListener('click', () => {
    apiBase = apiInput.value.trim();
    localStorage.setItem(API_KEY, apiBase);
  });
}

function appendMessage(role, text, id) {
  const box = document.getElementById('chat-box');
  const div = document.createElement('div');
  div.className = role;
  if (id) div.id = id;
  div.textContent = text;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

async function sendMessage() {
  const input = document.getElementById('message');
  const msg = input.value.trim();
  if (!msg) return;

  // Prepare auxiliary elements for optional web search
  const useWeb = document.getElementById('use-web').checked;
  const webQueryInput = document.getElementById('web-query');
  const webQuery = webQueryInput.value.trim();

  // Build request payload
  const payload = { message: msg };
  if (useWeb) {
    payload.use_web = true;
    if (webQuery) payload.web_query = webQuery;
  }

  // Display user's message immediately
  appendMessage('user', msg);

  // Placeholder element for assistant response
  const placeholderId = `assist-${Date.now()}`;
  appendMessage('assistant pending', '...', placeholderId);

  try {
    const res = await fetch(`${apiBase}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      throw new Error(`Request failed with status ${res.status}`);
    }
    const data = await res.json();
    history.push({ user: msg, assistant: data.reply });
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    document.getElementById(placeholderId).textContent = data.reply;
  } catch (err) {
    // Replace placeholder with error message and log for diagnostics
    const errMsg = `Error: ${err.message || err}`;
    document.getElementById(placeholderId).textContent = errMsg;
    console.error('Chat request failed', err);
  } finally {
    renderHistory();
    input.value = '';
    webQueryInput.value = '';
  }
}

document.getElementById('send').addEventListener('click', sendMessage);

document.getElementById('message').addEventListener('keypress', (e) => {
  if (e.key === 'Enter') {
    sendMessage();
  }
});

renderHistory();

/* ------------------------------------------------------------------
 * 3D Visualization using Three.js
 * ------------------------------------------------------------------
 */

// Grab the container element that will hold our WebGL canvas.
const sceneContainer = document.getElementById('scene-container');

// Only attempt to set up the scene if the container exists on the page.
if (sceneContainer) {
  let scene, camera, renderer, cube;
  let rotationSpeed = parseFloat(document.getElementById('rotation-speed').value);
  let isRotating = true;

  // Initialize Three.js scene, camera, renderer, and objects.
  function initScene() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);

    const width = sceneContainer.clientWidth;
    const height = sceneContainer.clientHeight;

    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(2, 2, 5);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    sceneContainer.appendChild(renderer.domElement);

    // Primary cube geometry that users can manipulate.
    const geometry = new THREE.BoxGeometry();
    const material = new THREE.MeshStandardMaterial({ color: document.getElementById('cube-color').value });
    cube = new THREE.Mesh(geometry, material);
    scene.add(cube);

    // Add a star field for visual interest.
    createStarField();

    // Lighting: a mix of ambient and directional light for realism.
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambient);

    const directional = new THREE.DirectionalLight(0xffffff, 0.8);
    directional.position.set(5, 5, 5);
    scene.add(directional);

    // Kick off the animation loop.
    animate();
  }

  // Create a field of randomly placed stars.
  function createStarField() {
    const starGeometry = new THREE.BufferGeometry();
    const starCount = 500;
    const positions = [];

    for (let i = 0; i < starCount; i++) {
      const x = THREE.MathUtils.randFloatSpread(200);
      const y = THREE.MathUtils.randFloatSpread(200);
      const z = THREE.MathUtils.randFloatSpread(200);
      positions.push(x, y, z);
    }

    starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const starMaterial = new THREE.PointsMaterial({ color: 0xffffff });
    const stars = new THREE.Points(starGeometry, starMaterial);
    scene.add(stars);
  }

  // Animation loop: rotate the cube and render the scene.
  function animate() {
    requestAnimationFrame(animate);

    if (isRotating) {
      cube.rotation.x += rotationSpeed;
      cube.rotation.y += rotationSpeed;
    }

    renderer.render(scene, camera);
  }

  // Ensure the canvas and camera stay responsive on window resize.
  function onWindowResize() {
    const width = sceneContainer.clientWidth;
    const height = sceneContainer.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  }

  // UI Controls for interacting with the visualization.
  const colorInput = document.getElementById('cube-color');
  const speedInput = document.getElementById('rotation-speed');
  const toggleRotationBtn = document.getElementById('toggle-rotation');
  const toggleWireframeBtn = document.getElementById('toggle-wireframe');
  const resetViewBtn = document.getElementById('reset-view');

  colorInput.addEventListener('input', () => {
    cube.material.color.set(colorInput.value);
  });

  speedInput.addEventListener('input', () => {
    rotationSpeed = parseFloat(speedInput.value);
  });

  toggleRotationBtn.addEventListener('click', () => {
    isRotating = !isRotating;
    toggleRotationBtn.textContent = isRotating ? 'Pause Rotation' : 'Resume Rotation';
  });

  toggleWireframeBtn.addEventListener('click', () => {
    cube.material.wireframe = !cube.material.wireframe;
  });

  resetViewBtn.addEventListener('click', () => {
    camera.position.set(2, 2, 5);
    camera.lookAt(0, 0, 0);
  });

  window.addEventListener('resize', onWindowResize);

  // Initialize everything when the script loads.
  initScene();
}
