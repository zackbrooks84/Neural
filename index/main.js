const HISTORY_KEY = 'chat-history';
let history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');

function renderHistory() {
  const box = document.getElementById('chat-box');
  box.innerHTML = '';
  history.forEach(item => {
    box.innerHTML += `<div class="user">${item.user}</div>`;
    box.innerHTML += `<div class="assistant">${item.assistant}</div>`;
  });
  box.scrollTop = box.scrollHeight;
}

async function sendMessage() {
  const input = document.getElementById('message');
  const msg = input.value.trim();
  if (!msg) return;
  const useWeb = document.getElementById('use-web').checked;
  const webQueryInput = document.getElementById('web-query');
  const webQuery = webQueryInput.value.trim();
  const payload = { message: msg };
  if (useWeb) {
    payload.use_web = true;
    if (webQuery) payload.web_query = webQuery;
  }
  const res = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  history.push({ user: msg, assistant: data.reply });
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  renderHistory();
  input.value = '';
  webQueryInput.value = '';
}

document.getElementById('send').addEventListener('click', sendMessage);

document.getElementById('message').addEventListener('keypress', (e) => {
  if (e.key === 'Enter') {
    sendMessage();
  }
});

renderHistory();
