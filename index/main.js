async function sendMessage() {
  const input = document.getElementById('message');
  const msg = input.value.trim();
  if (!msg) return;
  const res = await fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: msg}),
  });
  const data = await res.json();
  const box = document.getElementById('chat-box');
  box.innerHTML += `<div class="user">${msg}</div>`;
  box.innerHTML += `<div class="assistant">${data.reply}</div>`;
  input.value = '';
  box.scrollTop = box.scrollHeight;
}

document.getElementById('send').addEventListener('click', sendMessage);

document.getElementById('message').addEventListener('keypress', (e) => {
  if (e.key === 'Enter') {
    sendMessage();
  }
});
