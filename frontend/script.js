/* ═══════════════════════════════════════════════════════════════
   AEGIS AI — Frontend JavaScript
   Handles: tab nav, file upload, drag-drop, API calls, history
═══════════════════════════════════════════════════════════════ */

const API = '/api';

// ── Current state ──────────────────────────────────────────────
let currentType = 'image';
let selectedFile = null;

// ── Accept types map ───────────────────────────────────────────
const ACCEPT_MAP = {
  image: 'image/*',
  video: 'video/*',
  audio: 'audio/*',
  text:  null,
};

const HINT_MAP = {
  image: 'Supports: JPG, PNG, WEBP, GIF, BMP',
  video: 'Supports: MP4, MOV, AVI, MKV, WEBM',
  audio: 'Supports: MP3, WAV, FLAC, OGG, M4A',
};

// ══════════════════════════════════════════════════════════════
//  TAB NAVIGATION
// ══════════════════════════════════════════════════════════════
document.querySelectorAll('.pill[data-tab]').forEach(btn => {
  btn.addEventListener('click', () => {
    // Pills
    document.querySelectorAll('.pill[data-tab]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    // Panels
    document.querySelectorAll('.tab-panel').forEach(p => {
      p.classList.remove('active');
      p.classList.add('hidden');
    });
    const panel = document.getElementById('tab-' + btn.dataset.tab);
    panel.classList.remove('hidden');
    panel.classList.add('active');

    // Auto-load history
    if (btn.dataset.tab === 'history') loadHistory();
  });
});

// ══════════════════════════════════════════════════════════════
//  MEDIA TYPE SELECTOR
// ══════════════════════════════════════════════════════════════
document.querySelectorAll('.type-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.type-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    currentType = btn.dataset.type;
    selectedFile = null;
    hideResult();
    resetFilePreview();

    const fileZone = document.getElementById('file-upload-zone');
    const textZone = document.getElementById('text-upload-zone');
    const hint     = document.getElementById('upload-hint');
    const input    = document.getElementById('fileInput');

    if (currentType === 'text') {
      fileZone.classList.add('hidden');
      textZone.classList.remove('hidden');
    } else {
      textZone.classList.add('hidden');
      fileZone.classList.remove('hidden');
      input.accept = ACCEPT_MAP[currentType];
      hint.textContent = HINT_MAP[currentType];
    }
  });
});

// ══════════════════════════════════════════════════════════════
//  FILE INPUT LISTENER
// ══════════════════════════════════════════════════════════════
document.getElementById('fileInput').addEventListener('change', e => {
  if (e.target.files[0]) {
    selectedFile = e.target.files[0];
    showFilePreview(selectedFile.name);
  }
});

function showFilePreview(name) {
  const prev = document.getElementById('file-preview');
  prev.textContent = '📎 ' + name;
  prev.classList.remove('hidden');
}

function resetFilePreview() {
  const prev = document.getElementById('file-preview');
  prev.textContent = '';
  prev.classList.add('hidden');
  document.getElementById('fileInput').value = '';
}

// ══════════════════════════════════════════════════════════════
//  DRAG & DROP
// ══════════════════════════════════════════════════════════════
function handleDragOver(e) {
  e.preventDefault();
  document.getElementById('file-upload-zone').classList.add('drag-over');
}

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('file-upload-zone').classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) {
    selectedFile = file;
    showFilePreview(file.name);
  }
}

// ══════════════════════════════════════════════════════════════
//  MAIN SCAN HANDLER
// ══════════════════════════════════════════════════════════════
async function handleScan() {
  hideResult();

  if (currentType === 'text') {
    const content = document.getElementById('textInput').value.trim();
    if (!content) return alert('Please enter some text to analyze.');
    await scanText(content);
    return;
  }

  if (!selectedFile) return alert('Please select a file first.');

  showLoader();

  try {
    const formData = new FormData();
    formData.append('file', selectedFile);

    const endpoint = `${API}/detect/${currentType}`;
    const res = await fetch(endpoint, { method: 'POST', body: formData });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);

    const data = await res.json();
    hideLoader();
    showMediaResult(data);
  } catch (err) {
    hideLoader();
    alert('Error: ' + err.message);
  }
}

async function scanText(content) {
  showLoader();
  try {
    const formData = new FormData();
    formData.append('content', content);

    const res = await fetch(`${API}/detect/text`, { method: 'POST', body: formData });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);

    const data = await res.json();
    hideLoader();
    showMediaResult(data);
  } catch (err) {
    hideLoader();
    alert('Error: ' + err.message);
  }
}

// ══════════════════════════════════════════════════════════════
//  LINK SCAN HANDLER
// ══════════════════════════════════════════════════════════════
async function handleLinkScan() {
  const url = document.getElementById('linkInput').value.trim();
  if (!url) return alert('Please enter a URL.');
  closeLinkResult();
  showLoader();

  try {
    const formData = new FormData();
    formData.append('url', url);

    const res = await fetch(`${API}/detect/link`, { method: 'POST', body: formData });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);

    const data = await res.json();
    hideLoader();
    showLinkResult(data);
  } catch (err) {
    hideLoader();
    alert('Error: ' + err.message);
  }
}

// Enter key on link input
document.getElementById('linkInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') handleLinkScan();
});

// ══════════════════════════════════════════════════════════════
//  RESULT DISPLAY — MEDIA
// ══════════════════════════════════════════════════════════════
function showMediaResult(data) {
  const card    = document.getElementById('result-card');
  const icon    = document.getElementById('verdict-icon');
  const text    = document.getElementById('verdict-text');
  const bar     = document.getElementById('conf-bar');
  const confVal = document.getElementById('conf-value');
  const meta    = document.getElementById('meta-block');

  const result     = data.result || 'Unknown';
  const confidence = parseFloat(data.confidence) || 0;

  // Verdict styling
  const isBad = ['fake', 'ai-generated', 'dangerous'].includes(result.toLowerCase());
  icon.textContent = isBad ? '⚠️' : '✅';
  text.textContent = result.toUpperCase();
  text.className = 'verdict-text ' + (isBad ? 'danger' : 'safe');

  // Confidence bar
  const pct = Math.min(confidence, 100);
  bar.style.width = '0%';
  bar.style.background = isBad
    ? 'linear-gradient(90deg, #cc1a40, #ff3860)'
    : 'linear-gradient(90deg, #00a870, #00e5a0)';
  setTimeout(() => { bar.style.width = pct + '%'; }, 50);
  confVal.textContent = confidence.toFixed(1) + '%';

  // Meta
  const parts = [];
  if (data.filename) parts.push(`FILE: ${data.filename}`);
  if (data.input_preview) parts.push(`TEXT: "${data.input_preview}..."`);
  parts.push(`TYPE: ${currentType.toUpperCase()}`);
  meta.innerHTML = parts.join('<br/>');

  card.classList.remove('hidden');
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideResult() {
  document.getElementById('result-card').classList.add('hidden');
}
function closeResult() { hideResult(); }

// ══════════════════════════════════════════════════════════════
//  RESULT DISPLAY — LINK
// ══════════════════════════════════════════════════════════════
function showLinkResult(data) {
  const card    = document.getElementById('link-result-card');
  const icon    = document.getElementById('link-verdict-icon');
  const text    = document.getElementById('link-verdict-text');
  const bar     = document.getElementById('link-conf-bar');
  const confVal = document.getElementById('link-conf-value');
  const meta    = document.getElementById('link-meta-block');

  const result     = data.result || 'Unknown';
  const confidence = parseFloat(data.confidence) || 0;

  const isBad = result.toLowerCase() === 'dangerous';
  icon.textContent = isBad ? '🚨' : '🛡️';
  text.textContent = result.toUpperCase();
  text.className = 'verdict-text ' + (isBad ? 'danger' : 'safe');

  const pct = Math.min(confidence, 100);
  bar.style.width = '0%';
  bar.style.background = isBad
    ? 'linear-gradient(90deg, #cc1a40, #ff3860)'
    : 'linear-gradient(90deg, #00a870, #00e5a0)';
  setTimeout(() => { bar.style.width = pct + '%'; }, 50);
  confVal.textContent = confidence.toFixed(1) + '%';

  meta.innerHTML = `URL: ${data.url}`;

  card.classList.remove('hidden');
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function closeLinkResult() {
  document.getElementById('link-result-card').classList.add('hidden');
}

// ══════════════════════════════════════════════════════════════
//  HISTORY
// ══════════════════════════════════════════════════════════════
async function loadHistory() {
  try {
    const [mediaRes, linksRes] = await Promise.all([
      fetch(`${API}/history/media`),
      fetch(`${API}/history/links`),
    ]);
    const media = await mediaRes.json();
    const links = await linksRes.json();

    renderMediaHistory(media);
    renderLinkHistory(links);
  } catch (err) {
    console.error('History load error:', err);
  }
}

function renderMediaHistory(items) {
  const el = document.getElementById('media-history');
  if (!items.length) {
    el.innerHTML = '<p class="empty-state">No scans yet.</p>';
    return;
  }
  el.innerHTML = items.map(r => {
    const isBad = ['fake','ai-generated','dangerous'].includes((r.result||'').toLowerCase());
    const badgeClass = isBad ? 'badge-danger' : 'badge-safe';
    const ts = r.timestamp ? r.timestamp.split('T')[0] : '—';
    return `
      <div class="history-item">
        <span class="history-name" title="${r.filename}">${r.filename}</span>
        <span class="history-badge ${badgeClass}">${r.result}</span>
        <span class="history-conf">${parseFloat(r.confidence).toFixed(0)}%</span>
        <span class="history-time">${ts}</span>
      </div>`;
  }).join('');
}

function renderLinkHistory(items) {
  const el = document.getElementById('link-history');
  if (!items.length) {
    el.innerHTML = '<p class="empty-state">No links scanned yet.</p>';
    return;
  }
  el.innerHTML = items.map(r => {
    const isBad = (r.status||'').toLowerCase() === 'dangerous';
    const badgeClass = isBad ? 'badge-danger' : 'badge-safe';
    const ts = r.timestamp ? r.timestamp.split('T')[0] : '—';
    const shortUrl = r.url.length > 35 ? r.url.slice(0,35) + '…' : r.url;
    return `
      <div class="history-item">
        <span class="history-name" title="${r.url}">${shortUrl}</span>
        <span class="history-badge ${badgeClass}">${r.status}</span>
        <span class="history-time">${ts}</span>
      </div>`;
  }).join('');
}

// ══════════════════════════════════════════════════════════════
//  LOADER
// ══════════════════════════════════════════════════════════════
function showLoader() { document.getElementById('loader').classList.remove('hidden'); }
function hideLoader() { document.getElementById('loader').classList.add('hidden'); }
