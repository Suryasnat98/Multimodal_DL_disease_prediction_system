/* ════════════════════════════════════════════════════════════════
   MedAI Dashboard — script.js
   Production-grade JS: real Flask API, GradCAM, Suggestions,
   Voice, History (localStorage), Reports, Modal, Auth.
   NO fake simulation logic.
════════════════════════════════════════════════════════════════ */

'use strict';

// ─── GLOBAL STATE ─────────────────────────────
let historyData = [];
let histFilter = 'all';

let recognition = null;
let isListening = false;


// Voice cache fix
let availableVoices = [];

speechSynthesis.onvoiceschanged = () => {
  availableVoices = speechSynthesis.getVoices();
};

// ─── CONSTANTS ────────────────────────────────────────────────────────────────
const FLASK_BASE = '';               // same-origin; set to 'http://127.0.0.1:5000' if needed
const HISTORY_KEY = 'medai_history_v2';
const DB_KEY = 'medai_users_v1';
const SESSION_KEY = 'medai_session_v1';

// Prediction colour palette
const PRED_COLORS = [
  '#00e5ff', '#2979ff', '#d500f9', '#ff2d78', '#ffab00',
  '#00e676', '#ff6b35', '#7c4dff', '#18ffff', '#69f0ae'
];

// ─── TOAST SYSTEM ─────────────────────────────────────────────────────────────
function showToast(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  const icons = { error: '❌', success: '✅', info: 'ℹ️', warning: '⚠️' };
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.classList.add('hiding');
    setTimeout(() => toast.remove(), 350);
  }, duration);
}

// ─── THEME ────────────────────────────────────────────────────────────────────
let currentTheme = localStorage.getItem('medai_theme') || 'dark';
applyTheme(currentTheme);

function applyTheme(t) {
  currentTheme = t;
  document.documentElement.setAttribute('data-theme', t);
  localStorage.setItem('medai_theme', t);
}
function toggleTheme() { applyTheme(currentTheme === 'dark' ? 'light' : 'dark'); }

// ─── AUTH ─────────────────────────────────────────────────────────────────────
function getUsers() {
  try { return JSON.parse(localStorage.getItem(DB_KEY) || '[]'); } catch { return []; }
}
function saveUsers(u) { localStorage.setItem(DB_KEY, JSON.stringify(u)); }
function getSession() {
  try { return JSON.parse(sessionStorage.getItem(SESSION_KEY)); } catch { return null; }
}
function validateEmail(e) { return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e); }

function clearAuthUI() {
  document.querySelectorAll('.auth-alert').forEach(el => el.classList.remove('show'));
  document.querySelectorAll('.err-msg').forEach(el => el.classList.remove('show'));
  document.querySelectorAll('.auth-input').forEach(el => el.classList.remove('field-err'));
}
function showAuthAlert(id, msg) {
  const txtEl = document.getElementById(id + 'Txt');
  if (txtEl) txtEl.textContent = msg;
  document.getElementById(id).classList.add('show');
}
function fieldError(inputId, errId, msg) {
  document.getElementById(inputId).classList.add('field-err');
  const e = document.getElementById(errId);
  e.textContent = msg; e.classList.add('show');
}
function setBtnLoading(prefix, loading, label) {
  document.getElementById(prefix + 'Btn').disabled = loading;
  document.getElementById(prefix + 'Spin').style.display = loading ? 'block' : 'none';
  document.getElementById(prefix + 'BtnTxt').textContent =
    loading ? (prefix === 'login' ? 'Signing in…' : 'Creating account…') : label;
}
function togglePw(inputId, btn) {
  const inp = document.getElementById(inputId);
  inp.type = inp.type === 'password' ? 'text' : 'password';
  btn.textContent = inp.type === 'text' ? '🙈' : '👁';
}
function switchPane(to) {
  const fromEl = document.querySelector('.auth-pane.active');
  const toEl = document.getElementById('pane' + to[0].toUpperCase() + to.slice(1));
  if (!toEl) return;
  clearAuthUI();
  if (fromEl) fromEl.classList.remove('active');
  toEl.classList.add('active');
}
function checkStrength(pw) {
  const checks = { len: pw.length >= 8, upper: /[A-Z]/.test(pw), num: /[0-9]/.test(pw), special: /[^A-Za-z0-9]/.test(pw) };
  Object.entries(checks).forEach(([k, v]) => { const el = document.getElementById('req-' + k); if (el) el.classList.toggle('met', v); });
  document.getElementById('pwReqs').classList.toggle('show', pw.length > 0);
  const score = Object.values(checks).filter(Boolean).length;
  const fill = document.getElementById('strengthFill');
  const label = document.getElementById('strengthLabel');
  const cfg = [null, ['15%', '#ff2d78', 'Weak'], ['35%', '#ff6b35', 'Fair'], ['60%', '#ffab00', 'Good'], ['100%', '#00e676', 'Strong']];
  const c = cfg[score] || cfg[1];
  fill.style.width = pw.length ? c[0] : '0%';
  fill.style.background = c[1];
  label.textContent = pw.length ? c[2] : '';
  label.style.color = c[1];
}

function doLogin() {
  clearAuthUI();
  const email = document.getElementById('liEmail').value.trim().toLowerCase();
  const pass = document.getElementById('liPass').value;
  let valid = true;
  if (!email) { fieldError('liEmail', 'liEmailErr', 'Email is required'); valid = false; }
  else if (!validateEmail(email)) { fieldError('liEmail', 'liEmailErr', 'Enter a valid email'); valid = false; }
  if (!pass) { fieldError('liPass', 'liPassErr', 'Password is required'); valid = false; }
  if (!valid) return;
  setBtnLoading('login', true);
  setTimeout(() => {
    setBtnLoading('login', false, 'Sign In');
    const user = getUsers().find(u => u.email === email && u.password === btoa(pass));
    if (!user) { showAuthAlert('loginAlert', 'Incorrect email or password.'); document.getElementById('liPass').value = ''; return; }
    sessionStorage.setItem(SESSION_KEY, JSON.stringify(user));
    launchApp(user);
  }, 900);
}

function doSignup() {
  clearAuthUI();
  const first = document.getElementById('suFirst').value.trim();
  const last = document.getElementById('suLast').value.trim();
  const email = document.getElementById('suEmail').value.trim().toLowerCase();
  const pass = document.getElementById('suPass').value;
  const conf = document.getElementById('suConf').value;
  let valid = true;
  if (!first || first.length < 2) { fieldError('suFirst', 'suFirstErr', 'Enter your first name'); valid = false; }
  if (!last || last.length < 2) { fieldError('suLast', 'suLastErr', 'Enter your last name'); valid = false; }
  if (!email) { fieldError('suEmail', 'suEmailErr', 'Email is required'); valid = false; }
  else if (!validateEmail(email)) { fieldError('suEmail', 'suEmailErr', 'Enter a valid email'); valid = false; }
  if (!pass) { fieldError('suPass', 'suPassErr', 'Password is required'); valid = false; }
  else if (pass.length < 8) { fieldError('suPass', 'suPassErr', 'Min 8 characters'); valid = false; }
  if (!conf) { fieldError('suConf', 'suConfErr', 'Confirm your password'); valid = false; }
  else if (pass !== conf) { fieldError('suConf', 'suConfErr', 'Passwords do not match'); valid = false; }
  if (!valid) return;
  setBtnLoading('signup', true);
  setTimeout(() => {
    setBtnLoading('signup', false, 'Create Account');
    const users = getUsers();
    if (users.find(u => u.email === email)) { showAuthAlert('signupAlert', 'Email already registered.'); return; }
    const user = { id: 'u_' + Date.now(), name: first + ' ' + last, email, password: btoa(pass), created: new Date().toISOString() };
    users.push(user); saveUsers(users);
    document.getElementById('signupSuccess').classList.add('show');
    ['suFirst', 'suLast', 'suEmail', 'suPass', 'suConf'].forEach(id => { document.getElementById(id).value = ''; });
    checkStrength('');
    setTimeout(() => switchPane('login'), 1800);
  }, 1000);
}

function launchApp(user) {
  const initial = (user.name || user.email).charAt(0).toUpperCase();
  document.getElementById('userAvatar').textContent = initial;
  document.getElementById('userDisplayName').textContent = user.name || user.email;
  const authPage = document.getElementById('authPage');
  const mainApp = document.getElementById('mainApp');
  authPage.classList.add('hiding');
  mainApp.style.display = 'flex';
  setTimeout(() => { authPage.style.display = 'none'; }, 600);
  updateHistBadge();
}

function doLogout() {
  sessionStorage.removeItem(SESSION_KEY);
  document.getElementById('authPage').style.display = 'flex';
  document.getElementById('authPage').classList.remove('hiding');
  document.getElementById('mainApp').style.display = 'none';
  clearAuthUI();
  switchPane('login');
}
//

function loadHistory() {
  try { historyData = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); } catch { historyData = []; }
}
loadHistory();
updateHistBadge();


// Restore session on load
const _session = getSession();
if (_session) launchApp(_session);

// Enter key in auth
document.addEventListener('keydown', e => {
  if (e.key !== 'Enter') return;
  if (document.getElementById('paneLogin')?.classList.contains('active')) doLogin();
  if (document.getElementById('paneSignup')?.classList.contains('active')) doSignup();
});

// ─── HISTORY STORE ────────────────────────────────────────────────────────────

function saveHistory() {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(historyData));
  updateHistBadge();
}
function updateHistBadge() {
  document.getElementById('histBadge').textContent = historyData.length;
}
loadHistory();
updateHistBadge();

// ─── PAGE NAVIGATION ──────────────────────────────────────────────────────────
function showPage(id, btn) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
  document.getElementById('pg-' + id).classList.add('active');
  btn.classList.add('active');
  if (id === 'history') renderHistory();
  if (id === 'reports') renderReports();
}

// ─── FILE UPLOAD ──────────────────────────────────────────────────────────────
let currentFile = null;

function handleFile(e) {
  const f = e.target.files[0];
  if (!f) return;
  if (f.size > 5 * 1024 * 1024) { showToast('File too large. Max 5 MB.', 'error'); return; }
  currentFile = f;
  document.getElementById('fileName').textContent = f.name + ' (' + (f.size / 1024).toFixed(0) + ' KB)';
  const r = new FileReader();
  r.onload = ev => {
    const img = document.getElementById('previewImg');
    img.src = ev.target.result; img.style.display = 'block';
  };
  r.readAsDataURL(f);
}

const dz = document.getElementById('dropZone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('drag');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) {
    const dt = new DataTransfer(); dt.items.add(f);
    document.getElementById('fileInput').files = dt.files;
    handleFile({ target: { files: [f] } });
  }
});

// ─── ANALYSIS — MAIN ─────────────────────────────────────────────────────────
async function runAnalysis() {
  const symp = document.getElementById('sympInput').value.trim();
  if (!symp && !currentFile) {
    showToast('Please enter symptoms or upload an image.', 'warning');
    return;
  }

  // UI: set loading
  setAnalyzeLoading(true);
  showShimmer();

  try {
    const formData = new FormData();
    if (symp) formData.append('symptoms', symp);
    formData.append('language', selectedLang);
    if (currentFile) formData.append('image', currentFile);

    const res = await fetch(`${FLASK_BASE}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: `Server error ${res.status}` }));
      throw new Error(err.error || `HTTP ${res.status}`);
    }

    const data = await res.json();
    hideShimmer();

    if (data.error) throw new Error(data.error);

    handlePredictionResponse(data, symp);

  } catch (err) {
    hideShimmer();
    showWaitState();
    if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
      showToast('Cannot reach the Flask server. Make sure app.py is running.', 'error', 6000);
    } else {
      showToast('Analysis failed: ' + err.message, 'error');
    }
    console.error('[MedAI] Prediction error:', err);
  } finally {
    setAnalyzeLoading(false);
  }
}

function setAnalyzeLoading(loading) {
  const btn = document.getElementById('analyzeBtn');
  const spin = document.getElementById('spinner');
  const txt = document.getElementById('btnTxt');
  btn.disabled = loading;
  spin.style.display = loading ? 'block' : 'none';
  txt.innerHTML = loading ? ' Analyzing…' : '⚡ &nbsp;Analyze';
}

function showShimmer() {
  document.getElementById('waitState').style.display = 'none';
  document.getElementById('resultState').style.display = 'none';
  document.getElementById('conflictState').style.display = 'none';
  document.getElementById('shimmerState').style.display = 'block';
  // CAM shimmer
  document.getElementById('camEmpty').style.display = 'none';
  document.getElementById('camImg').style.display = 'none';
  document.getElementById('camShimmer').style.display = 'block';
}

function hideShimmer() {
  document.getElementById('shimmerState').style.display = 'none';
  document.getElementById('camShimmer').style.display = 'none';
}

function showWaitState() {
  document.getElementById('waitState').style.display = 'flex';
  document.getElementById('resultState').style.display = 'none';
  document.getElementById('conflictState').style.display = 'none';
  document.getElementById('camEmpty').style.display = 'block';
  document.getElementById('camImg').style.display = 'none';
}

// ─── HANDLE BACKEND RESPONSE ─────────────────────────────────────────────────
function handlePredictionResponse(data, symptoms) {
  const mode = data.mode;

  // Update mode tag
  const modeLabels = {
    image_only: '🖼 Image Analysis Only',
    symptom_only: '📝 Symptoms Analysis Only',
    both_agree: '🔗 Image + Symptoms (Agreed)',
    conflict: '⚡ Conflict — Models Disagree',
  };
  document.getElementById('modeTag').textContent = modeLabels[mode] || mode;

  if (mode === 'conflict') {
    showConflictResult(data, symptoms);
    return;
  }

  // Normal result path
  const conf = Math.round(data.confidence * 100);
  const disease = data.primary_disease;
  const topPreds = data.top_predictions || [];
  const suggestions = data.suggestions || [];

  // Classify type
  const diseaseLower = disease.toLowerCase();
  let type = 'warning';
  if (diseaseLower.includes('benign')) type = 'benign';
  else if (diseaseLower.includes('malignant') || diseaseLower.includes('cancer') || diseaseLower.includes('melanoma')) type = 'malignant';

  const icon = type === 'benign' ? '✅' : type === 'malignant' ? '🚨' : '⚠️';
  const badge = conf >= 70 ? 'high' : conf >= 40 ? 'medium' : 'low';

  showNormalResult({ disease, conf, type, icon, badge, topPreds, suggestions, mode });
  updateGradCAM(data.images);

  // Save to history
  const record = buildHistoryRecord({ disease, conf, type, icon, badge, topPreds, suggestions, mode, symptoms });
  historyData.unshift(record);
  saveHistory();

  showToast(`Analysis complete: ${disease} (${conf}%)`, 'success');

  ///////////////
  // speakDiagnosis(data);
  if (selectedLang === 'hi-IN') {

    speakDiagnosis(
      data.hindi_response,
      'hi-IN'
    );

  } else {

    speakDiagnosis(
      data.english_response,
      'en-US'
    );
  }
}



// // ─── TEXT TO SPEECH ─────────────────────────────────────────
function speakDiagnosis(text, lang = 'en-US') {

  if (!window.speechSynthesis) {
    console.log('Speech synthesis not supported');
    return;
  }

  if (!text || text.trim() === '') {
    console.log('No speech text received');
    return;
  }

  // Stop previous speech
  window.speechSynthesis.cancel();

  // Small delay fixes Chrome speech bug
  setTimeout(() => {

    const utterance = new SpeechSynthesisUtterance(text);

    utterance.lang = lang;
    utterance.rate = 0.9;
    utterance.pitch = 1;
    utterance.volume = 1;

    // Reload voices safely
    const voices = window.speechSynthesis.getVoices();

    console.log('Available voices:', voices);

    let selectedVoice = null;

    if (lang === 'hi-IN') {

      selectedVoice =
        voices.find(v => v.lang === 'hi-IN') ||
        voices.find(v => v.lang.includes('hi'));

    } else {

      selectedVoice =
        voices.find(v => v.lang === 'en-US') ||
        voices.find(v => v.lang.includes('en'));
    }

    if (selectedVoice) {
      utterance.voice = selectedVoice;
      console.log('Using voice:', selectedVoice.name);
    }

    utterance.onstart = () => {
      console.log('Speech started');
    };

    utterance.onend = () => {
      console.log('Speech finished');
    };

    utterance.onerror = (e) => {
      console.error('Speech error:', e);
    };

    window.speechSynthesis.speak(utterance);

  }, 300);
}
// function speakDiagnosis(text, lang = 'en-US') {

//   if (!window.speechSynthesis) return;

//   speechSynthesis.cancel();

//   const utterance = new SpeechSynthesisUtterance(text);

//   utterance.lang = lang;

//   utterance.rate = 0.95;
//   utterance.pitch = 1;

//   // Better Hindi voice selection
//   const voices = availableVoices;

//   const hindiVoice = voices.find(v =>
//     v.lang.includes('hi')
//   );

//   const englishVoice = voices.find(v =>
//     v.lang.includes('en')
//   );

//   if (lang === 'hi-IN' && hindiVoice) {
//     utterance.voice = hindiVoice;
//   } else if (englishVoice) {
//     utterance.voice = englishVoice;
//   }

//   speechSynthesis.speak(utterance);
// }


function showNormalResult({ disease, conf, type, icon, badge, topPreds, suggestions, mode }) {
  document.getElementById('resultState').style.display = 'block';
  document.getElementById('conflictState').style.display = 'none';
  document.getElementById('waitState').style.display = 'none';

  // Banner
  const banner = document.getElementById('diagBanner');
  const colorMap = {
    benign: ['linear-gradient(135deg,rgba(0,230,118,.10),rgba(0,229,255,.06))', 'rgba(0,230,118,.25)'],
    malignant: ['linear-gradient(135deg,rgba(255,45,120,.10),rgba(255,45,120,.04))', 'rgba(255,45,120,.3)'],
    warning: ['linear-gradient(135deg,rgba(255,171,0,.10),rgba(255,171,0,.04))', 'rgba(255,171,0,.3)'],
  };
  const [bg, bc] = colorMap[type] || colorMap.warning;
  banner.style.background = bg;
  banner.style.borderColor = bc;

  const diagColors = { benign: 'var(--green)', malignant: 'var(--pink)', warning: 'var(--amber)' };
  document.getElementById('diagVal').textContent = disease;
  document.getElementById('diagVal').style.color = diagColors[type] || 'var(--cyan)';
  document.getElementById('diagIcon').textContent = icon;

  // Confidence
  document.getElementById('confNum').textContent = conf + '%';
  const fill = document.getElementById('confFill');
  fill.style.animation = 'none';
  void fill.offsetWidth;
  fill.style.width = conf + '%';

  // Colour confidence bar by risk
  if (type === 'malignant') fill.style.background = 'linear-gradient(90deg,var(--pink),#ff6b35)';
  else if (type === 'warning') fill.style.background = 'linear-gradient(90deg,var(--amber),#ff9800)';
  else fill.style.background = 'linear-gradient(90deg,var(--green),var(--cyan))';

  fill.style.animation = 'growW 1s ease-out forwards';

  const tagLabels = { high: '🟢 High Confidence', medium: '🟡 Medium Confidence', low: '🔴 Low Confidence' };
  const tagColors = { high: 'var(--green)', medium: 'var(--amber)', low: 'var(--pink)' };
  document.getElementById('confTag').innerHTML = tagLabels[badge] || badge;
  document.getElementById('confTag').style.color = tagColors[badge] || 'var(--text2)';

  // Top predictions
  document.getElementById('predList').innerHTML = topPreds.map(([name, prob], i) => {
    const pct = Math.round(prob * 100);
    const col = PRED_COLORS[i] || PRED_COLORS[0];
    return `<div class="pred-row">
      <div class="pred-dot" style="background:${col}"></div>
      <div class="pred-name">${name}</div>
      <div class="pred-track"><div class="pred-fill" style="background:${col};width:${pct}%"></div></div>
      <div class="pred-pct">${pct}%</div>
    </div>`;
  }).join('');

  // Suggestions
  renderSuggestionsCard(suggestions, badge);

  // Insight
  const insightMap = {
    benign: '💬 The analysis shows low-risk indicators. Regular monitoring is still advised.',
    malignant: '💬 High-risk indicators detected. Immediate consultation with a dermatologist is strongly recommended.',
    warning: '💬 Moderate risk indicators found. Professional evaluation is advised to rule out serious conditions.',
  };
  document.getElementById('insightMini').textContent = insightMap[type] || '💬 Review results with a qualified physician.';
}

function renderSuggestionsCard(suggestions, badge) {
  const card = document.getElementById('suggCard');
  const list = document.getElementById('suggList');
  const tag = document.getElementById('suggRiskTag');
  if (!suggestions || !suggestions.length) { card.style.display = 'none'; return; }

  // Remove old risk class
  card.classList.remove('high-risk', 'low-risk');
  tag.classList.remove('high', 'medium', 'low');

  if (badge === 'low') {
    card.classList.add('high-risk');
    tag.classList.add('high');
    tag.textContent = '🔴 High Risk';
  } else if (badge === 'high') {
    card.classList.add('low-risk');
    tag.classList.add('low');
    tag.textContent = '🟢 Low Risk';
  } else {
    tag.classList.add('medium');
    tag.textContent = '🟡 Medium Risk';
  }

  list.innerHTML = suggestions.map(s => {
    const isUrgent = s.startsWith('⚠️') || s.toLowerCase().includes('immediately') || s.toLowerCase().includes('seek');
    const isSafe = s.toLowerCase().includes('non-cancerous') || s.toLowerCase().includes('monitor') || s.toLowerCase().includes('routine');
    const cls = isUrgent ? 'urgent' : isSafe ? 'safe' : 'normal';
    return `<div class="sugg-item ${cls}">
      <div class="sugg-bullet"></div>
      <span>${s}</span>
    </div>`;
  }).join('');

  card.style.display = 'block';
}

function showConflictResult(data, symptoms) {
  document.getElementById('resultState').style.display = 'none';
  document.getElementById('conflictState').style.display = 'block';
  document.getElementById('waitState').style.display = 'none';

  const imgConf = Math.round((data.image_confidence || 0) * 100);
  const symConf = Math.round((data.symptom_confidence || 0) * 100);

  document.getElementById('conflictImgDiag').textContent = data.image_prediction || '—';
  document.getElementById('conflictSymDiag').textContent = data.symptom_prediction || '—';
  document.getElementById('conflictImgConf').textContent = imgConf + '% confidence';
  document.getElementById('conflictSymConf').textContent = symConf + '% confidence';

  const sugg = data.suggestions || ['Consult a doctor for accurate diagnosis.'];
  document.getElementById('conflictSugg').innerHTML = sugg.map(s => `
    <div class="sugg-item urgent">
      <div class="sugg-bullet"></div>
      <span>${s}</span>
    </div>`).join('');

  updateGradCAM(data.images);

  // Save conflict record
  const record = {
    id: 'MED-' + Date.now(),
    date: new Date().toLocaleString(),
    timestamp: Date.now(),
    diag: `⚡ Conflict: ${data.image_prediction} vs ${data.symptom_prediction}`,
    type: 'conflict',
    conf: Math.max(imgConf, symConf),
    badge: 'medium',
    icon: '⚡',
    symptoms: symptoms || '(no symptoms entered)',
    preds: (data.top_predictions || []).map(([n, p], i) => ({ n, p: Math.round(p * 100), c: PRED_COLORS[i] })),
    insight: 'Model disagreement detected. Manual medical review required.',
    mode: 'conflict',
    suggestions: data.suggestions || [],
  };
  historyData.unshift(record);
  saveHistory();

  showToast('⚡ Model conflict detected — consult a doctor for accurate diagnosis.', 'warning', 6000);

  // speakDiagnosis(data);
  if (selectedLang === 'hi-IN') {

    speakDiagnosis(
      data.hindi_response ||
      'कृपया डॉक्टर से सलाह लें।',
      'hi-IN'
    );

  } else {

    speakDiagnosis(
      data.english_response ||
      'Please consult a doctor.',
      'en-US'
    );
  }
}

// ─── GRADCAM ─────────────────────────────────────────────────────────────────
// Store current image URLs so tab switching works
let _camImages = { original: null, heatmap: null, overlay: null };
let _activeCamTab = 'original';

function updateGradCAM(images) {
  const camImg = document.getElementById('camImg');
  const camEmpty = document.getElementById('camEmpty');

  if (!images || (!images.heatmap && !images.original)) {
    camEmpty.style.display = 'block';
    camImg.style.display = 'none';
    return;
  }

  _camImages = {
    original: images.original || null,
    heatmap: images.heatmap || null,
    overlay: images.overlay || null,
  };

  // Show active tab image
  setCamImage(_activeCamTab);
  camEmpty.style.display = 'none';
}

function setCamImage(mode) {
  const camImg = document.getElementById('camImg');
  const url = _camImages[mode];

  // if (!url) return;
  if (!url) {
    camImg.style.display = 'none';
    document.getElementById('camEmpty').style.display = 'block';
    return;
  }

  // Cache-bust with timestamp to avoid stale images
  const bust = '?t=' + Date.now();
  camImg.style.opacity = '0';
  camImg.src = FLASK_BASE + url + bust;
  camImg.style.display = 'block';
  camImg.onload = () => { camImg.style.opacity = '1'; };
  camImg.onerror = () => {
    camImg.style.display = 'none';
    document.getElementById('camEmpty').style.display = 'block';
  };
}

function camTab(btn, mode) {
  document.querySelectorAll('.cam-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  _activeCamTab = mode;
  // Only switch if we have images loaded
  if (_camImages.original) setCamImage(mode);
}

// ─── ONLINE VOICE INPUT (Browser SpeechRecognition) ─────────────────────────


// // Load browser voices properly
// speechSynthesis.onvoiceschanged = () => {
//   speechSynthesis.getVoices();
// };

// Initialize browser speech recognition
function initVoice() {
  const SpeechRecognition =
    window.SpeechRecognition ||
    window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    showToast('Speech Recognition not supported in this browser.', 'error');
    return null;
  }

  recognition = new SpeechRecognition();

  recognition.continuous = false;
  recognition.interimResults = false;
  // recognition.lang = selectedLang || 'en-US';
  recognition.lang =
    selectedLang === 'hi-IN'
      ? 'hi-IN'
      : 'en-US';

  recognition.onstart = () => {
    isListening = true;

    const micBtn = document.getElementById('micBtn');
    const statusEl = document.getElementById('voiceStatus');
    const statusTxt = document.getElementById('voiceStatusTxt');

    micBtn.classList.add('loading-voice');
    micBtn.textContent = '🎙';

    statusEl.classList.add('show');
    statusTxt.textContent = 'Listening... Speak now';
  };

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;

    document.getElementById('sympInput').value = transcript;

    showToast('Voice captured successfully!', 'success');

    document.getElementById('voiceStatusTxt').textContent =
      `Captured: "${transcript}"`;
  };

  recognition.onerror = (event) => {
    console.error(event);

    showToast('Voice error: ' + event.error, 'error');

    document.getElementById('voiceStatusTxt').textContent =
      'Voice error: ' + event.error;
  };

  recognition.onend = () => {
    isListening = false;

    const micBtn = document.getElementById('micBtn');

    micBtn.classList.remove('loading-voice');
    micBtn.textContent = '🎙';

    setTimeout(() => {
      document.getElementById('voiceStatus')
        .classList.remove('show');
    }, 2500);
  };

  return recognition;
}

// Start voice recognition
function startVoice() {
  if (isListening) return;

  if (!recognition) {
    recognition = initVoice();
  }

  if (!recognition) return;

  // recognition.lang = selectedLang || 'en-US';
  recognition.lang =
    selectedLang === 'hi-IN'
      ? 'hi-IN'
      : 'en-US';

  recognition.start();
}

// Language toggle (kept for UI, server handles language internally)
let selectedLang = 'hi-IN';
function setLang(btn) {
  document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  selectedLang = btn.dataset.lang;
}

// ─── HISTORY ─────────────────────────────────────────────────────────────────
function setFilter(btn, val) {
  document.querySelectorAll('.filt-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  histFilter = val;
  renderHistory();
}

function renderHistory() {
  const q = (document.getElementById('histSearch').value || '').toLowerCase();
  let data = [...historyData];
  if (histFilter !== 'all') data = data.filter(d => d.type === histFilter);
  if (q) data = data.filter(d => (d.diag + (d.symptoms || '')).toLowerCase().includes(q));

  const grid = document.getElementById('histGrid');
  const empty = document.getElementById('histEmpty');

  if (!data.length) { grid.innerHTML = ''; empty.style.display = 'block'; return; }
  empty.style.display = 'none';

  grid.innerHTML = data.map(d => {
    const typeColor = d.type === 'benign' ? 'var(--green)' : d.type === 'conflict' ? 'var(--amber)' : d.type === 'malignant' ? 'var(--pink)' : 'var(--amber)';
    const badgeLabel = d.badge === 'high' ? '🟢 High' : d.badge === 'medium' ? '🟡 Medium' : '🔴 Low';
    const predsHTML = (d.preds || []).slice(0, 3).map(p => {
      const name = typeof p.n === 'string' ? p.n.split(' ')[0] : '—';
      return `<span class="hc-tag">${name}: ${p.p}%</span>`;
    }).join('');
    return `<div class="hist-card" onclick="openModal('${d.id}')">
      <div class="hc-head">
        <span class="hc-id">${d.id}</span>
        <span class="hc-date">${d.date}</span>
      </div>
      <div class="hc-mini-bar"><div class="hc-mini-fill" style="width:${d.conf}%;background:${typeColor}"></div></div>
      <div class="hc-diag ${d.type}">${d.icon} ${d.diag}</div>
      <div class="hc-symp">${(d.symptoms || '').length > 60 ? d.symptoms.slice(0, 60) + '…' : d.symptoms || '—'}</div>
      <div class="hc-foot">
        <span class="hc-conf">${d.conf}%</span>
        <span class="hc-badge ${d.badge}">${badgeLabel}</span>
      </div>
      <div class="hc-tags" style="margin-top:8px;">
        <span class="hc-tag">${d.mode || '—'}</span>
        ${predsHTML}
      </div>
    </div>`;
  }).join('');
}

// ─── REPORTS ─────────────────────────────────────────────────────────────────
function renderReports() {
  const data = historyData;
  const total = data.length;
  const benign = data.filter(d => d.type === 'benign').length;
  const mal = data.filter(d => d.type !== 'benign').length;
  const avgConf = total ? (data.reduce((a, b) => a + b.conf, 0) / total).toFixed(1) + '%' : '—';

  document.getElementById('rTotal').textContent = total;
  document.getElementById('rBenign').textContent = benign;
  document.getElementById('rBenignPct').textContent = total ? ((benign / total) * 100).toFixed(0) + '% of total' : '—';
  document.getElementById('rMal').textContent = mal;
  document.getElementById('rMalPct').textContent = total ? ((mal / total) * 100).toFixed(0) + '% of total' : '—';
  document.getElementById('rAvgConf').textContent = avgConf;

  // Distribution chart
  const counts = {};
  data.forEach(d => { counts[d.diag] = (counts[d.diag] || 0) + 1; });
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const maxC = sorted[0]?.[1] || 1;
  const colorMap = { benign: '#00e676', malignant: '#ff2d78', warning: '#ffab00', conflict: '#ffab00' };

  document.getElementById('distChart').innerHTML = sorted.length
    ? sorted.map(([name, cnt]) => {
      const type = data.find(d => d.diag === name)?.type || 'warning';
      const col = colorMap[type] || '#00e5ff';
      return `<div class="bc-row">
          <div class="bc-label" title="${name}">${name}</div>
          <div class="bc-track"><div class="bc-fill" style="width:${(cnt / maxC * 100).toFixed(0)}%;background:${col}"></div></div>
          <div class="bc-val">${cnt}</div>
        </div>`;
    }).join('')
    : '<div style="color:var(--text3);font-size:12px;">No data yet.</div>';

  // Timeline
  document.getElementById('tlChart').innerHTML = data.slice(0, 8).length
    ? data.slice(0, 8).map(d => {
      const col = d.type === 'benign' ? 'var(--green)' : d.type === 'warning' || d.type === 'conflict' ? 'var(--amber)' : 'var(--pink)';
      return `<div class="tl-row">
          <div class="tl-date">${new Date(d.timestamp).toLocaleDateString('en', { month: 'short', day: 'numeric' })}</div>
          <div class="tl-dot" style="background:${col}"></div>
          <div class="tl-label">${d.diag.slice(0, 30)}</div>
          <div class="tl-conf">${d.conf}%</div>
        </div>`;
    }).join('')
    : '<div style="color:var(--text3);font-size:12px;">No data yet.</div>';

  // Table
  const tbody = document.getElementById('repTable');
  const rempty = document.getElementById('repEmpty');
  if (!data.length) { tbody.innerHTML = ''; rempty.style.display = 'block'; return; }
  rempty.style.display = 'none';
  const tcMap = { benign: 'var(--green)', warning: 'var(--amber)', malignant: 'var(--pink)', conflict: 'var(--amber)' };
  tbody.innerHTML = data.map((d, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${d.date}</td>
      <td class="td-diag" style="color:${tcMap[d.type] || 'var(--cyan)'}">${d.icon} ${d.diag}</td>
      <td class="td-conf">${d.conf}%</td>
      <td>${(d.symptoms || '').length > 40 ? (d.symptoms || '').slice(0, 40) + '…' : d.symptoms || '—'}</td>
      <td><span class="hc-badge ${d.badge}">${d.badge}</span></td>
    </tr>`).join('');
}

// ─── MODAL ────────────────────────────────────────────────────────────────────
function openModal(id) {
  const d = historyData.find(x => x.id === id);
  if (!d) return;
  document.getElementById('modalTitle').textContent = d.id + ' — ' + d.date;
  const col = d.type === 'benign' ? 'green' : d.type === 'conflict' || d.type === 'warning' ? 'amber' : 'pink';

  const predsHTML = (d.preds || []).map(p => {
    const pct = typeof p.p === 'number' ? p.p : 0;
    const col2 = p.c || 'var(--cyan)';
    return `<div class="pred-row">
      <div class="pred-dot" style="background:${col2}"></div>
      <div class="pred-name">${p.n || '—'}</div>
      <div class="pred-track"><div class="pred-fill" style="background:${col2};width:${pct}%"></div></div>
      <div class="pred-pct">${pct}%</div>
    </div>`;
  }).join('');

  const suggHTML = (d.suggestions || []).length
    ? `<div class="md-item" style="margin-bottom:12px;">
        <label>AI Recommendations</label>
        <div class="md-sugg-list">${(d.suggestions || []).map(s => `<div class="md-sugg-item">${s}</div>`).join('')}</div>
       </div>`
    : '';

  document.getElementById('modalBody').innerHTML = `
    <div class="md-row">
      <div class="md-item"><label>Diagnosis</label><div class="mdv ${col}">${d.icon} ${d.diag}</div></div>
      <div class="md-item"><label>Confidence</label><div class="mdv cyan">${d.conf}%</div></div>
    </div>
    <div class="md-row">
      <div class="md-item"><label>Mode</label><div class="mdv" style="color:var(--text2)">${d.mode || '—'}</div></div>
      <div class="md-item"><label>Badge</label><div class="mdv ${col}">${(d.badge || '').charAt(0).toUpperCase() + (d.badge || '').slice(1)} Confidence</div></div>
    </div>
    <div class="md-item" style="margin-bottom:12px;">
      <label>Symptoms</label>
      <div class="mdv" style="color:var(--text2);font-size:13px;margin-top:4px;">${d.symptoms || '—'}</div>
    </div>
    <div class="md-item" style="margin-bottom:16px;">
      <label>AI Insight</label>
      <div class="mdv" style="color:var(--text2);font-size:13px;font-weight:400;line-height:1.6;margin-top:4px;">${d.insight || '—'}</div>
    </div>
    ${suggHTML}
    <div class="md-preds">
      <h4>Top Predictions</h4>
      ${predsHTML || '<div style="color:var(--text3);font-size:12px;">No predictions.</div>'}
    </div>`;
  document.getElementById('modalOverlay').classList.add('open');
}
function closeModal(e) { if (e.target === document.getElementById('modalOverlay')) closeModalBtn(); }
function closeModalBtn() { document.getElementById('modalOverlay').classList.remove('open'); }

// ─── EXPORT CSV ───────────────────────────────────────────────────────────────
function exportCSV() {
  if (!historyData.length) { showToast('No data to export.', 'warning'); return; }
  const rows = [['ID', 'Date', 'Diagnosis', 'Type', 'Confidence', 'Badge', 'Symptoms', 'Mode']];
  historyData.forEach(d => rows.push([
    d.id, d.date, '"' + d.diag + '"', d.type, d.conf + '%', d.badge,
    '"' + (d.symptoms || '').replace(/"/g, "'") + '"', d.mode || '—'
  ]));
  const csv = rows.map(r => r.join(',')).join('\n');
  const a = document.createElement('a');
  a.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
  a.download = 'medai_report_' + Date.now() + '.csv';
  a.click();
}

// ─── HELPER: BUILD HISTORY RECORD ────────────────────────────────────────────
function buildHistoryRecord({ disease, conf, type, icon, badge, topPreds, suggestions, mode, symptoms }) {
  return {
    id: 'MED-' + Date.now(),
    date: new Date().toLocaleString(),
    timestamp: Date.now(),
    diag: disease,
    type: type,
    conf: conf,
    badge: badge,
    icon: icon,
    symptoms: symptoms || '(no symptoms entered)',
    preds: topPreds.map(([n, p], i) => ({ n, p: Math.round(p * 100), c: PRED_COLORS[i] || '#00e5ff' })),
    insight: buildInsightText(type, conf),
    mode: modeLabel(mode),
    suggestions: suggestions,
  };
}

function buildInsightText(type, conf) {
  if (type === 'benign') return 'Low-risk indicators found. Routine monitoring advised.';
  if (type === 'malignant') return `High-risk indicators (${conf}% confidence). Dermatologist consultation strongly recommended.`;
  return `Moderate-risk indicators (${conf}%). Professional evaluation advised.`;
}

function modeLabel(mode) {
  const map = { image_only: 'Image Only', symptom_only: 'Symptoms Only', both_agree: 'Image + Symptoms', conflict: 'Conflict' };
  return map[mode] || mode;
}






document.addEventListener('click', () => {
  speechSynthesis.resume();
});









// /////////////////////////////////////////////////
window.camTab = camTab;
window.runAnalysis = runAnalysis;
window.showPage = showPage;
window.doLogin = doLogin;
window.doSignup = doSignup;
window.toggleTheme = toggleTheme;
window.doLogout = doLogout;
window.startVoice = startVoice;
window.setLang = setLang;
window.exportCSV = exportCSV;
window.closeModal = closeModal;
window.closeModalBtn = closeModalBtn;
window.handleFile = handleFile;
window.setFilter = setFilter;
window.togglePw = togglePw;
window.switchPane = switchPane;
window.checkStrength = checkStrength;
window.camTab = camTab;