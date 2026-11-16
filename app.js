/* SoundCue UI logic */
"use strict";
const CLASSES = ['crying_baby','clock_alarm','door_wood_knock','glass_breaking','siren','car_horn','dog','footsteps','thunderstorm','pouring_water'];
const NICE = {crying_baby:'Baby crying',clock_alarm:'Alarm clock',door_wood_knock:'Door knock',glass_breaking:'Glass breaking',siren:'Siren',car_horn:'Car horn',dog:'Dog barking',footsteps:'Footsteps',thunderstorm:'Thunder',pouring_water:'Water running'};
const DEFAULT_TIERS = {siren:'high',glass_breaking:'high',car_horn:'high',crying_baby:'high',clock_alarm:'medium',door_wood_knock:'medium',dog:'medium',footsteps:'low',thunderstorm:'low',pouring_water:'low'};
const $ = id => document.getElementById(id);
let modelReady = false, tiers = JSON.parse(localStorage.getItem('sc_tiers') || 'null') || {...DEFAULT_TIERS};
let events = JSON.parse(localStorage.getItem('sc_log') || '[]');

DSP.loadModel('model.json').then(() => {
  modelReady = true;
  $('btnMic').disabled = false;
  $('micStatus').innerHTML = 'Model ready (79,866 parameters, runs on this device).';
  if (selectedClip) $('btnClassifyClip').disabled = false;
}).catch(e => { $('micStatus').textContent = 'Model failed to load: ' + e; });

$('btnMic').disabled = true;

// ---------- tiers editor ----------
function renderTierEdits() {
  $('tierEdits').innerHTML = CLASSES.map(c =>
    `<label style="margin-right:8px">${NICE[c]} <select data-cls="${c}">${['high','medium','low'].map(t =>
      `<option ${tiers[c]===t?'selected':''}>${t}</option>`).join('')}</select></label>`).join(' ');
  $('tierEdits').querySelectorAll('select').forEach(s => s.onchange = () => {
    tiers[s.dataset.cls] = s.value; localStorage.setItem('sc_tiers', JSON.stringify(tiers));
  });
}
renderTierEdits();

// ---------- bars ----------
function renderBars(el, probs, labels) {
  const top = probs.map((p,i)=>({p,l:labels[i]})).sort((a,b)=>b.p-a.p).slice(0,3);
  el.innerHTML = top.map(t =>
    `<div class="bar-row"><span>${t.l}</span><div class="bar-track"><div class="bar-fill ${t.p<0.5?'dim':''}" style="width:${(t.p*100).toFixed(1)}%"></div></div><span class="mono">${(t.p*100).toFixed(0)}%</span></div>`).join('');
}

// ---------- timeline ----------
function tierBadge(t) { return `<span class="tier ${t}">${t}</span>`; }
function addEvent(label, conf, tier) {
  const now = Date.now();
  const last = events[events.length-1];
  if (last && last.label === label && now - last.tLast < 10000) {
    last.tLast = now; last.count++; last.conf = Math.max(last.conf, conf);
  } else {
    events.push({t0: now, tLast: now, label, conf, tier, count: 1});
    if (events.length > 200) events.shift();
  }
  localStorage.setItem('sc_log', JSON.stringify(events));
  renderTimeline();
}
function renderTimeline() {
  $('evtCount').textContent = events.length + ' event' + (events.length===1?'':'s');
  $('btnExport').disabled = events.length === 0;
  $('timeline').innerHTML = events.slice().reverse().map(e => {
    const d = new Date(e.tLast);
    const hh = String(d.getHours()).padStart(2,'0'), mm = String(d.getMinutes()).padStart(2,'0'), ss = String(d.getSeconds()).padStart(2,'0');
    return `<div class="event ${e.tier}"><span class="t">${hh}:${mm}:${ss}</span><span><b>${e.label}</b> ${tierBadge(e.tier)}</span><span class="meta">${(e.conf*100).toFixed(0)}% · ×${e.count}</span></div>`;
  }).join('');
}
renderTimeline();
$('btnClear').onclick = () => { events = []; localStorage.setItem('sc_log','[]'); renderTimeline(); };
$('btnExport').onclick = () => {
  const rows = [['time','sound','tier','confidence','repeats']].concat(events.map(e =>
    [new Date(e.t0).toISOString(), e.label, e.tier, e.conf.toFixed(3), e.count]));
  const blob = new Blob([rows.map(r=>r.join(',')).join('\n')], {type:'text/csv'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'soundcue-log.csv'; a.click();
};
function flashScreen() {
  const f = $('flash'); f.classList.add('on');
  setTimeout(()=>f.classList.remove('on'), 250);
  setTimeout(()=>f.classList.add('on'), 450);
  setTimeout(()=>f.classList.remove('on'), 700);
}

// ---------- mic ----------
let audioCtx = null, ring = null, ringFill = 0, listening = false, micStream = null;
let lastTick = 0, busy = false;
const RING_SEC = 8;

async function startMic() {
  micStream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation:false, noiseSuppression:false, autoGainControl:false}});
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === 'suspended') await audioCtx.resume();
  const src = audioCtx.createMediaStreamSource(micStream);
  const proc = audioCtx.createScriptProcessor(4096, 1, 1);
  const cap = RING_SEC * audioCtx.sampleRate;
  ring = new Float32Array(cap); ringFill = 0;
  src.connect(proc); proc.connect(audioCtx.destination);
  proc.onaudioprocess = (ev) => {
    const d = ev.inputBuffer.getChannelData(0);
    if (ringFill + d.length > cap) { // shift left
      ring.copyWithin(0, d.length, ringFill); ringFill -= d.length;
    }
    ring.set(d, ringFill); ringFill += d.length;
  };
  listening = true;
  $('btnMic').textContent = 'Stop listening';
  $('micStatus').innerHTML = 'Listening… classifying every 2 s on this device. <b>Nothing is uploaded.</b>';
  lastTick = performance.now();
  requestAnimationFrame(tickLoop);
}
function stopMic() {
  listening = false;
  if (micStream) micStream.getTracks().forEach(t => t.stop());
  $('btnMic').textContent = 'Start listening';
  $('micStatus').textContent = 'Stopped.';
}
$('btnMic').onclick = async () => {
  if (listening) { stopMic(); return; }
  try { await startMic(); } catch(e) { $('micStatus').textContent = 'Microphone unavailable: ' + e.message + ' (needs Chrome/Edge/Firefox on HTTPS or localhost)'; }
};

function tickLoop() {
  if (!listening) return;
  const now = performance.now();
  if (now - lastTick >= 2000 && !busy && modelReady && ringFill >= audioCtx.sampleRate * 4) {
    busy = true; lastTick = now;
    setTimeout(() => { try { classifyLive(); } finally { busy = false; } }, 0);
  }
  drawWave();
  requestAnimationFrame(tickLoop);
}

function latestWindow16k() {
  const n = Math.min(ringFill, audioCtx.sampleRate * 4);
  const seg = new Float32Array(n);
  seg.set(ring.subarray(ringFill - n, ringFill));
  let x = DSP.resampleTo16k(seg, audioCtx.sampleRate);
  if (x.length > DSP.WIN) x = x.subarray(x.length - DSP.WIN);
  const padded = new Float32Array(DSP.WIN);
  padded.set(x, DSP.WIN - x.length);
  return padded;
}

function classifyLive() {
  const x = latestWindow16k();
  const feats = DSP.computeFeatures(x);
  const { logits, embedding } = DSP.forward(feats);
  const probs = DSP.softmax(logits);
  drawSpec(feats);
  const thr = parseFloat($('thresh').value);
  const top = probs.indexOf(Math.max(...probs));
  const conf = probs[top];
  if (conf >= thr) {
    const t = tiers[CLASSES[top]];
    $('verdict').innerHTML = `${NICE[CLASSES[top]]} ${tierBadge(t)}`;
    $('verdict').className = 'verdict';
    addEvent(NICE[CLASSES[top]], conf, t);
    if (t === 'high') flashScreen();
  } else {
    $('verdict').innerHTML = `Unsure — closest is ${NICE[CLASSES[top]]} at ${(conf*100).toFixed(0)}%, below your ${(thr*100).toFixed(0)}% threshold`;
    $('verdict').className = 'verdict unsure';
  }
  renderBars($('bars'), probs, CLASSES.map(c=>NICE[c]));
}

// ---------- canvases ----------
const specCtx = $('spec').getContext('2d');
specCtx.fillStyle = '#0b0e0e'; specCtx.fillRect(0,0,760,150);
function drawSpec(feats) {
  const cols = 50, w = $('spec').width, h = $('spec').height;
  specCtx.drawImage($('spec'), -Math.floor(w/cols), 0);
  const colW = w / cols;
  for (let i = 0; i < cols; i++) {
    const f = DSP.FRAMES - cols + i;
    for (let m = 0; m < DSP.NMEL; m++) {
      const v = feats[m * DSP.FRAMES + f];
      const a = Math.max(0, Math.min(1, (v + 2) / 4));
      specCtx.fillStyle = `rgba(185,230,47,${(a*a*0.95).toFixed(3)})`;
      specCtx.fillRect(i * colW, h - (m+1) * (h / DSP.NMEL), colW + 0.5, h / DSP.NMEL + 0.5);
    }
  }
}
const waveCtx = $('wave').getContext('2d');
function drawWave() {
  const c = $('wave'); waveCtx.fillStyle = '#0b0e0e'; waveCtx.fillRect(0,0,c.width,c.height);
  if (!ring || !ringFill) return;
  const n = Math.min(ringFill, audioCtx.sampleRate * 2);
  waveCtx.strokeStyle = '#b9e62f'; waveCtx.beginPath();
  for (let px = 0; px < c.width; px++) {
    const i = ringFill - n + Math.floor(px / c.width * n);
    const v = ring[i] || 0;
    const y = c.height/2 - v * c.height * 0.9;
    px === 0 ? waveCtx.moveTo(px, y) : waveCtx.lineTo(px, y);
  }
  waveCtx.stroke();
}
$('thresh').oninput = () => $('threshVal').textContent = parseFloat($('thresh').value).toFixed(2);

// ---------- try a clip ----------
let selectedClip = null, clipCache = {};
async function loadClipGrid() {
  const manifest = await (await fetch('manifest.json')).json();
  $('clipgrid').innerHTML = manifest.map((m,i) =>
    `<div class="clip" data-i="${i}"><div class="name">${NICE[m.class]}</div><div class="note">held-out test clip · <span class="mono">${m.file}</span></div></div>`).join('');
  $('clipgrid').querySelectorAll('.clip').forEach(el => el.onclick = () => {
    $('clipgrid').querySelectorAll('.clip').forEach(x=>x.classList.remove('sel'));
    el.classList.add('sel');
    selectedClip = manifest[+el.dataset.i];
    if (modelReady) $('btnClassifyClip').disabled = false;
    $('clipStatus').textContent = 'Selected: ' + NICE[selectedClip.class] + ' — classify it, or double-click to play.';
  });
  $('clipgrid').querySelectorAll('.clip').forEach(el => el.ondblclick = () => {
    const m = manifest[+el.dataset.i];
    new Audio(m.file).play();
  });
}
loadClipGrid();
$('noise').oninput = () => {
  const v = +$('noise').value;
  $('noiseVal').textContent = v === 0 ? 'clean' : `SNR ~${31 - v} dB`;
};
async function decodeClip(url) {
  if (clipCache[url]) return clipCache[url];
  const buf = await (await fetch(url)).arrayBuffer();
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  const ab = await audioCtx.decodeAudioData(buf);
  const pcm = ab.getChannelData(0);
  clipCache[url] = DSP.resampleTo16k(pcm, ab.sampleRate);
  return clipCache[url];
}
function addNoise(x, sliderVal) {
  if (sliderVal === 0) return x;
  let p = 0; for (let i=0;i<x.length;i++) p += x[i]*x[i];
  p = p / x.length + 1e-9;
  const snr = 31 - sliderVal;
  const nrms = Math.sqrt(p / Math.pow(10, snr/10));
  const out = new Float32Array(x.length);
  for (let i=0;i<x.length;i++) out[i] = x[i] + nrms * (Math.random()*2-1) * 1.732; // ~uniform white
  return out;
}
$('btnClassifyClip').onclick = async () => {
  if (!selectedClip) return;
  $('btnClassifyClip').disabled = true;
  $('clipStatus').textContent = 'Classifying…';
  try {
    const raw = selectedClip.file === 'upload' ? clipCache['upload'] : await decodeClip(selectedClip.file);
    let x = addNoise(raw, +$('noise').value);
    const padded = new Float32Array(DSP.WIN);
    padded.set(x.subarray(Math.max(0, x.length - DSP.WIN)), Math.max(0, DSP.WIN - x.length));
    const { logits } = DSP.forward(DSP.computeFeatures(padded));
    const probs = DSP.softmax(logits);
    const top = probs.indexOf(Math.max(...probs));
    const thr = parseFloat($('thresh').value);
    $('clipResult').style.display = 'block';
    $('clipVerdict').innerHTML = probs[top] >= thr ? `${NICE[CLASSES[top]]} ${tierBadge(tiers[CLASSES[top]])}` : `Unsure — below threshold`;
    $('clipVerdict').className = 'verdict' + (probs[top] >= thr ? '' : ' unsure');
    renderBars($('clipBars'), probs, CLASSES.map(c=>NICE[c]));
    $('clipTruth').textContent = (selectedClip.class ? 'Actual class: ' + NICE[selectedClip.class] : 'Your own upload (unknown class)') +
      (+$('noise').value ? ` · with added noise (SNR ~${31- +$('noise').value} dB)` : ' · clean') +
      (selectedClip.class ? (probs[top] >= thr
        ? (CLASSES[top] === selectedClip.class ? ' · detected correctly' : ' · misdetected — the failure modes described in Limits')
        : (CLASSES[top] === selectedClip.class ? ' · top-ranked class matches, but below threshold — no detection' : ' · top-ranked class also mismatches — no detection')) : '');
    $('clipStatus').textContent = '';
  } catch(e) { $('clipStatus').textContent = 'Could not classify: ' + e.message; }
  $('btnClassifyClip').disabled = false;
};
$('wavUp').onchange = async (ev) => {
  const f = ev.target.files[0]; if (!f) return;
  selectedClip = null;
  $('clipgrid').querySelectorAll('.clip').forEach(x=>x.classList.remove('sel'));
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  try {
    const ab = await audioCtx.decodeAudioData(await f.arrayBuffer());
    clipCache['upload'] = DSP.resampleTo16k(ab.getChannelData(0), ab.sampleRate);
    selectedClip = {file: 'upload', class: null};
    if (modelReady) $('btnClassifyClip').disabled = false;
    $('clipStatus').textContent = 'Loaded ' + f.name + ' (' + ab.duration.toFixed(1) + ' s) — classify it.';
  } catch(e) { $('clipStatus').textContent = 'Could not decode that file (WAV works best): ' + e.message; }
};

// ---------- evidence ----------
(async () => {
  const r = await (await fetch('results.json')).json();
  $('metrics').innerHTML = `<table>
    <tr><th>Model</th><th class="num">Validation (80)</th><th class="num">Test (80)</th></tr>
    <tr><td>Baseline: mel mean/std + logistic regression</td><td class="num">${(r.baseline.val*100).toFixed(1)}%</td><td class="num">${(r.baseline.test*100).toFixed(1)}%</td></tr>
    <tr><td><b>CNN</b> (4 conv blocks, ${r.cnn.params.toLocaleString()} params, trained this project)</td><td class="num"><b>${(r.cnn.best_val*100).toFixed(1)}%</b></td><td class="num"><b>${(r.cnn.test*100).toFixed(1)}%</b></td></tr>
  </table>
  <p class="small" style="margin-top:8px">Split: official ESC-50 folds 1–3 train / 4 validation / 5 test — no clip overlaps folds. Seed 7, best checkpoint by validation accuracy. 80-clip test denominator: each cell below is out of 8.</p>`;
  const cm = r.confusion;
  let rows = CLASSES.map((c,i) => {
    const tp = cm[i][i], rowSum = cm[i].reduce((a,b)=>a+b,0);
    let fp = 0; for (let k=0;k<10;k++) fp += cm[k][i];
    fp -= tp;
    const prec = tp+fp ? tp/(tp+fp) : 0, rec = rowSum ? tp/rowSum : 0;
    return `<tr><td>${NICE[c]}</td><td class="num">${(prec*100).toFixed(0)}%</td><td class="num">${(rec*100).toFixed(0)}%</td><td class="num">${tp}/${rowSum}</td></tr>`;
  }).join('');
  $('perclass').innerHTML = `<table><tr><th>Class</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">Correct</th></tr>${rows}</table>`;
  const short = ['baby','alarm','knock','glass','siren','horn','dog','steps','thunder','water'];
  let html = '<table class="cm"><tr><th></th>' + short.map(s=>`<th>${s}</th>`).join('') + '</tr>';
  for (let i=0;i<10;i++) {
    html += `<tr><th>${short[i]}</th>`;
    for (let j=0;j<10;j++) {
      const v = cm[i][j];
      const cls = i===j ? 'diag' : (v>0 ? 'err' : 'off');
      html += `<td class="${cls}">${v||''}</td>`;
    }
    html += '</tr>';
  }
  $('cmatrix').innerHTML = html + '</table>';
})();
