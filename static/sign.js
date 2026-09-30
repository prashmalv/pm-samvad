// Sign -> Speech. MediaPipe runs here in the browser; only landmark points go to the server.
// Frame layout must match app/sign_features.py: 13 pose points, 21 left-hand, 21 right-hand, (x, y) each.
import { FilesetResolver, HandLandmarker, PoseLandmarker } from "./vendor/mediapipe/vision_bundle.mjs";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

const POSE_IDX = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
const N_POSE = 13, N_HAND = 21, N_POINTS = N_POSE + 2 * N_HAND;
const L_SH = 7, R_SH = 8, L_WR = 11, R_WR = 12;       // indices within POSE_IDX
const REST_Y = 1.75;                                  // same "hands at rest" line as training
const SAMPLE_MS = 66;                                 // ~15 fps, like the training data
const LOW_CONF = 0.35;
// Segmentation timing (ms) and hand speeds (shoulder-widths per second).
// A sign starts when a raised hand moves, and ends when the hands are lowered / leave the view,
// OR hold still - webcams rarely see hands drop below REST_Y, so waiting for that alone is slow.
const START_SPEED = 0.6;
const STILL_SPEED = 0.35;
const STILL_MS = 450;
const IDLE_MS = 400;
const MIN_ACTIVE_MS = 330;
const MAX_SIGN_MS = 6000;
const AUTO_SENTENCE_MS = 1800;                        // pause after the last sign before the sentence is made
const HAND_EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];

const state = { signs: [], english: "", running: false, known: [], model: null };
let hand = null, pose = null, lastTs = 0, stream = null, rafId = 0, lastSample = 0;

/* ---------------- theme (same as the other page) ---------------- */
const root = document.documentElement;
try { const t = localStorage.getItem("theme"); if (t) root.dataset.theme = t; } catch {}
if (!root.dataset.theme) root.dataset.theme = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
$("#themeBtn").onclick = () => {
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch {}
};

function toast(msg, warn = false) { const t = $("#toast"); t.textContent = msg; t.classList.toggle("warn", warn); t.classList.remove("hidden"); }
function hideToast() { $("#toast").classList.add("hidden"); }
async function errorText(res) {
  try { const j = await res.json(); return typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); }
  catch { return `${res.status} ${res.statusText}`; }
}

/* ---------------- MediaPipe setup ---------------- */
async function loadTrackers() {
  if (hand) return;
  $("#camLoading").classList.remove("hidden");
  const vision = await FilesetResolver.forVisionTasks("./vendor/mediapipe/wasm");
  const make = async (delegate) => {
    hand = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: "./models/hand_landmarker.task", delegate },
      runningMode: "VIDEO", numHands: 2,
      minHandDetectionConfidence: 0.4, minHandPresenceConfidence: 0.4, minTrackingConfidence: 0.4,
    });
    pose = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: "./models/pose_landmarker_lite.task", delegate },
      runningMode: "VIDEO", numPoses: 1,
    });
  };
  try { await make("GPU"); } catch { await make("CPU"); }
  $("#camLoading").classList.add("hidden");
}

function nextTs(desired) { lastTs = Math.max(lastTs + 1, Math.round(desired)); return lastTs; }

/* ---------------- landmarks -> frame (mirrors app/sign_features.build_frame) ---------------- */
function buildFrame(poseRes, handRes) {
  const f = new Array(N_POINTS * 2).fill(null);
  let pts = null;
  if (poseRes.landmarks?.length) {
    const lm = poseRes.landmarks[0];
    pts = POSE_IDX.map((j) => [lm[j].x, lm[j].y]);
    pts.forEach(([x, y], i) => { f[2 * i] = x; f[2 * i + 1] = y; });
  }
  const hands = (handRes.landmarks || []).map((h) => h.map((p) => [p.x, p.y]));
  let left = null, right = null;
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  if (hands.length && pts) {
    const lw = pts[L_WR], rw = pts[R_WR];
    const score = (h) => dist(h[0], lw) - dist(h[0], rw);
    if (hands.length === 1) { if (score(hands[0]) < 0) left = hands[0]; else right = hands[0]; }
    else { const s = [...hands].sort((a, b) => score(a) - score(b)); left = s[0]; right = s[s.length - 1]; }
  } else if (hands.length) {
    const s = [...hands].sort((a, b) => b[0][0] - a[0][0]);   // camera faces signer: image-right = signer's left
    left = s[0]; right = s[1] || null;
  }
  const put = (h, start) => h && h.forEach(([x, y], i) => { f[2 * (start + i)] = x; f[2 * (start + i) + 1] = y; });
  put(left, N_POSE); put(right, N_POSE + N_HAND);
  return f;
}

function isActive(f) {
  // a sign is happening while at least one hand is raised above the rest line
  const hasPose = f[2 * L_SH] !== null && f[2 * R_SH] !== null;
  for (const start of [N_POSE, N_POSE + N_HAND]) {
    if (f[2 * start] === null) continue;
    if (!hasPose) return true;
    const cy = (f[2 * L_SH + 1] + f[2 * R_SH + 1]) / 2;
    const scale = Math.hypot(f[2 * L_SH] - f[2 * R_SH], f[2 * L_SH + 1] - f[2 * R_SH + 1]) || 0.25;
    if ((f[2 * start + 1] - cy) / scale < REST_Y) return true;
  }
  return false;
}

function bodyScale(f) {
  if (f[2 * L_SH] === null || f[2 * R_SH] === null) return 0.25;
  return Math.hypot(f[2 * L_SH] - f[2 * R_SH], f[2 * L_SH + 1] - f[2 * R_SH + 1]) || 0.25;
}

// Speed of the faster hand (mean of its 21 points, so finger-only movement counts) and its
// vertical share (positive = moving down the image), in shoulder-widths per second.
function handMotion(prev, f, dtMs) {
  let speed = 0, vy = 0;
  if (!prev || dtMs <= 0) return { speed, vy };
  const k = bodyScale(f) * (dtMs / 1000);
  for (const start of [N_POSE, N_POSE + N_HAND]) {
    if (f[2 * start] === null || prev[2 * start] === null) continue;
    let d = 0, dy = 0;
    for (let i = 0; i < N_HAND; i++) {
      const j = 2 * (start + i);
      d += Math.hypot(f[j] - prev[j], f[j + 1] - prev[j + 1]); dy += f[j + 1] - prev[j + 1];
    }
    const s = d / N_HAND / k;
    if (s > speed) { speed = s; vy = dy / N_HAND / k; }
  }
  return { speed, vy };
}

/* ---------------- segmentation: one sign = hand raised + moving ... lowered or held still ---------------- */
class Segmenter {
  constructor(onSign, onStart) { this.onSign = onSign; this.onStart = onStart; this.reset(); }
  reset() { this.pre = []; this.rec = null; this.prev = null; this.prevT = 0; this.speed = 0; this.startRun = 0; this.held = false; }
  push(f, t) {
    const dt = this.prev ? t - this.prevT : 0;
    const m = handMotion(this.prev, f, dt);
    this.speed = this.prev ? 0.5 * this.speed + 0.5 * m.speed : 0;  // light smoothing against tracker jitter
    this.prev = f; this.prevT = t;
    const up = isActive(f);
    if (!this.rec) {
      this.pre.push(f); if (this.pre.length > 4) this.pre.shift();
      if (!up) this.held = false;
      // Right after a sign ended on a hold, lowering the hands must not count as the next sign.
      const lowering = this.held && m.vy > 0.7 * m.speed;
      this.startRun = up && this.speed > START_SPEED && !lowering ? this.startRun + 1 : 0;
      if (this.startRun >= 2) {
        this.rec = [...this.pre]; this.t0 = t; this.activeMs = 0;
        this.idleAt = this.stillAt = null;
        this.onStart?.(); setStatus("recording", "Signing…");
      }
      return;
    }
    this.rec.push(f);
    const n = this.rec.length - 1;
    if (up) { this.activeMs += dt; this.idleAt = null; }
    else if (!this.idleAt) this.idleAt = { t, n };
    if (up && this.speed < STILL_SPEED) { if (!this.stillAt) this.stillAt = { t, n }; }
    else this.stillAt = null;
    $("#recFill").style.width = `${Math.min(100, ((t - this.t0) / MAX_SIGN_MS) * 100)}%`;
    if (this.idleAt && t - this.idleAt.t >= IDLE_MS) this.finish(this.idleAt.n + 2);
    else if (this.stillAt && t - this.stillAt.t >= STILL_MS && this.activeMs >= MIN_ACTIVE_MS) { this.finish(this.stillAt.n + 3); this.held = true; }
    else if (t - this.t0 >= MAX_SIGN_MS) this.finish(this.rec.length);
  }
  finish(end = this.rec.length) {
    const frames = this.rec.slice(0, end);
    const enough = this.activeMs >= MIN_ACTIVE_MS;
    this.rec = null; this.pre = []; this.startRun = 0;
    $("#recFill").style.width = "0";
    if (enough) this.onSign(frames); else setStatus("ready", "Show a sign");
  }
}

function setStatus(kind, text) {
  const s = $("#signStatus"); s.className = `sign-status ${kind}`; $("#signStatusText").textContent = text;
  $("#recBar").classList.toggle("hidden", kind !== "recording");
}

/* ---------------- drawing ---------------- */
const canvas = $("#overlay"), ctx = canvas.getContext("2d");
function draw(poseRes, handRes, w, h) {
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  ctx.clearRect(0, 0, w, h);
  if (!$("#showSkeleton").checked) return;
  ctx.lineCap = "round";
  if (poseRes.landmarks?.length) {
    const lm = poseRes.landmarks[0];
    ctx.strokeStyle = "rgba(255,255,255,.55)"; ctx.lineWidth = 3;
    for (const [a, b] of [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]) {
      ctx.beginPath(); ctx.moveTo(lm[a].x * w, lm[a].y * h); ctx.lineTo(lm[b].x * w, lm[b].y * h); ctx.stroke();
    }
  }
  for (const hl of handRes.landmarks || []) {
    ctx.strokeStyle = "#8b73ff"; ctx.lineWidth = 3;
    for (const [a, b] of HAND_EDGES) { ctx.beginPath(); ctx.moveTo(hl[a].x * w, hl[a].y * h); ctx.lineTo(hl[b].x * w, hl[b].y * h); ctx.stroke(); }
    ctx.fillStyle = "#ec5fa5";
    for (const p of hl) { ctx.beginPath(); ctx.arc(p.x * w, p.y * h, 3.2, 0, Math.PI * 2); ctx.fill(); }
  }
}

/* ---------------- live camera ---------------- */
const cam = $("#cam");
const liveSeg = new Segmenter((frames) => recognize(frames), () => clearTimeout(autoTimer));

$("#startCam").onclick = async () => {
  hideToast();
  try {
    await loadTrackers();
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false });
  } catch (e) {
    $("#camLoading").classList.add("hidden");
    toast(`Camera unavailable: ${e.message}. Open the app on localhost and allow camera access.`);
    return;
  }
  cam.srcObject = stream; cam.classList.add("mirror"); await cam.play();
  $("#camStage").classList.add("mirrored");
  $("#camEmpty").classList.add("hidden"); $("#signStatus").classList.remove("hidden");
  $("#stopCam").disabled = false;
  state.running = true; liveSeg.reset(); setStatus("ready", "Show a sign");
  loop();
};

function loop() {
  if (!state.running) return;
  const now = performance.now();
  if (cam.readyState >= 2 && now - lastSample >= SAMPLE_MS) {
    lastSample = now;
    const ts = nextTs(now);
    const pr = pose.detectForVideo(cam, ts), hr = hand.detectForVideo(cam, ts);
    draw(pr, hr, cam.videoWidth, cam.videoHeight);
    liveSeg.push(buildFrame(pr, hr), now);
  }
  rafId = requestAnimationFrame(loop);
}

function stopCamera() {
  state.running = false; cancelAnimationFrame(rafId);
  stream?.getTracks().forEach((t) => t.stop()); stream = null;
  cam.srcObject = null; ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("#stopCam").disabled = true; $("#signStatus").classList.add("hidden"); $("#recBar").classList.add("hidden");
  $("#camEmpty").classList.remove("hidden");
}
$("#stopCam").onclick = stopCamera;

/* ---------------- uploaded video ---------------- */
$("#videoFile").addEventListener("change", async (e) => {
  const file = e.target.files[0]; e.target.value = "";
  if (!file) return;
  hideToast();
  if (state.running) stopCamera();
  try { await loadTrackers(); } catch (err) { toast(`Could not load hand tracking: ${err.message}`); return; }
  const v = cam;
  $("#camStage").classList.remove("mirrored");
  v.srcObject = null; v.src = URL.createObjectURL(file); v.muted = true;
  await new Promise((r, j) => { v.onloadedmetadata = r; v.onerror = () => j(new Error("unsupported video format")); }).catch((err) => { toast(err.message); });
  if (!v.duration) return;
  $("#camEmpty").classList.add("hidden"); $("#signStatus").classList.remove("hidden");
  $("#camLoading").classList.remove("hidden");
  const found = [];
  const seg = new Segmenter((frames) => found.push(frames));
  const step = SAMPLE_MS / 1000;
  for (let t = 0; t < v.duration; t += step) {
    v.currentTime = t;
    await new Promise((r) => v.addEventListener("seeked", r, { once: true }));
    const ts = nextTs(lastTs + SAMPLE_MS);
    const pr = pose.detectForVideo(v, ts), hr = hand.detectForVideo(v, ts);
    draw(pr, hr, v.videoWidth, v.videoHeight);
    seg.push(buildFrame(pr, hr), t * 1000);
    $("#camLoadingText").textContent = `Reading signs… ${Math.round((t / v.duration) * 100)}%`;
  }
  // flush a sign still in progress at the end of the clip
  if (seg.rec) seg.finish(seg.idleAt ? seg.idleAt.n + 2 : seg.rec.length);
  $("#camLoading").classList.add("hidden"); $("#camLoadingText").textContent = "Loading hand tracking…";
  if (!found.length) {
    setStatus("ready", "No signs found");
    toast("No signs found in this video. Make sure the signer's hands and shoulders are visible.", true);
    return;
  }
  setStatus("ready", `${found.length} sign(s) found`);
  await Promise.all(found.map((frames) => recognize(frames, false)));
  if ($("#autoSentence").checked) makeSentence();
});

/* ---------------- recognition ---------------- */
let autoTimer = 0;
function scheduleSentence() {
  clearTimeout(autoTimer);
  if ($("#autoSentence").checked) autoTimer = setTimeout(() => { if (!liveSeg.rec) makeSentence(); }, AUTO_SENTENCE_MS);
}

async function recognize(frames, autoSentence = true) {
  const item = { frames, candidates: null, pick: 0, pending: true };
  state.signs.push(item); renderSigns();
  setStatus("thinking", "Recognising…");
  try {
    const res = await fetch("/recognize", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ frames, top_k: 5 }) });
    if (!res.ok) throw new Error(await errorText(res));
    item.candidates = (await res.json()).candidates;
  } catch (e) {
    state.signs.splice(state.signs.indexOf(item), 1);
    toast(e.message);
  }
  item.pending = false;
  renderSigns();
  if (autoSentence && item.candidates && !state.signs.some((s) => s.pending)) scheduleSentence();
  if (state.running) setStatus("ready", "Show a sign");
  else if (!state.signs.some((s) => s.pending)) setStatus("ready", `Done: ${currentGlosses().length} sign(s)`);
}

function currentGlosses() { return state.signs.filter((s) => s.candidates).map((s) => s.candidates[s.pick].gloss); }

function renderSigns() {
  const box = $("#signChips"); box.innerHTML = "";
  state.signs.forEach((s, i) => {
    const b = document.createElement("button");
    if (s.pending) { b.className = "sign-chip pending"; b.innerHTML = "<b>…</b><small>recognising</small>"; box.appendChild(b); return; }
    const c = s.candidates[s.pick];
    b.className = "sign-chip" + (c.confidence < LOW_CONF ? " low" : "");
    b.innerHTML = `<b>${c.gloss}</b><small>${Math.round(c.confidence * 100)}%</small><i class="conf" style="transform:scaleX(${c.confidence})"></i>`;
    b.title = c.confidence < LOW_CONF ? "Not sure. Click to see other guesses" : "Click to see other guesses";
    b.onclick = (e) => openAlternatives(i, e.currentTarget);
    box.appendChild(b);
  });
  if (!state.signs.length) box.innerHTML = '<span class="placeholder">Signs appear here as you make them.</span>';
  const n = currentGlosses().length;
  $("#undoBtn").disabled = $("#clearBtn").disabled = !state.signs.length;
  $("#sentenceBtn").disabled = !n;
}

const pop = $("#altPop");
function openAlternatives(i, anchor) {
  const s = state.signs[i];
  pop.innerHTML = "";
  s.candidates.forEach((c, k) => {
    const b = document.createElement("button");
    b.innerHTML = `${c.gloss}<span>${Math.round(c.confidence * 100)}%</span>`;
    if (k === s.pick) b.style.color = "var(--accent)";
    b.onclick = () => { s.pick = k; closeAlternatives(); renderSigns(); };
    pop.appendChild(b);
  });
  const rm = document.createElement("button"); rm.className = "remove"; rm.textContent = "Remove this sign";
  rm.onclick = () => { state.signs.splice(i, 1); closeAlternatives(); renderSigns(); };
  pop.appendChild(rm);
  pop.classList.remove("hidden");
  const r = anchor.getBoundingClientRect();
  pop.style.left = `${Math.min(r.left, innerWidth - pop.offsetWidth - 12)}px`;
  pop.style.top = `${Math.min(r.bottom + 6, innerHeight - pop.offsetHeight - 12)}px`;
}
function closeAlternatives() { pop.classList.add("hidden"); }
document.addEventListener("click", (e) => { if (!pop.contains(e.target) && !e.target.closest(".sign-chip")) closeAlternatives(); });

$("#undoBtn").onclick = () => { state.signs.pop(); renderSigns(); };
$("#clearBtn").onclick = () => { clearTimeout(autoTimer); translated = ""; state.signs = []; renderSigns(); $("#english").innerHTML = '<span class="placeholder">—</span>'; $("#speakBtn").disabled = true; };

/* ---------------- English + speech ---------------- */
let translated = "";  // gloss string of the sentence on screen, so an unchanged list isn't sent again
$("#sentenceBtn").onclick = () => makeSentence(true);

async function makeSentence(force = false) {
  clearTimeout(autoTimer);
  const glosses = currentGlosses(); if (!glosses.length) return;
  const key = glosses.join(" ");
  if (!force && key === translated) return;
  const btn = $("#sentenceBtn"); btn.disabled = true; hideToast();
  $("#english").innerHTML = '<span class="placeholder">Translating…</span>';
  try {
    const res = await fetch("/to-english", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ glosses }) });
    if (!res.ok) throw new Error(await errorText(res));
    state.english = (await res.json()).english;
    translated = key;
    $("#english").textContent = state.english;
    $("#speakBtn").disabled = !state.english;
    if ($("#autoSpeak").checked) speak();
  } catch (e) { toast(e.message); $("#english").innerHTML = '<span class="placeholder">—</span>'; }
  finally { btn.disabled = !currentGlosses().length; }
}

function speak() {
  if (!state.english || !("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(state.english);
  const voices = speechSynthesis.getVoices();
  u.voice = voices.find((v) => v.lang === "en-IN") || voices.find((v) => v.lang.startsWith("en")) || null;
  u.rate = 0.95;
  speechSynthesis.speak(u);
}
$("#speakBtn").onclick = speak;
if ("speechSynthesis" in window) speechSynthesis.getVoices(); // warm the voice list

/* ---------------- model info + known signs ---------------- */
async function loadModelInfo() {
  const info = $("#modelInfo");
  try {
    const d = await (await fetch("/sign-model")).json();
    state.model = d; state.known = d.signs || [];
    $("#knownCount").textContent = state.known.length; $("#knownCount2").textContent = state.known.length;
    renderKnown("");
    if (!d.available) {
      info.className = "model-info warn";
      info.innerHTML = "<b>No recognition model yet.</b> Extract INCLUDE landmarks and train it: <code>python scripts/include_extract.py</code> then <code>python scripts/train_sign_model.py</code>.";
      return;
    }
    const r = d.report || {};
    info.className = "model-info";
    info.innerHTML = `Recognises <b>${r.signs ?? state.known.length} signs</b>. On INCLUDE's held-out test videos it picks the right sign first <b>${Math.round((r.test_top1 ?? 0) * 100)}%</b> of the time, and within its top 5 guesses <b>${Math.round((r.test_top5 ?? 0) * 100)}%</b>. Accuracy on your own webcam will usually be lower.`;
  } catch { info.textContent = "Could not load model info."; }
}
function renderKnown(q) {
  const grid = $("#knownGrid"); grid.innerHTML = "";
  const items = state.known.filter((g) => g.includes(q.trim().toUpperCase()));
  for (const g of items) { const b = document.createElement("button"); b.textContent = g; b.title = g; grid.appendChild(b); }
  if (!items.length) grid.innerHTML = '<span class="muted">No matching signs.</span>';
}
$("#knownSearch").addEventListener("input", (e) => renderKnown(e.target.value));
function openDrawer() { $("#drawer").classList.add("open"); $("#drawer").setAttribute("aria-hidden", "false"); $("#scrim").classList.remove("hidden"); }
function closeDrawer() { $("#drawer").classList.remove("open"); $("#drawer").setAttribute("aria-hidden", "true"); $("#scrim").classList.add("hidden"); }
$("#knownBtn").onclick = openDrawer; $("#closeDrawer").onclick = closeDrawer; $("#scrim").onclick = closeDrawer;
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeDrawer(); closeAlternatives(); } });

loadModelInfo();
renderSigns();
