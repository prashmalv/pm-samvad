// Speak to Sign — vanilla JS frontend. Pipeline: /transcribe -> /gloss -> /render (Human) or /avatar-motion (3D Avatar)
import { AvatarPlayer } from "./avatar.js";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

// Used until GET /topics answers (or if it fails).
const FALLBACK_EXAMPLES = [
  "Hello, what is your name?",
  "I don't understand, can you help me?",
  "Where is your mother?",
  "Good morning, how are you?",
];

const TOPIC_ICONS = {
  casual: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/><path d="M8.5 12h.01M12 12h.01M15.5 12h.01"/></svg>',
  hotel: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 19V6M3 13h18v6M21 19v-4a3 3 0 0 0-3-3h-7v1"/><circle cx="7" cy="10" r="2"/></svg>',
  hospital: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="4"/><path d="M12 8v8M8 12h8"/></svg>',
};

const state = {
  tab: "record",
  audio: null,          // { blob, name }
  segments: [],         // [{gloss, start, end}]
  videoUrl: null,
  busy: false,
  vocab: [],
  alphabet: [],
  fingerspelling: false,
  signer: "human",      // "human" (stitched real clips) | "avatar" (3D, rendered in the browser)
  last: null,           // { tokens, text } of the most recent translation, for switching signer
  topics: [],           // from GET /topics
  topic: null,          // chosen topic id: casual | hotel | hospital
};
try { if (localStorage.getItem("signer") === "avatar") state.signer = "avatar"; } catch {}
try { state.topic = localStorage.getItem("topic"); } catch {}

/* ---------------- theme ---------------- */
const root = document.documentElement;
try { const t = localStorage.getItem("theme"); if (t) root.dataset.theme = t; } catch {}
if (!root.dataset.theme) root.dataset.theme = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
$("#themeBtn").onclick = () => {
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch {}
  if (recorder?.state !== "recording") drawIdleWave();
};

/* ---------------- tabs ---------------- */
$$(".tab").forEach((btn) => btn.addEventListener("click", () => {
  state.tab = btn.dataset.tab;
  $$(".tab").forEach((b) => b.classList.toggle("active", b === btn));
  $$(".tab-body").forEach((b) => b.classList.toggle("hidden", b.dataset.body !== state.tab));
  $("#clipReady").classList.toggle("hidden", !state.audio || state.tab === "type");
  refreshCta();
}));

function refreshCta() {
  const ready = state.tab === "type" ? $("#typeInput").value.trim().length > 0 : !!state.audio;
  $("#translateBtn").disabled = state.busy || !ready || recorder?.state === "recording";
}

/* ---------------- topics ---------------- */
const currentTopic = () => state.topics.find((t) => t.id === state.topic) || null;

function renderExamples() {
  const t = currentTopic();
  $("#examplesHead").textContent = t ? `${t.label} phrases` : "Try a phrase";
  const box = $("#examples"); box.innerHTML = "";
  for (const ex of t?.examples || FALLBACK_EXAMPLES) {
    const b = document.createElement("button");
    b.textContent = ex;
    b.onclick = () => { $("#typeInput").value = ex; refreshCta(); run(); };
    box.appendChild(b);
  }
}

function renderTopicButton() {
  const t = currentTopic();
  $("#topicBtnLabel").textContent = t ? t.label : "Choose";
  $("#topicBtnIcon").innerHTML = t ? TOPIC_ICONS[t.id] || "" : "";
}

function renderTopicCards() {
  const box = $("#topicCards"); box.innerHTML = "";
  for (const t of state.topics) {
    const b = document.createElement("button");
    b.className = "topic-card" + (t.id === state.topic ? " on" : "");
    b.innerHTML = `<span class="topic-ico">${TOPIC_ICONS[t.id] || ""}</span><b></b><span class="topic-desc"></span>` +
      `<span class="topic-cov"><i style="transform:scaleX(${t.covered / t.total})"></i></span><small></small>`;
    $("b", b).textContent = t.label;
    $(".topic-desc", b).textContent = t.description;
    $("small", b).textContent = `${t.covered} of ${t.total} topic signs ready`;
    b.onclick = () => chooseTopic(t.id);
    box.appendChild(b);
  }
}

function chooseTopic(id) {
  state.topic = id;
  try { localStorage.setItem("topic", id); } catch {}
  renderTopicButton(); renderExamples(); renderTopicCards(); renderLibrary($("#libSearch").value);
  closeTopicModal();
}

function openTopicModal() {
  renderTopicCards(); $("#topicModal").classList.remove("hidden");
  ($(".topic-card.on") || $(".topic-card"))?.focus();
}
function closeTopicModal() { if (currentTopic()) $("#topicModal").classList.add("hidden"); }  // a topic must be picked first
$("#topicBtn").onclick = openTopicModal;
$("#topicModal").addEventListener("click", (e) => { if (e.target.id === "topicModal") closeTopicModal(); });

async function loadTopics() {
  try {
    const res = await fetch("/topics");
    if (!res.ok) throw new Error(await errorText(res));
    state.topics = (await res.json()).topics;
  } catch { state.topics = []; }
  if (!state.topics.length) { renderExamples(); $("#topicBtn").hidden = true; return; }  // server without topics: work as before
  if (!currentTopic()) { state.topic = null; openTopicModal(); }
  renderTopicButton(); renderExamples(); renderLibrary($("#libSearch").value);
}

$("#typeInput").addEventListener("input", refreshCta);
$("#typeInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!$("#translateBtn").disabled) run(); }
});

/* ---------------- audio selection ---------------- */
function setAudio(blob, name) {
  state.audio = { blob, name };
  $("#clipName").textContent = name;
  $("#audioPreview").src = URL.createObjectURL(blob);
  $("#clipReady").classList.remove("hidden");
  refreshCta();
}
$("#clearAudio").onclick = () => {
  state.audio = null;
  $("#clipReady").classList.add("hidden");
  $("#audioPreview").removeAttribute("src");
  $("#fileInput").value = "";
  refreshCta();
};

/* upload + drag/drop */
const dz = $("#dropzone");
$("#fileInput").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) setAudio(f, f.name); });
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) setAudio(f, f.name); });

/* ---------------- recorder with live waveform ---------------- */
let recorder = null, chunks = [], audioCtx = null, analyser = null, rafWave = 0, recStart = 0, timerId = 0;
const canvas = $("#wave"), ctx2d = canvas.getContext("2d");

function drawIdleWave() {
  const w = canvas.width, h = canvas.height, bars = 48, gap = 4, bw = (w - gap * (bars - 1)) / bars;
  ctx2d.clearRect(0, 0, w, h);
  ctx2d.fillStyle = getComputedStyle(root).getPropertyValue("--line");
  for (let i = 0; i < bars; i++) roundBar(i * (bw + gap), h / 2 - 2, bw, 4);
}
function roundBar(x, y, w, h) { ctx2d.beginPath(); ctx2d.roundRect(x, y, w, h, w / 2); ctx2d.fill(); }
function drawLiveWave() {
  const w = canvas.width, h = canvas.height, bars = 48, gap = 4, bw = (w - gap * (bars - 1)) / bars;
  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(data);
  ctx2d.clearRect(0, 0, w, h);
  const g = ctx2d.createLinearGradient(0, 0, w, 0);
  g.addColorStop(0, getComputedStyle(root).getPropertyValue("--accent"));
  g.addColorStop(1, getComputedStyle(root).getPropertyValue("--accent-2"));
  ctx2d.fillStyle = g;
  for (let i = 0; i < bars; i++) {
    const v = data[Math.floor((i / bars) * data.length * 0.7)] / 255;
    const bh = Math.max(4, v * h);
    roundBar(i * (bw + gap), (h - bh) / 2, bw, bh);
  }
  rafWave = requestAnimationFrame(drawLiveWave);
}
drawIdleWave();

$("#micBtn").onclick = async () => {
  if (recorder?.state === "recording") { recorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioCtx = new AudioContext();
    analyser = audioCtx.createAnalyser(); analyser.fftSize = 256;
    audioCtx.createMediaStreamSource(stream).connect(analyser);
    recorder = new MediaRecorder(stream); chunks = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      cancelAnimationFrame(rafWave); clearInterval(timerId); audioCtx.close(); drawIdleWave();
      $("#micBtn").classList.remove("recording");
      $("#recHint").textContent = "Recorded. Tap the mic to record again";
      const type = recorder.mimeType || "audio/webm";
      const ext = type.includes("ogg") ? "ogg" : type.includes("mp4") ? "m4a" : "webm";
      setAudio(new Blob(chunks, { type }), `recording.${ext}`);
      if (chunks.length) run(); // auto-translate after stopping
    };
    recorder.start();
    recStart = Date.now();
    timerId = setInterval(() => {
      const s = Math.floor((Date.now() - recStart) / 1000);
      $("#recTimer").textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
    }, 250);
    $("#recTimer").textContent = "00:00";
    $("#micBtn").classList.add("recording");
    $("#recHint").textContent = "Listening… tap to stop";
    hideToast(); refreshCta(); drawLiveWave();
  } catch (err) {
    showToast("Microphone unavailable: " + err.message + " (open the app on localhost or https)");
  }
};

/* ---------------- stepper + toast ---------------- */
function step(name, status, ms) {
  const li = $(`#stepper li[data-step="${name}"]`);
  li.className = status || "";
  $("time", li).textContent = ms != null ? (ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`) : "";
}
function resetSteps() { ["listen", "gloss", "sign"].forEach((s) => step(s, "")); }
function showToast(msg, warn = false) { const t = $("#toast"); t.textContent = msg; t.classList.toggle("warn", warn); t.classList.remove("hidden"); }
function hideToast() { $("#toast").classList.add("hidden"); }
function loading(on, text = "Working…") { $("#loading").classList.toggle("hidden", !on); $("#loadingText").textContent = text; }

async function errorText(res) {
  try {
    const j = await res.json();
    return typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
  } catch { return `${res.status} ${res.statusText}`; }
}
async function timed(name, fn) {
  step(name, "active");
  const t0 = performance.now();
  try { const out = await fn(); step(name, "done", Math.round(performance.now() - t0)); return out; }
  catch (e) { step(name, "error"); throw e; }
}

/* ---------------- pipeline ---------------- */
$("#translateBtn").onclick = run;

async function run() {
  if (state.busy) return;
  state.busy = true; refreshCta(); hideToast(); resetSteps();
  $("#bigPlay").classList.add("hidden");
  try {
    let text;
    if (state.tab === "type") {
      text = $("#typeInput").value.trim();
      step("listen", "skipped");
    } else {
      loading(true, "Listening to your audio…");
      text = await timed("listen", async () => {
        const fd = new FormData(); fd.append("audio", state.audio.blob, state.audio.name);
        if (state.topic) fd.append("topic", state.topic);
        const res = await fetch("/transcribe", { method: "POST", body: fd });
        if (!res.ok) throw new Error(await errorText(res));
        return (await res.json()).text;
      });
      if (!text) throw new Error("No speech detected in the audio. Try speaking a little louder or longer.");
    }
    renderTranscript(text);

    loading(true, "Translating to ISL gloss…");
    const tokens = await timed("gloss", async () => {
      const res = await fetch("/gloss", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, topic: state.topic || undefined }) });
      if (!res.ok) throw new Error(await errorText(res));
      return (await res.json()).tokens;
    });
    renderChips(tokens, []);
    if (!tokens.length) throw new Error("The translation came back empty.");

    state.last = { tokens, text };
    await signTokens(tokens, text);
  } catch (err) {
    showToast(err.message);
  } finally {
    loading(false);
    state.busy = false; refreshCta();
  }
}

async function signTokens(tokens, text) {
  const avatarMode = state.signer === "avatar";
  loading(true, avatarMode ? "Animating the avatar…" : "Signing…");
  if (avatarMode) {
    const seq = await timed("sign", async () => {
      const res = await fetch("/avatar-motion", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ tokens }) });
      if (res.status === 422) { const j = await res.json(); renderChips(tokens, j.missing || tokens); throw new Error(j.detail); }
      if (!res.ok) throw new Error(await errorText(res));
      return res.json();
    });
    renderChips(tokens, seq.missing || []);
    loadAvatar(seq);
    if (seq.missing?.length) showToast(`No sign for: ${seq.missing.join(", ")}. Skipped.`, true);
    return;
  }
  const { blob, info } = await timed("sign", async () => {
    const res = await fetch("/render", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ tokens, transcript: text }) });
    if (res.status === 422) {
      const j = await res.json();
      renderChips(tokens, j.missing || tokens);
      throw new Error(j.detail);
    }
    if (!res.ok) throw new Error(await errorText(res));
    return { blob: await res.blob(), info: JSON.parse(decodeURIComponent(res.headers.get("X-ISL-Info") || "%7B%7D")) };
  });
  renderChips(tokens, info.missing || [], info.spelled || []);
  loadVideo(blob, info.segments || []);
  if (info.missing?.length) {
    const hint = state.fingerspelling ? "" : " Add A–Z letter clips to fingerspell them instead.";
    showToast(`No sign clip for: ${info.missing.join(", ")}. Skipped.${hint}`, true);
  }
}

function renderTranscript(text) {
  $("#transcript").textContent = text;
  $("#transcriptMeta").textContent = `${text.split(/\s+/).filter(Boolean).length} words`;
}

const spellOut = (word) => [...word.replace(/[^A-Z0-9]/g, "")].join("·");

function renderChips(tokens, missing, spelled = []) {
  const miss = new Set(missing), spell = new Set(spelled);
  const box = $("#chips"); box.innerHTML = "";
  let segIdx = 0;
  tokens.forEach((t, i) => {
    const b = document.createElement("button");
    b.className = "chip" + (miss.has(t) ? " missing" : spell.has(t) ? " spelled" : "");
    b.textContent = spell.has(t) ? spellOut(t) : t;
    b.style.animationDelay = `${i * 40}ms`;
    if (miss.has(t)) b.title = "No sign clip for this word, skipped";
    else {
      b.dataset.seg = segIdx++;
      b.title = spell.has(t) ? `Fingerspelled: ${t}` : "Jump to this sign";
      b.onclick = () => seekSeg(+b.dataset.seg);
    }
    box.appendChild(b);
  });
  if (!tokens.length) box.innerHTML = '<span class="placeholder">—</span>';
}

/* ---------------- player ---------------- */
const video = $("#video");
let avatar = null;      // created on first use (WebGL)
let media = video;      // whichever is on stage

function getAvatar() {
  if (!avatar) {
    avatar = new AvatarPlayer($("#avatarCanvas")); wireMedia(avatar);
    if (new URLSearchParams(location.search).has("debug")) window.__avatar = avatar; // for testing only
  }
  return avatar;
}

function showOnStage(m) {
  if (media !== m && !media.paused) media.pause();
  media = m;
  const isAvatar = m !== video;
  video.hidden = isAvatar;
  $("#avatarCanvas").hidden = !isAvatar;
  $("#resetView").hidden = !isAvatar;
  if (isAvatar) avatar.resize();
  m.playbackRate = +$(".speed .on").dataset.speed;
  m.loop = $("#loopBtn").getAttribute("aria-pressed") === "true";
}

function loadVideo(blob, segments) {
  if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
  state.videoUrl = URL.createObjectURL(blob);
  state.segments = segments;
  video.src = state.videoUrl;
  showOnStage(video);
  $("#emptyState").classList.add("hidden");
  $("#playBtn").disabled = false;
  const dl = $("#downloadBtn"); dl.href = state.videoUrl; dl.download = "isl-sign.mp4"; dl.classList.remove("disabled");
  buildTimeline();
  video.play().catch(() => $("#bigPlay").classList.remove("hidden"));
}

function loadAvatar(seq) {
  const a = getAvatar();
  a.load(seq);
  state.segments = seq.segments;
  showOnStage(a);
  $("#emptyState").classList.add("hidden");
  $("#playBtn").disabled = false;
  const dl = $("#downloadBtn"); dl.removeAttribute("href"); dl.classList.remove("disabled");
  buildTimeline();
  a.play();
}

function buildTimeline() {
  const tl = $("#timeline"); tl.innerHTML = "";
  const total = state.segments.at(-1)?.end || 1;
  state.segments.forEach((s, i) => {
    const b = document.createElement("button");
    b.className = "tl-seg" + (s.spelled ? " spelled" : "");
    b.style.flex = `${(s.end - s.start) / total} 1 0`;
    b.innerHTML = `<div class="fill"></div><span>${s.spelled ? spellOut(s.gloss) : s.gloss}</span>`;
    b.title = `${s.spelled ? "Fingerspelled " : ""}${s.gloss} · ${s.start.toFixed(1)}s`;
    b.onclick = () => seekSeg(i);
    tl.appendChild(b);
  });
}

function seekSeg(i) {
  const s = state.segments[i]; if (!s) return;
  media.currentTime = s.start + 0.01;
  media.play();
}

let lastSeg = -1, lastLetter = -1;
function renderNowWord(seg, t) {
  const el = $("#nowWord");
  if (!seg.spelled) { el.textContent = seg.gloss; $("#nowSigning").classList.remove("spelling"); return; }
  const li = seg.letters.findIndex((l) => t >= l.start && t < l.end);
  if (li === lastLetter && el.childElementCount) return;
  lastLetter = li;
  $("#nowSigning").classList.add("spelling");
  el.innerHTML = seg.letters.map((l, i) => `<b class="${i === li ? "cur" : i < li ? "past" : ""}">${l.letter}</b>`).join("");
}
function tick() {
  const t = media.currentTime;
  const idx = state.segments.findIndex((s) => t >= s.start && t < s.end);
  const segs = $$(".tl-seg");
  segs.forEach((el, i) => {
    const s = state.segments[i];
    const p = Math.min(1, Math.max(0, (t - s.start) / (s.end - s.start)));
    $(".fill", el).style.width = `${p * 100}%`;
  });
  if (idx !== lastSeg) {
    lastSeg = idx;
    segs.forEach((el, i) => el.classList.toggle("active", i === idx));
    $$("#chips .chip").forEach((el) => el.classList.toggle("active", el.dataset.seg !== undefined && +el.dataset.seg === idx));
    const ns = $("#nowSigning");
    lastLetter = -2;
    if (idx >= 0) ns.classList.remove("hidden");
    else ns.classList.add("hidden");
  }
  if (idx >= 0) renderNowWord(state.segments[idx], t);
  if (!media.paused) requestAnimationFrame(tick);
}
function wireMedia(m) {
  const mine = (fn) => () => { if (media === m) fn(); };
  m.addEventListener("play", mine(() => { $("#playBtn").classList.add("playing"); $("#bigPlay").classList.add("hidden"); requestAnimationFrame(tick); }));
  m.addEventListener("pause", mine(() => { $("#playBtn").classList.remove("playing"); tick(); }));
  m.addEventListener("seeked", mine(tick));
  m.addEventListener("ended", mine(() => { if (!m.loop) { $("#bigPlay").classList.remove("hidden"); lastSeg = -2; tick(); } }));
}
wireMedia(video);

$("#playBtn").onclick = () => (media.paused ? media.play() : media.pause());
$("#bigPlay").onclick = () => media.play();
video.addEventListener("click", () => (video.paused ? video.play() : video.pause()));
$$(".speed button").forEach((b) => b.addEventListener("click", () => {
  $$(".speed button").forEach((x) => x.classList.toggle("on", x === b));
  for (const m of [video, avatar]) if (m) m.playbackRate = +b.dataset.speed;
  $("#previewVideo").playbackRate = +b.dataset.speed;
}));
$("#loopBtn").onclick = () => {
  const on = $("#loopBtn").getAttribute("aria-pressed") !== "true";
  for (const m of [video, avatar]) if (m) m.loop = on;
  $("#loopBtn").setAttribute("aria-pressed", String(on));
};
$("#resetView").onclick = () => avatar?.resetView();

// Download: the Human video is a file already; the Avatar is recorded from the canvas in real time.
$("#downloadBtn").addEventListener("click", async (e) => {
  if (media === video) return;
  e.preventDefault();
  const dl = $("#downloadBtn"), label = dl.lastChild;
  if (dl.classList.contains("disabled")) return;
  dl.classList.add("disabled");
  try {
    const blob = await avatar.record((p) => { label.textContent = ` Recording ${Math.round(p * 100)}%`; });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "isl-avatar.webm"; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  } catch (err) { showToast(`Recording failed: ${err.message}`); }
  finally { label.textContent = "Download"; dl.classList.remove("disabled"); }
});

/* ---------------- signer switch (Human | 3D Avatar) ---------------- */
const SIGNER_HINTS = {
  human: "Real signer videos, joined in order",
  avatar: "3D avatar animated from the same videos. Drag to rotate",
};
function renderSignerSwitch() {
  $$(".signer-switch button").forEach((b) => {
    const on = b.dataset.signer === state.signer;
    b.classList.toggle("on", on); b.setAttribute("aria-checked", String(on));
  });
  $("#signerHint").textContent = SIGNER_HINTS[state.signer];
}
$$(".signer-switch button").forEach((b) => b.addEventListener("click", async () => {
  if (state.signer === b.dataset.signer || state.busy) return;
  state.signer = b.dataset.signer;
  try { localStorage.setItem("signer", state.signer); } catch {}
  renderSignerSwitch();
  if (!state.last) return;
  state.busy = true; refreshCta(); hideToast(); step("sign", "");
  try { await signTokens(state.last.tokens, state.last.text); }
  catch (err) { showToast(err.message); }
  finally { loading(false); state.busy = false; refreshCta(); }
}));
renderSignerSwitch();

document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !["TEXTAREA", "INPUT", "BUTTON"].includes(document.activeElement.tagName) && media.src) {
    e.preventDefault(); media.paused ? media.play() : media.pause();
  }
  if (e.key === "Escape") { closeDrawer(); closeTopicModal(); }
});

/* ---------------- sign library ---------------- */
async function loadVocab() {
  try {
    const res = await fetch("/vocabulary");
    if (!res.ok) throw new Error(await errorText(res));
    const data = await res.json();
    state.vocab = data.signs;
    state.alphabet = data.alphabet || [];
    state.fingerspelling = !!data.fingerspelling;
    $("#vocabCount").textContent = state.vocab.length;
    $("#libCount").textContent = state.vocab.length;
    renderLibrary("");
  } catch (e) { $("#vocabCount").textContent = "!"; }
}
function libHeading(text) {
  const h = document.createElement("div"); h.className = "lib-heading"; h.textContent = text;
  $("#libGrid").appendChild(h);
}
function renderLibrary(q) {
  const grid = $("#libGrid"); grid.innerHTML = "";
  const query = q.trim().toUpperCase();
  const t = currentTopic();
  if (t) {  // the topic's signs first; ones without a clip yet are shown greyed out
    const words = t.words.filter((w) => w.gloss.includes(query));
    if (words.length) {
      libHeading(`${t.label} · ${t.covered}/${t.total} signs ready`);
      for (const w of words) {
        if (w.clip) addLibButton({ gloss: w.gloss, clip: w.clip }, "topic");
        else addLibButton({ gloss: w.gloss }, "missing");
      }
    }
  }
  const letters = state.alphabet.filter((l) => !query || l.letter === query);
  if (letters.length) {
    libHeading(state.fingerspelling ? "Alphabet · fingerspelling on" : `Alphabet · ${state.alphabet.length}/26, fingerspelling off`);
    for (const l of letters) addLibButton({ gloss: l.letter, clip: l.clip }, "letter");
  }
  if (t || letters.length) libHeading("All signs");
  const items = state.vocab.filter((s) => s.gloss.includes(query));
  for (const s of items) addLibButton(s);
  if (!items.length) grid.insertAdjacentHTML("beforeend", '<span class="muted">No matching signs.</span>');
}
function addLibButton(s, cls = "") {
  const grid = $("#libGrid");
  {
    const b = document.createElement("button");
    b.textContent = s.gloss; b.title = s.gloss;
    if (!s.clip) {
      b.disabled = true;
      b.title = `${s.gloss}: no sign clip yet (${state.fingerspelling ? "fingerspelled" : "skipped"} in translations)`;
    }
    b.onclick = () => {
      $$("#libGrid button").forEach((x) => x.classList.toggle("on", x === b));
      const pv = $("#previewVideo"); pv.src = s.clip; pv.play().catch(() => {});
      $("#previewLabel").textContent = s.gloss;
    };
    if (cls) b.classList.add(cls);
    grid.appendChild(b);
  }
}
$("#libSearch").addEventListener("input", (e) => renderLibrary(e.target.value));
function openDrawer() { $("#drawer").classList.add("open"); $("#drawer").setAttribute("aria-hidden", "false"); $("#scrim").classList.remove("hidden"); $("#libSearch").focus(); }
function closeDrawer() { $("#drawer").classList.remove("open"); $("#drawer").setAttribute("aria-hidden", "true"); $("#scrim").classList.add("hidden"); $("#previewVideo").pause(); }
$("#libraryBtn").onclick = openDrawer;
$("#closeDrawer").onclick = closeDrawer;
$("#scrim").onclick = closeDrawer;

renderExamples();
loadVocab();
loadTopics();
refreshCta();
