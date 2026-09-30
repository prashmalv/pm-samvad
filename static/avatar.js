// Browser 3D signing avatar (three.js, MIT). Driven by joint positions from POST /avatar-motion
// (layout documented in app/avatar_motion.py). AvatarPlayer mimics the <video> element API
// (play, pause, currentTime, duration, playbackRate, loop, events) so the page's controls work as-is.
// It shows a realistic MPFB2 character (models/signer.glb or signer-male.glb, see avatar-human.js)
// when available, and falls back to a simple figure built from primitives in code.
import * as THREE from "three";
import { OrbitControls } from "./vendor/three/OrbitControls.js";
import { GLTFLoader } from "./vendor/three/loaders/GLTFLoader.js";
import { HumanSigner } from "./avatar-human.js";

// The selectable characters: same rig and motion, different body and colours. URLs are relative to this
// script, not the page.
export const CHARACTERS = {
  woman: { url: new URL("./models/signer.glb", import.meta.url).href, colors: {} },
  man: { url: new URL("./models/signer-male.glb", import.meta.url).href, colors: { shirt: 0x3f5f7a, hair: 0x1e1a18 } },
};

const N = 52;
const J = { pelvis: 0, neck: 1, head: 2, face: 3, lSh: 4, rSh: 5, lEl: 6, rEl: 7, lWr: 8, rWr: 9, lHand: 10, rHand: 31 };
// finger chains within one hand's 21 points
const FINGERS = [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
const PALM_EDGES = [[0, 1], [0, 5], [0, 9], [0, 13], [0, 17], [5, 9], [9, 13], [13, 17], [1, 5]];

const COLORS = {
  skin: 0xb07b58, shirt: 0x5a46d8, shirtDark: 0x4636b4, hair: 0x241c2a, eye: 0x1a1420, pants: 0x2c2b3d,
};

export class AvatarPlayer extends EventTarget {
  /** @param {{character?: string, signerUrl?: string}} opts signerUrl loads a specific .glb instead (dev) */
  constructor(canvas, { character = "woman", signerUrl = null } = {}) {
    super();
    this.canvas = canvas;
    this.character = CHARACTERS[character] ? character : "woman";
    this._humans = new Map();  // url -> HumanSigner already built, so switching back is instant
    this.frames = null; this.fps = 25; this.T = 0;
    this._t = 0; this.paused = true; this.loop = false; this.playbackRate = 1; this.ended = false;
    this._last = 0;
    this.human = null;       // HumanSigner once signer.glb has loaded
    this._initScene();
    this.ready = this._loadHuman(signerUrl || CHARACTERS[this.character].url, signerUrl ? {} : CHARACTERS[this.character].colors);
    this._raf = requestAnimationFrame((ts) => this._tick(ts));
  }

  /** Switch to another character ("woman" | "man"), keeping the current signing and time. */
  async setCharacter(id) {
    if (!CHARACTERS[id] || id === this.character) return;
    this.character = id;
    await this.ready;
    this.ready = this._loadHuman(CHARACTERS[id].url, CHARACTERS[id].colors);
    return this.ready;
  }

  async _loadHuman(url, colors) {
    try {
      let human = this._humans.get(url);
      if (!human) {  // built once: HumanSigner reads the rest pose from the bones, so never rebuild a posed one
        human = new HumanSigner(await new GLTFLoader().loadAsync(url), colors);
        this._humans.set(url, human);
      }
      if (this.human) this.scene.remove(this.human.root);
      this.human = human;
      this.scene.add(this.human.root);
      this.prims.visible = false;
      // neutral studio light for real skin (the coloured lights suit the simple figure only)
      const L = this.lights;
      L.hemi.color.set(0xffffff); L.hemi.groundColor.set(0x4a4450); L.hemi.intensity = 1.25;
      L.rim.color.set(0xfff1e6); L.rim.intensity = 0.8;
      L.fill.color.set(0xfff6ee); L.fill.intensity = 0.7;
      if (this.frames) { this.human.prepare(this.frames, this.T); this._pose(this._t); } else this.human.restPose();
      this.resetView();
      this._emit("human");
    } catch (err) {
      console.info("Realistic signer not available, using the simple avatar:", err.message || err);
    }
  }

  /* ---------------- video-like API ---------------- */
  get duration() { return this.T > 1 ? (this.T - 1) / this.fps : 0; }
  get currentTime() { return this._t; }
  set currentTime(v) { this._t = Math.min(Math.max(0, v), this.duration); this.ended = false; this._pose(this._t); this._emit("seeked"); }
  get src() { return this.frames ? "avatar" : ""; }
  play() { if (!this.frames) return Promise.resolve(); if (this.ended || this._t >= this.duration) this._t = 0; this.ended = false; this.paused = false; this._emit("play"); return Promise.resolve(); }
  pause() { if (!this.paused) { this.paused = true; this._emit("pause"); } }
  load(seq) {
    this.fps = seq.fps; this.T = seq.frames.length;
    this.frames = new Float32Array(this.T * N * 3);
    seq.frames.forEach((f, i) => this.frames.set(f, i * N * 3));
    this.human?.prepare(this.frames, this.T);
    this._t = 0; this.ended = false; this.paused = true; this._pose(0);
  }
  resize() {
    const r = this.canvas.getBoundingClientRect();
    if (!r.width) return;
    this.renderer.setSize(r.width, r.height, false);
    this.camera.aspect = r.width / r.height; this.camera.updateProjectionMatrix();
  }
  resetView() {
    if (this.human) {  // framed for a 1.58 m character, scaled to this one's height
      const k = (this.human.height || 1.58) / 1.58;
      this.camera.position.set(0, 1.24 * k, 1.95 * k); this.controls.target.set(0, 1.16 * k, 0);
    }
    else { this.camera.position.set(0, 1.34, 2.15); this.controls.target.set(0, 1.26, 0); }
    this.controls.update();
  }
  _emit(type) { this.dispatchEvent(new Event(type)); }

  /* ---------------- scene ---------------- */
  _initScene() {
    const r = this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, preserveDrawingBuffer: true });
    r.setPixelRatio(Math.min(devicePixelRatio, 2));
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.shadowMap.enabled = true; r.shadowMap.type = THREE.PCFShadowMap;
    const scene = this.scene = new THREE.Scene();
    scene.background = new THREE.Color(0x14122a);
    scene.fog = new THREE.Fog(0x14122a, 3.5, 7);

    this.camera = new THREE.PerspectiveCamera(32, 16 / 9, 0.05, 20);
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true; this.controls.enablePan = false;
    this.controls.minDistance = 0.8; this.controls.maxDistance = 3.2;
    this.controls.minPolarAngle = 0.9; this.controls.maxPolarAngle = 2.0;
    this.resetView();

    const hemi = new THREE.HemisphereLight(0xdcd6ff, 0x2a2440, 1.1); scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(1.2, 2.6, 2.2); key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024); key.shadow.camera.left = -1; key.shadow.camera.right = 1;
    key.shadow.camera.top = 2; key.shadow.camera.bottom = 0; key.shadow.bias = -0.0005;
    scene.add(key);
    const rim = new THREE.DirectionalLight(0xec5fa5, 1.2); rim.position.set(-2, 2, -1.5); scene.add(rim);
    const fill = new THREE.DirectionalLight(0x8b73ff, 0.6); fill.position.set(-1.5, 1, 2); scene.add(fill);
    this.lights = { hemi, key, rim, fill };

    // backdrop disc + floor glow
    const floor = new THREE.Mesh(new THREE.CircleGeometry(1.6, 64), new THREE.MeshStandardMaterial({ color: 0x1d1a38, roughness: 0.9 }));
    floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);

    const mat = (c, rough = 0.62) => new THREE.MeshStandardMaterial({ color: c, roughness: rough, metalness: 0.02 });
    this.m = { skin: mat(COLORS.skin, 0.55), shirt: mat(COLORS.shirt, 0.72), shirtDark: mat(COLORS.shirtDark, 0.72), hair: mat(COLORS.hair, 0.8), eye: mat(COLORS.eye, 0.3), pants: mat(COLORS.pants, 0.8) };
    const unitCyl = new THREE.CylinderGeometry(1, 1, 1, 20, 1);
    const sphere = new THREE.SphereGeometry(1, 24, 16);
    // the simple figure lives in one group, hidden once the realistic signer has loaded
    const prims = this.prims = new THREE.Group(); scene.add(prims);
    const add = (geo, material) => { const o = new THREE.Mesh(geo, material); o.castShadow = true; prims.add(o); return o; };
    this.seg = (radius, material, radius2 = radius) => {
      const geo = radius2 === radius ? unitCyl : new THREE.CylinderGeometry(radius2 / radius, 1, 1, 20, 1);
      const o = add(geo, material); o.userData.r = radius; return o;
    };
    this.ball = (radius, material) => { const o = add(sphere, material); o.scale.setScalar(radius); return o; };

    // torso, hips, legs (static - the signing is all upper body)
    const torso = add(new THREE.CapsuleGeometry(0.15, 0.3, 8, 24), this.m.shirt);
    torso.scale.set(1.24, 1, 0.66); torso.position.set(0, 1.18, -0.005);
    const hips = add(new THREE.CapsuleGeometry(0.14, 0.08, 8, 24), this.m.pants);
    hips.scale.set(1.15, 1, 0.7); hips.position.set(0, 0.93, 0);
    for (const x of [-0.09, 0.09]) { const leg = add(new THREE.CapsuleGeometry(0.07, 0.72, 6, 16), this.m.pants); leg.position.set(x, 0.47, 0.01); }
    this.neck = this.seg(0.045, this.m.skin);
    // head group: skull, hair cap, eyes, nose
    this.headGroup = new THREE.Group(); prims.add(this.headGroup);
    const skull = new THREE.Mesh(sphere, this.m.skin); skull.scale.set(0.092, 0.112, 0.1); skull.castShadow = true;
    const hair = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16, 0, Math.PI * 2, 0, Math.PI * 0.43), this.m.hair);
    hair.scale.set(0.099, 0.119, 0.107); hair.position.set(0, 0.008, -0.01); hair.rotation.x = -0.55;
    const eyes = [-0.032, 0.032].map((x) => { const e = new THREE.Mesh(sphere, this.m.eye); e.scale.setScalar(0.0115); e.position.set(x, 0.014, 0.088); return e; });
    const brow = [-0.032, 0.032].map((x) => { const b = new THREE.Mesh(new THREE.CapsuleGeometry(0.004, 0.022, 4, 8), this.m.hair); b.rotation.z = Math.PI / 2; b.position.set(x, 0.037, 0.087); return b; });
    const nose = new THREE.Mesh(sphere, this.m.skin); nose.scale.set(0.012, 0.018, 0.016); nose.position.set(0, -0.012, 0.097);
    const mouth = new THREE.Mesh(new THREE.CapsuleGeometry(0.004, 0.026, 4, 8), mat(0x7a3f3a, 0.5)); mouth.rotation.z = Math.PI / 2; mouth.position.set(0, -0.05, 0.085);
    this.headGroup.add(skull, hair, ...eyes, ...brow, nose, mouth);

    // arms: shoulder ball, sleeve (upper arm), forearm, wrist
    this.arm = {};
    for (const s of ["l", "r"]) {
      this.arm[s] = {
        shoulder: this.ball(0.052, this.m.shirt), upper: this.seg(0.047, this.m.shirt, 0.041),
        elbow: this.ball(0.036, this.m.skin), fore: this.seg(0.036, this.m.skin, 0.03), wrist: this.ball(0.029, this.m.skin),
      };
    }
    // hands: knuckle balls, finger bones, palm (thick edges + a filled fan)
    this.hand = {};
    for (const s of ["l", "r"]) {
      const h = { joints: [], bones: [], palm: [] };
      for (let i = 0; i < 21; i++) h.joints.push(this.ball(i === 0 ? 0.022 : [4, 8, 12, 16, 20].includes(i) ? 0.0085 : 0.0105, this.m.skin));
      for (const chain of FINGERS) for (let k = 0; k < chain.length - 1; k++) h.bones.push([chain[k], chain[k + 1], this.seg(chain[0] === 0 ? 0.011 : 0.0095, this.m.skin)]);
      for (const [a, b] of PALM_EDGES) h.palm.push([a, b, this.seg(0.014, this.m.skin)]);
      const fanGeo = new THREE.BufferGeometry();
      fanGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6 * 3), 3));
      fanGeo.setIndex([0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5]);
      h.fan = new THREE.Mesh(fanGeo, new THREE.MeshStandardMaterial({ color: COLORS.skin, roughness: 0.55, side: THREE.DoubleSide }));
      h.fan.castShadow = true; prims.add(h.fan);
      this.hand[s] = h;
    }
    this._v = [...Array(N)].map(() => new THREE.Vector3());
    this._up = new THREE.Vector3(0, 1, 0);
    this._tmp = new THREE.Vector3();
    this.resize();
    new ResizeObserver(() => this.resize()).observe(this.canvas);
  }

  /* ---------------- posing ---------------- */
  _place(obj, a, b) {
    const d = this._tmp.subVectors(b, a); const len = d.length() || 1e-4;
    obj.position.addVectors(a, b).multiplyScalar(0.5);
    obj.quaternion.setFromUnitVectors(this._up, d.divideScalar(len));
    const r = obj.userData.r; obj.scale.set(r, len, r);
  }

  _pose(t) {
    if (!this.frames) return;
    const f = Math.min(t * this.fps, this.T - 1), i0 = Math.floor(f), i1 = Math.min(i0 + 1, this.T - 1), w = f - i0;
    const F = this.frames, o0 = i0 * N * 3, o1 = i1 * N * 3;
    for (let j = 0; j < N; j++) {
      const k = j * 3;
      this._v[j].set(F[o0 + k] * (1 - w) + F[o1 + k] * w, F[o0 + k + 1] * (1 - w) + F[o1 + k + 1] * w, F[o0 + k + 2] * (1 - w) + F[o1 + k + 2] * w);
    }
    const V = this._v;
    if (this.human) { this.human.pose(V, f); return; }
    this._place(this.neck, V[J.neck], new THREE.Vector3().copy(V[J.neck]).add(new THREE.Vector3(0, 0.1, 0)));
    this.headGroup.position.copy(V[J.head]);
    this.headGroup.lookAt(this._tmp.copy(V[J.face]).sub(V[J.head]).multiplyScalar(4).add(V[J.head]));
    for (const [s, sh, el, wr, base] of [["l", J.lSh, J.lEl, J.lWr, J.lHand], ["r", J.rSh, J.rEl, J.rWr, J.rHand]]) {
      const a = this.arm[s];
      a.shoulder.position.copy(V[sh]); this._place(a.upper, V[sh], V[el]);
      a.elbow.position.copy(V[el]); this._place(a.fore, V[el], V[wr]); a.wrist.position.copy(V[wr]);
      const h = this.hand[s];
      for (let i = 0; i < 21; i++) h.joints[i].position.copy(V[base + i]);
      for (const [p, q, m] of h.bones) this._place(m, V[base + p], V[base + q]);
      for (const [p, q, m] of h.palm) this._place(m, V[base + p], V[base + q]);
      const pos = h.fan.geometry.attributes.position;
      [0, 1, 5, 9, 13, 17].forEach((idx, n) => pos.setXYZ(n, V[base + idx].x, V[base + idx].y, V[base + idx].z));
      pos.needsUpdate = true; h.fan.geometry.computeVertexNormals(); h.fan.geometry.computeBoundingSphere();
    }
  }

  _tick(ts) {
    const dt = this._last ? Math.min(0.1, (ts - this._last) / 1000) : 0; this._last = ts;
    if (!this.paused && this.frames) {
      this._t += dt * this.playbackRate;
      if (this._t >= this.duration) {
        if (this.loop) this._t = 0;
        else { this._t = this.duration; this.paused = true; this.ended = true; this._emit("pause"); this._emit("ended"); }
      }
      this._pose(this._t);
      this._emit("timeupdate");
    }
    this.human?.update(dt);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this._raf = requestAnimationFrame((t2) => this._tick(t2));
  }

  /* ---------------- recording (Download) ---------------- */
  async record(onProgress) {
    const stream = this.canvas.captureStream(30);
    const type = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find((t) => MediaRecorder.isTypeSupported(t));
    const rec = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 6e6 });
    const chunks = []; rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    const wasLoop = this.loop, wasRate = this.playbackRate;
    this.loop = false; this.playbackRate = 1; this.currentTime = 0;
    const done = new Promise((r) => { rec.onstop = r; });
    rec.start();
    await this.play();
    await new Promise((r) => {
      const iv = setInterval(() => { onProgress?.(this._t / this.duration); if (this.ended) { clearInterval(iv); r(); } }, 100);
    });
    rec.stop(); await done;
    this.loop = wasLoop; this.playbackRate = wasRate;
    return new Blob(chunks, { type: "video/webm" });
  }
}
