// Realistic human signer: drives a rigged character (static/models/signer.glb, made with MPFB2 by
// scripts/avatar/build_signer.py) from the same 52 joint positions that animate the simple avatar
// (layout in app/avatar_motion.py). Positions become bone rotations:
//   - arms: each bone is swung (shortest rotation) so it points where the video's joint points, parent
//     first, so every bone keeps the character's own length;
//   - fingers: anatomical hinges with human joint limits (avatar-hands.js), so noisy tracking can't
//     twist, cross or over-bend them;
//   - hands: full orientation from two palm directions (wrist -> middle knuckle, pinky -> index knuckle),
//     so palm facing is right, not just pointing;
//   - head: turned to face where the video's face points.
// Rig bone names are MPFB's "game_engine" rig (Unreal-style: upperarm_l, index_01_l ...).
// Both data and model use x = avatar's left, y = up, z = towards the viewer, so no mirroring is needed.
import * as THREE from "three";
import { HandRig, N_ANGLES } from "./avatar-hands.js";

const J = { head: 2, face: 3, lSh: 4, rSh: 5, lEl: 6, rEl: 7, lWr: 8, rWr: 9, lHand: 10, rHand: 31 };
// hand landmark indices per finger (MediaPipe order); bones <finger>_01.._03 span consecutive points
const FINGERS = { thumb: [1, 2, 3, 4], index: [5, 6, 7, 8], middle: [9, 10, 11, 12], ring: [13, 14, 15, 16], pinky: [17, 18, 19, 20] };
const Y = new THREE.Vector3(0, 1, 0);

// Look tweaks applied at load (kept here so the .glb can be rebuilt without losing them).
const LOOK = {
  shirt: { match: /t-shirt|shirt|top|sweater/i, color: 0x9a6440 },   // brown top (texture is light grey, so this sets the colour)
  hair: { match: /hair/i, color: 0x3a2a22 },                           // dark brown, whatever colour the hair asset ships in
  pants: { match: /pants|trousers/i, color: 0x4a4a52 },
  skin: { match: /body/i, color: 0xf3e2d6 },                          // a touch warmer than the texture
  clothes: /t-shirt|shirt|top|sweater|pants|trousers|shoes/i,
};

export class HumanSigner {
  /** @param {object} gltf result of GLTFLoader */
  constructor(gltf) {
    this.root = gltf.scene;
    this.bones = {};
    this.faceMeshes = [];
    this.root.traverse((o) => {
      if (o.isBone) this.bones[o.name] = o;
      if (o.isMesh) {
        o.castShadow = true; o.frustumCulled = false;  // skinned bounds don't follow the pose
        this._fixMaterial(o);
        if (o.morphTargetDictionary && "eyeBlinkLeft" in o.morphTargetDictionary) this.faceMeshes.push(o);
      }
    });
    const missing = ["upperarm_l", "lowerarm_l", "hand_l", "index_01_l", "head"].filter((n) => !this.bones[n]);
    if (missing.length) throw new Error(`signer.glb is missing bones: ${missing.join(", ")} (needs MPFB's game_engine rig)`);
    this.rest = new Map(Object.values(this.bones).map((b) => [b, b.quaternion.clone()]));
    this.root.updateMatrixWorld(true);
    // head: which local direction is "forward" (+z in the world at rest)
    this.headFwd = new THREE.Vector3(0, 0, 1).applyQuaternion(this.bones.head.getWorldQuaternion(new THREE.Quaternion()).invert());
    this.hands = {};
    for (const s of ["l", "r"]) {
      try { this.hands[s] = new HandRig(this.bones, s, this.rest); } catch (err) { console.warn(err.message); }
    }
    this._q = new THREE.Quaternion(); this._q2 = new THREE.Quaternion(); this._q3 = new THREE.Quaternion();
    this._a = new THREE.Vector3(); this._b = new THREE.Vector3(); this._c = new THREE.Vector3(); this._d = new THREE.Vector3();
    this._m1 = new THREE.Matrix4(); this._m2 = new THREE.Matrix4();
    this.nextBlink = 1.5; this.blinkT = -1;
  }

  _fixMaterial(mesh) {
    for (const m of [mesh.material].flat()) {
      if (m.transparent) { m.transparent = false; m.alphaTest = 0.5; m.depthWrite = true; }  // hair, brows, lashes
      const name = `${m.name} ${mesh.name}`;
      if (LOOK.clothes.test(name)) { m.polygonOffset = true; m.polygonOffsetFactor = -1; m.polygonOffsetUnits = -2; } // skin never pokes through
      for (const k of ["shirt", "pants", "skin", "hair"]) if (LOOK[k].match.test(name)) m.color.set(LOOK[k].color);
    }
  }

  // ---------------------------------------------------------------- bone helpers
  /** Rotate `bone` (in world space) by `q`, keeping its parent as is. */
  _rotateWorld(bone, q) {
    bone.updateWorldMatrix(true, false);
    const world = bone.getWorldQuaternion(this._q2);
    const parent = bone.parent.getWorldQuaternion(this._q3).invert();
    bone.quaternion.copy(parent.multiply(q.clone().multiply(world)));
    bone.updateWorldMatrix(false, true);
  }

  /** Current world direction of a bone: towards `child` bone if given, else along its own +Y axis. */
  _dir(bone, child, out) {
    bone.updateWorldMatrix(true, false);
    if (child) {
      child.updateWorldMatrix(true, false);
      return out.setFromMatrixPosition(child.matrixWorld).sub(this._d.setFromMatrixPosition(bone.matrixWorld)).normalize();
    }
    return out.copy(Y).applyQuaternion(bone.getWorldQuaternion(this._q2)).normalize();
  }

  /** Swing `bone` so it points along the target direction a -> b (world joint positions). */
  _aim(name, childName, a, b) {
    const bone = this.bones[name];
    if (!bone || !a || !b) return;
    const target = this._a.subVectors(b, a);
    if (target.lengthSq() < 1e-10) return;
    target.normalize();
    const cur = this._dir(bone, childName && this.bones[childName], this._b);
    this._rotateWorld(bone, this._q.setFromUnitVectors(cur, target));
  }

  /** Full hand orientation from two palm directions. */
  _orientHand(side, P) {
    const hand = this.bones[`hand_${side}`], mid = this.bones[`middle_01_${side}`];
    const idx = this.bones[`index_01_${side}`], pnk = this.bones[`pinky_01_${side}`];
    if (!hand || !mid || !idx || !pnk) return;
    const basis = (fwd, across, m) => {
      const x = fwd.clone().normalize();
      const z = across.clone().sub(x.clone().multiplyScalar(across.dot(x))).normalize();
      return m.makeBasis(x, new THREE.Vector3().crossVectors(z, x), z);
    };
    [hand, mid, idx, pnk].forEach((b) => b.updateWorldMatrix(true, false));
    const pos = (b) => new THREE.Vector3().setFromMatrixPosition(b.matrixWorld);
    const cur = basis(pos(mid).sub(pos(hand)), pos(idx).sub(pos(pnk)), this._m1);
    const tgt = basis(P[9].clone().sub(P[0]), P[5].clone().sub(P[17]), this._m2);
    if (!Number.isFinite(tgt.elements[0])) return;
    const q = new THREE.Quaternion().setFromRotationMatrix(tgt.multiply(cur.transpose()));  // cur is orthonormal: inverse = transpose
    this._rotateWorld(hand, q);
  }

  // ---------------------------------------------------------------- posing
  /**
   * Pre-compute time-filtered finger angles for a whole sentence.
   * @param {Float32Array} F  frames, T x 52 joints x 3   @param {number} T frame count
   */
  prepare(F, T) {
    this.handAngles = {};
    for (const [s, base] of [["l", J.lHand], ["r", J.rHand]]) {
      const rig = this.hands[s];
      if (!rig) continue;
      const pts = [...Array(21)].map(() => new THREE.Vector3());
      this.handAngles[s] = rig.prepare(T, (i) => {
        for (let k = 0; k < 21; k++) pts[k].fromArray(F, (i * 52 + base + k) * 3);
        return pts;
      });
    }
    this._angleBuf = new Float32Array(N_ANGLES);
  }

  _fingers(s, P, f) {
    const rig = this.hands[s], A = this.handAngles?.[s];
    if (!rig) return;
    if (!A || f === undefined) { rig.pose(P); return; }
    const T = A.length / N_ANGLES, i0 = Math.min(Math.floor(f), T - 1), i1 = Math.min(i0 + 1, T - 1), w = f - i0;
    const out = this._angleBuf;
    for (let c = 0; c < N_ANGLES; c++) out[c] = A[i0 * N_ANGLES + c] * (1 - w) + A[i1 * N_ANGLES + c] * w;
    rig.apply(out);
  }

  /** Pose from joint positions V (array of 52 THREE.Vector3, avatar space); f = fractional frame index. */
  pose(V, f) {
    for (const [b, q] of this.rest) b.quaternion.copy(q);
    this.root.updateMatrixWorld(true);
    for (const [s, sh, el, wr, base] of [["l", J.lSh, J.lEl, J.lWr, J.lHand], ["r", J.rSh, J.rEl, J.rWr, J.rHand]]) {
      this._aim(`upperarm_${s}`, `lowerarm_${s}`, V[sh], V[el]);
      this._aim(`lowerarm_${s}`, `hand_${s}`, V[el], V[wr]);
      const P = V.slice(base, base + 21);
      this._orientHand(s, P);
      this._aim(`thumb_01_${s}`, `thumb_02_${s}`, P[FINGERS.thumb[0]], P[FINGERS.thumb[1]]);  // thumb base: tracked direction
      this._fingers(s, P, f);                                                                 // fingers + thumb joints: anatomical
    }
    this._turnHead(V[J.head], V[J.face]);
  }

  _turnHead(head, face) {
    const bone = this.bones.head;
    if (!head || !face) return;
    const target = this._a.subVectors(face, head);
    if (target.lengthSq() < 1e-10) return;
    target.normalize();
    bone.updateWorldMatrix(true, false);
    const cur = this._b.copy(this.headFwd).applyQuaternion(bone.getWorldQuaternion(this._q2)).normalize();
    const q = this._q.setFromUnitVectors(cur, target);
    const neck = this.bones.neck_01;
    if (neck) this._rotateWorld(neck, new THREE.Quaternion().slerp(q, 0.35));   // share the turn with the neck
    this._rotateWorld(bone, new THREE.Quaternion().slerp(q, neck ? 0.65 : 1));
  }

  /** Relaxed standing pose (arms down) for before any sentence is loaded. */
  restPose() {
    for (const [b, q] of this.rest) b.quaternion.copy(q);
    this.root.updateMatrixWorld(true);
    for (const [s, sx] of [["l", 1], ["r", -1]]) {
      const sh = new THREE.Vector3().setFromMatrixPosition(this.bones[`upperarm_${s}`].matrixWorld);
      const el = sh.clone().add(new THREE.Vector3(0.07 * sx, -1, 0.06).normalize());
      const wr = el.clone().add(new THREE.Vector3(0.03 * sx, -1, 0.2).normalize());
      this._aim(`upperarm_${s}`, `lowerarm_${s}`, sh, el);
      this._aim(`lowerarm_${s}`, `hand_${s}`, el, wr);
    }
  }

  /** Natural blinking (call every frame with elapsed seconds). */
  update(dt) {
    if (!this.faceMeshes.length) return;
    this.nextBlink -= dt;
    if (this.nextBlink <= 0 && this.blinkT < 0) { this.blinkT = 0; this.nextBlink = 2.5 + Math.random() * 3.5; }
    let w = 0;
    if (this.blinkT >= 0) {
      this.blinkT += dt;
      const p = this.blinkT / 0.16;                 // one blink takes ~160 ms
      w = p < 0.5 ? p * 2 : Math.max(0, 2 - p * 2);
      if (p >= 1) this.blinkT = -1;
    }
    for (const m of this.faceMeshes) {
      const d = m.morphTargetDictionary;
      m.morphTargetInfluences[d.eyeBlinkLeft] = w;
      if ("eyeBlinkRight" in d) m.morphTargetInfluences[d.eyeBlinkRight] = w;
    }
  }
}
