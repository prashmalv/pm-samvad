// Anatomical finger posing for the realistic signer (used by avatar-human.js).
//
// Pointing each finger bone straight at the tracked points (a free swing) lets tracking noise twist
// fingers sideways, bend joints backwards and cross fingers. Real fingers don't move like that, so here
// each joint is posed the way a hand can actually move:
//   - knuckle (MCP): bends towards the palm, plus a little sideways spread;
//   - middle and end joints (PIP, DIP): pure hinges, bending towards the palm only;
//   - every angle is clamped to a human range, and the end joint follows the middle joint (as tendons
//     couple them), which also smooths out noisy tracking;
//   - thumb: its base keeps the tracked direction, its two outer joints are hinges in its own bend plane;
//   - when fingers hide each other (fists) the tracked angles come out impossible; then each finger's
//     curl is taken from how close its tip is to the palm instead;
//   - over a whole sentence (prepare), every angle is median-filtered in time (drops one-frame tracking
//     glitches, keeps real handshape changes sharp) and ring/pinky partly follow their neighbours, as
//     their shared tendons make real fingers do.
// Angles are measured in the hand's own frame (wrist -> middle knuckle, pinky -> index knuckle, palm
// normal) on both the video data and the character, so the result doesn't depend on how the hand is held.
import * as THREE from "three";

const DEG = Math.PI / 180;
const LIMITS = {                      // degrees; positive = curling towards the palm
  mcp: [-15, 90],
  pip: [0, 105],
  dip: [0, 80],
  spread: 15,                         // max sideways change at the knuckle
  thumb: [[0, 60], [0, 80]],          // thumb MCP and IP bend
};
const DIP_FOLLOWS_PIP = 0.7;          // anatomical DIP ~ 0.7 x PIP
const SPREAD_DAMPING = 0.6;
const MEDIAN_FRAMES = 7;              // ~0.28 s at 25 fps
const COUPLING = { ring: [["middle", 0.2], ["pinky", 0.2]], pinky: [["ring", 0.25]] };  // share of a neighbour's flex
const N_ANGLES = 18;                  // 4 fingers x (mcp, pip, dip, spread) + thumb (2 bends)
// Fingertip-distance fallback: tip distance from the palm centre (in palm lengths) of a fully curled finger,
// the total bend of a fist, and how a curl splits over the three joints (knuckle, middle, end).
const FIST_TIP = 0.35;
const FIST_CURL = 235 * DEG;
const CURL_SHARE = [0.38, 0.42, 0.2];
const FINGER_POINTS = { index: [5, 6, 7, 8], middle: [9, 10, 11, 12], ring: [13, 14, 15, 16], pinky: [17, 18, 19, 20] };
const THUMB_POINTS = [1, 2, 3, 4];
const Y = new THREE.Vector3(0, 1, 0);

const clamp = (v, [lo, hi]) => Math.min(hi * DEG, Math.max(lo * DEG, v));

/** Angle from u to v around `axis` (radians), after projecting both onto the plane perpendicular to it. */
function signedAngle(u, v, axis) {
  const a = u.clone().addScaledVector(axis, -u.dot(axis));
  const b = v.clone().addScaledVector(axis, -v.dot(axis));
  return Math.atan2(a.clone().cross(b).dot(axis), a.dot(b));
}

/** Hand frame: fwd = wrist -> middle knuckle, acr = pinky -> index knuckle (made perpendicular), n = fwd x acr. */
function handFrame(wrist, middle, index, pinky) {
  const fwd = middle.clone().sub(wrist).normalize();
  const acr = index.clone().sub(pinky);
  acr.addScaledVector(fwd, -acr.dot(fwd)).normalize();
  return { fwd, acr, n: new THREE.Vector3().crossVectors(fwd, acr) };
}

const worldPos = (b) => new THREE.Vector3().setFromMatrixPosition(b.matrixWorld);
const worldQuat = (b) => b.getWorldQuaternion(new THREE.Quaternion());
const toLocal = (bone, v) => v.clone().applyQuaternion(worldQuat(bone).invert()).normalize();
const tipDir = (bone) => Y.clone().applyQuaternion(worldQuat(bone)).normalize();  // last bone: along its own +Y

export class HandRig {
  /**
   * Measure the character's hand in its rest pose (call with the skeleton at rest, world matrices updated).
   * @param {Record<string, THREE.Bone>} bones  @param {"l"|"r"} side  @param {Map} rest  bone -> rest local quaternion
   */
  constructor(bones, side, rest) {
    this.rest = rest;
    const B = (n) => bones[`${n}_${side}`];
    const hand = bones[`hand_${side}`];
    if (!hand || !B("middle_01") || !B("index_01") || !B("pinky_01")) throw new Error(`hand ${side}: bones missing`);
    const F = handFrame(worldPos(hand), worldPos(B("middle_01")), worldPos(B("index_01")), worldPos(B("pinky_01")));
    // With fwd x acr as the palm normal, curling is a negative turn on the left hand and positive on the right.
    this.flexSign = side === "l" ? -1 : 1;

    this.fingers = [];
    for (const name of Object.keys(FINGER_POINTS)) {
      const bs = [1, 2, 3].map((k) => B(`${name}_0${k}`));
      if (bs.some((b) => !b)) continue;
      const s = [worldPos(bs[1]).sub(worldPos(bs[0])).normalize(), worldPos(bs[2]).sub(worldPos(bs[1])).normalize(), tipDir(bs[2])];
      this.fingers.push({
        name, bones: bs,
        rest: [signedAngle(F.fwd, s[0], F.acr), signedAngle(s[0], s[1], F.acr), signedAngle(s[1], s[2], F.acr)],
        restSpread: signedAngle(F.fwd, s[0], F.n),
        flexAxis: bs.map((b) => toLocal(b, F.acr)),     // world "across" axis, in each bone's own frame
        spreadAxis: toLocal(bs[0], F.n),
      });
    }

    const tb = [1, 2, 3].map((k) => B(`thumb_0${k}`));
    if (tb.every(Boolean)) {
      const t = [worldPos(tb[1]).sub(worldPos(tb[0])).normalize(), worldPos(tb[2]).sub(worldPos(tb[1])).normalize(), tipDir(tb[2])];
      // the thumb's own bend plane; if the rest thumb is nearly straight, fall back to "towards the palm"
      let axis = new THREE.Vector3().crossVectors(t[0], t[1]);
      if (axis.length() < 0.05) axis.crossVectors(t[0], F.n);
      axis.normalize();
      this.thumb = {
        bones: tb,
        rest: [signedAngle(t[0], t[1], axis), signedAngle(t[1], t[2], axis)],
        axis: [toLocal(tb[1], axis), toLocal(tb[2], axis)],
      };
    }
    this._q = new THREE.Quaternion(); this._q2 = new THREE.Quaternion();
  }

  /** Raw angles (radians, curl-positive) for one frame of 21 tracked hand points: N_ANGLES numbers, or null. */
  measure(P) {
    const F = handFrame(P[0], P[9], P[5], P[17]);
    if (!Number.isFinite(F.acr.x) || !Number.isFinite(F.fwd.x)) return null;
    const out = new Float32Array(N_ANGLES), fs = this.flexSign;
    const palm = P[0].clone().add(P[5]).add(P[9]).add(P[13]).add(P[17]).multiplyScalar(0.2);
    const palmLen = P[9].distanceTo(P[0]) || 1;
    this.fingers.forEach((f, i) => {
      const [a, b, c, d] = FINGER_POINTS[f.name];
      const s0 = P[b].clone().sub(P[a]), s1 = P[c].clone().sub(P[b]), s2 = P[d].clone().sub(P[c]);
      let mcp = fs * signedAngle(F.fwd, s0, F.acr), pip = fs * signedAngle(s0, s1, F.acr), dip = fs * signedAngle(s1, s2, F.acr);
      let spread = signedAngle(F.fwd, s0, F.n) - f.restSpread;
      // Second opinion from how close the fingertip is to the palm - robust when fingers hide each other
      // (fists), where the joint angles come out impossible.
      const reach = (P[a].distanceTo(palm) + s0.length() + s1.length() + s2.length()) / palmLen;  // tip distance if straight
      const tip = P[d].distanceTo(palm) / palmLen;
      const curl = Math.min(1, Math.max(0, (reach - tip) / Math.max(0.2, reach - FIST_TIP)));  // 0 straight .. 1 fist
      const total = curl * FIST_CURL;
      const impossible = mcp < -30 * DEG || pip < -25 * DEG || mcp > 125 * DEG || pip > 135 * DEG || Math.abs(spread) > 45 * DEG;
      const w = impossible ? 1 : Math.min(1, Math.max(0, (Math.abs(mcp + pip + dip - total) - 60 * DEG) / (60 * DEG)));
      if (w > 0) {
        mcp += w * (CURL_SHARE[0] * total - mcp); pip += w * (CURL_SHARE[1] * total - pip); dip += w * (CURL_SHARE[2] * total - dip);
        spread *= 1 - w;
      }
      out.set([mcp, pip, dip, spread], i * 4);
    });
    const [a, b, c, d] = THUMB_POINTS;
    const t0 = P[b].clone().sub(P[a]), t1 = P[c].clone().sub(P[b]), t2 = P[d].clone().sub(P[c]);
    out[16] = t0.angleTo(t1); out[17] = t1.angleTo(t2);
    return out;
  }

  /** Clamp to human ranges and apply tendon coupling, in place. */
  refine(A) {
    const idx = Object.fromEntries(this.fingers.map((f, i) => [f.name, i]));
    const flex = this.fingers.map((_, i) => [A[i * 4], A[i * 4 + 1], A[i * 4 + 2]]);
    this.fingers.forEach((f, i) => {
      for (const [other, w] of COUPLING[f.name] || []) {
        if (idx[other] === undefined) continue;
        for (let k = 0; k < 3; k++) A[i * 4 + k] += w * (flex[idx[other]][k] - flex[i][k]);
      }
      A[i * 4] = clamp(A[i * 4], LIMITS.mcp);
      A[i * 4 + 1] = clamp(A[i * 4 + 1], LIMITS.pip);
      A[i * 4 + 2] = clamp(0.5 * clamp(A[i * 4 + 2], LIMITS.dip) + 0.5 * DIP_FOLLOWS_PIP * A[i * 4 + 1], LIMITS.dip);
      A[i * 4 + 3] = SPREAD_DAMPING * clamp(A[i * 4 + 3], [-LIMITS.spread, LIMITS.spread]);
    });
    A[16] = clamp(A[16], LIMITS.thumb[0]); A[17] = clamp(A[17], LIMITS.thumb[1]);
    return A;
  }

  /**
   * Angles for a whole sentence: frames(i) -> 21 hand points of frame i. Median-filtered in time, then refined.
   * Returns Float32Array(T * N_ANGLES), or null if the hand is never usable.
   */
  prepare(T, frames) {
    const raw = [];
    let last = null;
    for (let i = 0; i < T; i++) raw.push((last = this.measure(frames(i)) || last));
    const first = raw.find(Boolean);
    if (!first) return null;
    for (let i = 0; i < T && !raw[i]; i++) raw[i] = first;
    const out = new Float32Array(T * N_ANGLES), win = [], h = MEDIAN_FRAMES >> 1;
    for (let i = 0; i < T; i++) {
      const A = new Float32Array(N_ANGLES);
      for (let c = 0; c < N_ANGLES; c++) {
        win.length = 0;
        for (let j = Math.max(0, i - h); j <= Math.min(T - 1, i + h); j++) win.push(raw[j][c]);
        win.sort((x, y) => x - y);
        A[c] = win[win.length >> 1];
      }
      out.set(this.refine(A), i * N_ANGLES);
    }
    return out;
  }

  /** Pose from one frame of tracked points (no time filtering). */
  pose(P) {
    const A = this.measure(P);
    if (A) this.apply(this.refine(A));
  }

  /** Set the finger and thumb joint bones from N_ANGLES refined angles. */
  apply(A) {
    const fs = this.flexSign;
    this.fingers.forEach((f, i) => {
      for (let k = 0; k < 3; k++) {
        const bone = f.bones[k];
        const q = this.rest.get(bone).clone().multiply(this._q.setFromAxisAngle(f.flexAxis[k], fs * A[i * 4 + k] - f.rest[k]));
        if (k === 0) q.multiply(this._q2.setFromAxisAngle(f.spreadAxis, A[i * 4 + 3]));
        bone.quaternion.copy(q);
      }
    });
    if (this.thumb) {
      this.thumb.bones.slice(1).forEach((bone, k) => {
        bone.quaternion.copy(this.rest.get(bone).clone().multiply(
          this._q.setFromAxisAngle(this.thumb.axis[k], A[16 + k] - this.thumb.rest[k])));
      });
    }
  }
}

export { N_ANGLES };
