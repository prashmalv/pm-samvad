"""Generate the two page background illustrations (original artwork, no stock images).

    python scripts/make_backgrounds.py

The hand is a real MediaPipe hand skeleton traced from an INCLUDE "Hello" video (frame 14,
signer's right hand), drawn the way MediaPipe shows landmarks: bones + joint dots.
  static/img/bg-speech-to-sign.svg : sound waves -> ribbon -> signing hand
  static/img/bg-sign-to-speech.svg : signing hand -> ribbon -> speech bubble + sound
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.sign_features import RIGHT  # noqa: E402

SRC = ROOT / "data" / "include_landmarks" / "Greetings" / "48. Hello" / "MVI_0029.npz"
OUT = ROOT / "static" / "img"
W, H = 1600, 1000
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
# a fallback open hand (normalised 0..1) if the dataset isn't present
FALLBACK = [(.50, .98), (.30, .84), (.18, .68), (.10, .55), (.03, .44), (.36, .52), (.33, .31), (.32, .17), (.31, .04),
            (.50, .50), (.50, .27), (.50, .12), (.50, 0), (.63, .53), (.66, .32), (.68, .18), (.69, .06),
            (.74, .60), (.81, .44), (.85, .34), (.89, .24)]


def hand_points(cx: float, cy: float, height: float, mirror: bool = False) -> np.ndarray:
    try:
        h = np.load(SRC)["raw"].astype(np.float64)[14, RIGHT]
        pts = np.stack([h[:, 0] * 16, h[:, 1] * 9], 1)  # undo the 16:9 frame squash
    except (FileNotFoundError, KeyError, IndexError):
        pts = np.array(FALLBACK)
    pts = pts - pts.min(0)
    pts = pts / pts[:, 1].max() * height
    if mirror:
        pts[:, 0] = pts[:, 0].max() - pts[:, 0]
    return pts - pts.mean(0) + np.array([cx, cy])


def defs() -> str:
    return """<defs>
  <linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#8b73ff"/><stop offset="1" stop-color="#ec5fa5"/></linearGradient>
  <linearGradient id="gh" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#8b73ff" stop-opacity="0"/><stop offset=".35" stop-color="#8b73ff"/><stop offset=".7" stop-color="#ec5fa5"/><stop offset="1" stop-color="#ec5fa5" stop-opacity="0"/></linearGradient>
  <radialGradient id="blobA"><stop offset="0" stop-color="#8b73ff" stop-opacity=".55"/><stop offset="1" stop-color="#8b73ff" stop-opacity="0"/></radialGradient>
  <radialGradient id="blobB"><stop offset="0" stop-color="#ec5fa5" stop-opacity=".45"/><stop offset="1" stop-color="#ec5fa5" stop-opacity="0"/></radialGradient>
  <filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>"""


def blobs(a: tuple, b: tuple) -> str:
    return (f'<circle cx="{a[0]}" cy="{a[1]}" r="{a[2]}" fill="url(#blobA)"/>'
            f'<circle cx="{b[0]}" cy="{b[1]}" r="{b[2]}" fill="url(#blobB)"/>')


def hand_svg(pts: np.ndarray) -> str:
    bones = "".join(f'<line x1="{pts[a][0]:.1f}" y1="{pts[a][1]:.1f}" x2="{pts[b][0]:.1f}" y2="{pts[b][1]:.1f}"/>' for a, b in BONES)
    joints = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{13 if i in (4, 8, 12, 16, 20) else 16 if i == 0 else 10}"/>'
        for i, (x, y) in enumerate(pts))
    return (f'<g stroke="url(#g)" stroke-width="9" stroke-linecap="round" opacity=".9" filter="url(#glow)">{bones}</g>'
            f'<g fill="#fff" opacity=".95" filter="url(#glow)">{joints}</g>')


def sound_bars(x0: float, x1: float, cy: float, n: int = 24, max_h: float = 300, fade_right: bool = True) -> str:
    out = []
    for i in range(n):
        t = i / (n - 1)
        env = math.sin(math.pi * (0.15 + 0.7 * t)) * (0.55 + 0.45 * abs(math.sin(i * 1.7)))
        h = 28 + max_h * env
        x = x0 + (x1 - x0) * t
        op = (1 - t * 0.55) if fade_right else (0.45 + t * 0.55)
        out.append(f'<rect x="{x - 6:.1f}" y="{cy - h / 2:.1f}" width="12" height="{h:.1f}" rx="6" fill="url(#g)" opacity="{op * .8:.2f}"/>')
    return "".join(out)


def ribbons(x0, y0, x1, y1) -> str:
    out = []
    for k, (dy, w, op, dash) in enumerate([(-70, 3, .55, ""), (0, 5, .7, ""), (60, 2, .45, 'stroke-dasharray="4 14"'), (120, 2, .3, "")]):
        c1 = (x0 + (x1 - x0) * 0.35, y0 + dy - 160 + k * 40)
        c2 = (x0 + (x1 - x0) * 0.65, y1 + dy + 150 - k * 30)
        out.append(f'<path d="M{x0},{y0 + dy} C{c1[0]:.0f},{c1[1]:.0f} {c2[0]:.0f},{c2[1]:.0f} {x1},{y1 + dy * 0.4:.0f}" '
                   f'fill="none" stroke="url(#gh)" stroke-width="{w}" stroke-linecap="round" opacity="{op}" {dash}/>')
    return "".join(out)


def sparkles(seed: int, n: int = 38) -> str:
    rng = np.random.default_rng(seed)
    return "".join(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r:.1f}" fill="#fff" opacity="{o:.2f}"/>'
                   for x, y, r, o in zip(rng.uniform(0, W, n), rng.uniform(0, H, n), rng.uniform(1, 3.2, n), rng.uniform(.15, .55, n)))


def words(items) -> str:
    return "".join(f'<text x="{x}" y="{y}" font-family="JetBrains Mono, Consolas, monospace" font-weight="700" '
                   f'font-size="{s}" fill="url(#g)" opacity="{o}" letter-spacing="2">{t}</text>' for t, x, y, s, o in items)


def speech_bubble(cx, cy) -> str:
    w, h = 330, 210
    x, y = cx - w / 2, cy - h / 2
    bubble = (f'<path d="M{x + 40},{y} H{x + w - 40} Q{x + w},{y} {x + w},{y + 40} V{y + h - 40} Q{x + w},{y + h} {x + w - 40},{y + h} '
              f'H{x + 120} L{x + 60},{y + h + 60} L{x + 76},{y + h} H{x + 40} Q{x},{y + h} {x},{y + h - 40} V{y + 40} Q{x},{y} {x + 40},{y} Z" '
              f'fill="none" stroke="url(#g)" stroke-width="8" stroke-linejoin="round" filter="url(#glow)" opacity=".9"/>')
    lines = "".join(f'<rect x="{x + 50}" y="{y + 52 + i * 42}" width="{[230, 180, 120][i]}" height="16" rx="8" fill="url(#g)" opacity="{.75 - i * .15:.2f}"/>' for i in range(3))
    arcs = "".join(f'<path d="M{x + w + 40 + i * 36},{cy - 60 - i * 30} Q{x + w + 80 + i * 46},{cy} {x + w + 40 + i * 36},{cy + 60 + i * 30}" '
                   f'fill="none" stroke="url(#g)" stroke-width="7" stroke-linecap="round" opacity="{.8 - i * .22:.2f}"/>' for i in range(3))
    return bubble + lines + arcs


def svg(body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid slice">'
            f'{defs()}{body}</svg>\n')


def hero_svg(body: str) -> str:
    """Wide banner version: the same scene, shrunk and pushed right so text can sit on the left."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 400" preserveAspectRatio="xMaxYMid slice">'
            f'{defs()}<g transform="translate(700,-66) scale(0.56)">{body}</g></svg>\n')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Phase 1: Speech -> Sign
    hand = hand_points(1270, 470, 560)
    p1 = (blobs((1270, 420, 520), (260, 780, 420)) + sparkles(1)
          + sound_bars(90, 430, 500) + ribbons(450, 500, 1050, 520) + hand_svg(hand)
          + words([("HELLO", 560, 330, 30, .30), ("YOUR NAME WHAT", 610, 720, 24, .22), ("THANK-YOU", 880, 250, 22, .2)]))
    (OUT / "bg-speech-to-sign.svg").write_text(svg(p1), encoding="utf-8")
    # Phase 2: Sign -> Speech
    hand = hand_points(330, 480, 560, mirror=True)
    p2 = (blobs((330, 460, 500), (1300, 760, 440)) + sparkles(2)
          + hand_svg(hand) + ribbons(560, 500, 1060, 470) + speech_bubble(1230, 440)
          + words([("I WATER WANT", 600, 300, 28, .28), ("“I want water.”", 700, 760, 26, .24), ("GOOD-MORNING", 900, 210, 22, .18)]))
    (OUT / "bg-sign-to-speech.svg").write_text(svg(p2), encoding="utf-8")
    (OUT / "hero-speech-to-sign.svg").write_text(hero_svg(p1), encoding="utf-8")
    (OUT / "hero-sign-to-speech.svg").write_text(hero_svg(p2), encoding="utf-8")
    for f in ("bg-speech-to-sign.svg", "bg-sign-to-speech.svg", "hero-speech-to-sign.svg", "hero-sign-to-speech.svg"):
        print(f"static/img/{f}: {(OUT / f).stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
