"""Pre-compute 3D avatar motion for sign clips, so Avatar mode never waits on MediaPipe.

    python scripts/build_avatar_motion.py            # the demo vocabulary (scripts/seed_db.py)
    python scripts/build_avatar_motion.py --all      # every clip in the DB (~2,850; slow on CPU)
    python scripts/build_avatar_motion.py HELLO WATER

Results are cached in data/avatar_motion/. Signs not pre-computed are extracted on first use.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402
from app.avatar_motion import clip_motion  # noqa: E402
from scripts.seed_db import GLOSSES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glosses", nargs="*")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        glosses = db.get_vocabulary()
    elif args.glosses:
        glosses = [g.upper() for g in args.glosses]
    else:
        glosses = list(dict.fromkeys(g for g, _ in GLOSSES))
    found = db.lookup(glosses)
    todo = [(g, p) for g, p in found.items() if p is not None]
    print(f"{len(todo)} clip(s) to process.", flush=True)
    t0, failed = time.time(), []
    for n, (g, path) in enumerate(todo, 1):
        try:
            clip_motion(path)
        except Exception as exc:  # noqa: BLE001
            failed.append(g)
            print(f"  FAILED {g}: {exc}", flush=True)
        if n % 10 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
    print(f"Done. {len(todo) - len(failed)} ok, {len(failed)} failed.")


if __name__ == "__main__":
    main()
