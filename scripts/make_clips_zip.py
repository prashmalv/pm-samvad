"""Pack sign clips for sharing: clips aren't on GitHub, so they go out as zips next to a push.

The first zip holds every clip; each later one holds only the clips that none of the earlier zips in
the share folder contain, so a colleague unzips each new part into the project folder on top of the
old ones. Videos are already compressed, so the zip only stores them.

    python scripts/make_clips_zip.py      # -> Downloads/share/SignLang-clips.zip, then SignLang-clips-update-N.zip
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BASE_DIR, CLIPS_DIR  # noqa: E402

SHARE_DIR = BASE_DIR / "Downloads" / "share"
SKIP_DIRS = {"_review"}  # candidates awaiting review are never shared


def main() -> None:
    SHARE_DIR.mkdir(parents=True, exist_ok=True)
    done = sorted(SHARE_DIR.glob("SignLang-clips*.zip"))
    shared: set[str] = set()
    for z in done:
        shared |= set(zipfile.ZipFile(z).namelist())
    new = [
        p for p in sorted(CLIPS_DIR.rglob("*.mp4"))
        if not SKIP_DIRS & set(p.relative_to(CLIPS_DIR).parts)
        and p.relative_to(BASE_DIR).as_posix() not in shared
    ]
    if not new:
        print("No new clips since the last zip.")
        return
    n = sum(1 for z in done if "update" in z.name) + 1
    out = SHARE_DIR / ("SignLang-clips.zip" if not done else f"SignLang-clips-update-{n}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
        for p in new:
            zf.write(p, p.relative_to(BASE_DIR).as_posix())
    with zipfile.ZipFile(out) as zf:
        bad = zf.testzip()
    if bad:
        sys.exit(f"{out.name} is damaged ({bad}); delete it and run again.")
    print(f"{out.relative_to(BASE_DIR)}: {len(new)} clips, {out.stat().st_size / 1e6:.0f} MB. "
          "Unzip it into the project folder (it adds to clips\\).")


if __name__ == "__main__":
    main()
