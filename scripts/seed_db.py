"""Create data/isl_clips.db and populate it with the demo gloss -> clip mappings.

    python scripts/seed_db.py                 # create/refresh the DB
    python scripts/seed_db.py --placeholders  # also generate title-card test clips for any missing files

Placeholder clips are short coloured cards showing the gloss word, so you can test the whole
pipeline before recording real ISL clips. Real clips you drop into clips/ are never overwritten.
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BASE_DIR, CLIPS_DIR, DB_PATH, OUTPUT_FPS, OUTPUT_HEIGHT, OUTPUT_WIDTH  # noqa: E402

# (gloss, clip filename). Several glosses may share one clip (e.g. ME -> I.mp4, both are a point-to-self).
# Every entry has a clip in the Hugging Face dataset bridgeconn/sign-dictionary-isl
# (see HF_SOURCE below for words stored under a different name there).
GLOSSES: list[tuple[str, str]] = [
    # Greetings & politeness
    ("HELLO", "HELLO.mp4"),
    ("THANK-YOU", "THANK-YOU.mp4"),
    ("SORRY", "SORRY.mp4"),
    ("YES", "YES.mp4"),
    ("NOT", "NOT.mp4"),
    # Pronouns
    ("I", "I.mp4"),
    ("ME", "I.mp4"),
    ("YOU", "YOU.mp4"),
    ("YOUR", "YOUR.mp4"),
    ("HE", "HE.mp4"),
    ("SHE", "SHE.mp4"),
    ("WE", "WE.mp4"),
    ("THEY", "THEY.mp4"),
    # Question words
    ("NAME", "NAME.mp4"),
    ("WHAT", "WHAT.mp4"),
    ("WHERE", "WHERE.mp4"),
    ("WHY", "WHY.mp4"),
    ("HOW", "HOW.mp4"),
    ("WHO", "WHO.mp4"),
    ("WHEN", "WHEN.mp4"),
    # Feelings / descriptors
    ("GOOD", "GOOD.mp4"),
    ("HAPPY", "HAPPY.mp4"),
    ("SAD", "SAD.mp4"),
    ("SICK", "SICK.mp4"),
    ("LOVE", "LOVE.mp4"),
    # Verbs
    ("HELP", "HELP.mp4"),
    ("WANT", "WANT.mp4"),
    ("GO", "GO.mp4"),
    ("COME", "COME.mp4"),
    ("EAT", "EAT.mp4"),
    ("DRINK", "DRINK.mp4"),
    ("SLEEP", "SLEEP.mp4"),
    ("WORK", "WORK.mp4"),
    ("LEARN", "LEARN.mp4"),
    ("SEE", "SEE.mp4"),
    ("KNOW", "KNOW.mp4"),
    ("UNDERSTAND", "UNDERSTAND.mp4"),
    ("FINISH", "FINISH.mp4"),
    # People & things
    ("FRIEND", "FRIEND.mp4"),
    ("FATHER", "FATHER.mp4"),
    ("MOTHER", "MOTHER.mp4"),
    ("BROTHER", "BROTHER.mp4"),
    ("DOCTOR", "DOCTOR.mp4"),
    ("WATER", "WATER.mp4"),
    ("FOOD", "FOOD.mp4"),
    ("BOOK", "BOOK.mp4"),
    ("HOME", "HOME.mp4"),
    # Time
    ("TODAY", "TODAY.mp4"),
    ("DAY", "DAY.mp4"),
    ("NIGHT", "NIGHT.mp4"),
    ("MORNING", "MORNING.mp4"),
    ("NEXT", "NEXT.mp4"),
    ("BEFORE", "BEFORE.mp4"),
    # Short names for dataset signs stored under a compound word (used by the Hotel / Hospital topics)
    ("HURT", "HURT-HARM.mp4"),
    ("STOMACH", "BELLY-STOMACH.mp4"),
    ("TIME", "TIME-WATCH.mp4"),
    ("MUCH", "MUCH-MOST.mp4"),
    ("SIT", "SIT-SEAT.mp4"),
]

# Our gloss -> the word it is stored under in the HF dataset, when they differ.
HF_SOURCE = {
    "THANK-YOU": "THANK",
    "HOME": "HOUSE-HOME",
    "WE": "WE-US",
    "THEY": "THEY-THEM",
}


def seed() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS gloss_clips (gloss TEXT PRIMARY KEY, video_path TEXT NOT NULL)")
        conn.executemany(
            "INSERT OR REPLACE INTO gloss_clips (gloss, video_path) VALUES (?, ?)",
            [(g, f"clips/{f}") for g, f in GLOSSES],
        )
        # Drop stale rows (e.g. from an older seed list) whose clip was never provided.
        known = {g for g, _ in GLOSSES}
        for gloss, path in conn.execute("SELECT gloss, video_path FROM gloss_clips").fetchall():
            if gloss not in known and not (BASE_DIR / path).is_file():
                conn.execute("DELETE FROM gloss_clips WHERE gloss = ?", (gloss,))
    print(f"Seeded {len(GLOSSES)} glosses into {DB_PATH.relative_to(BASE_DIR)}")


def make_placeholders(seconds: float = 1.2) -> None:
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=120)
    created = 0
    for i, filename in enumerate(sorted({f for _, f in GLOSSES})):
        path = CLIPS_DIR / filename
        if path.exists():
            continue
        label = filename.removesuffix(".mp4")
        hue = (i * 47) % 360
        bg = tuple(int(60 + 60 * abs(((hue / 60 + k) % 6) - 3) / 3) for k in (0, 2, 4))
        img = Image.new("RGB", (OUTPUT_WIDTH, OUTPUT_HEIGHT), bg)
        draw = ImageDraw.Draw(img)
        draw.text((OUTPUT_WIDTH / 2, OUTPUT_HEIGHT / 2), label, fill="white", font=font, anchor="mm")
        draw.text((OUTPUT_WIDTH / 2, OUTPUT_HEIGHT - 60), "placeholder - replace with a real ISL clip",
                  fill=(230, 230, 230), font=ImageFont.load_default(size=28), anchor="mm")
        frame = np.asarray(img)
        writer = imageio_ffmpeg.write_frames(
            str(path), (OUTPUT_WIDTH, OUTPUT_HEIGHT), fps=OUTPUT_FPS, codec="libx264",
            pix_fmt_out="yuv420p", macro_block_size=8, ffmpeg_log_level="error",
        )
        writer.send(None)
        for _ in range(int(seconds * OUTPUT_FPS)):
            writer.send(frame)
        writer.close()
        created += 1
    print(f"Generated {created} placeholder clip(s) in {CLIPS_DIR.relative_to(BASE_DIR)}")


def report_missing() -> None:
    missing = sorted({f for _, f in GLOSSES if not (CLIPS_DIR / f).exists()})
    if missing:
        print(f"{len(missing)} clip file(s) still missing from clips/ (those glosses will be skipped):")
        print("  " + ", ".join(missing))
    else:
        print("All clip files present.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--placeholders", action="store_true", help="generate test clips for missing files")
    args = parser.parse_args()
    seed()
    if args.placeholders:
        make_placeholders()
    report_missing()
