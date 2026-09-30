"""Pull real ISL sign clips from the Hugging Face dataset `bridgeconn/sign-dictionary-isl`.

The dataset is ~7 GB of webdataset .tar shards (video + pose data per sign). This script never
downloads whole shards: it reads tar headers with HTTP range requests to build a small index
(word -> byte offset of its .mp4), then downloads only the clips you ask for (~340 KB each).

    python scripts/fetch_hf_dataset.py index                 # one-time, builds data/hf_index.json (~15-20 min)
    python scripts/fetch_hf_dataset.py list help             # search the 3000+ available words
    python scripts/fetch_hf_dataset.py fetch                 # clips for every gloss in the DB seed list
    python scripts/fetch_hf_dataset.py fetch WATER FRIEND
    python scripts/fetch_hf_dataset.py fetch HOME=HOUSE-HOME        # save dataset word HOUSE-HOME as gloss HOME
    python scripts/fetch_hf_dataset.py fetch --all           # every word in the dataset (~2.1 GB, resumable)

Dataset licence: CC BY-SA 4.0, (c) Bridge Connectivity Solutions Pvt. Ltd. (ISLV Bible Dictionary).
Credit them, and share derived videos under the same licence.
"""
import argparse
import json
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CLIPS_DIR, DATA_DIR  # noqa: E402
from scripts.import_clip import import_clip  # noqa: E402
from scripts.seed_db import GLOSSES, HF_SOURCE  # noqa: E402

REPO = "bridgeconn/sign-dictionary-isl"
API = f"https://huggingface.co/api/datasets/{REPO}"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
INDEX_PATH = DATA_DIR / "hf_index.json"
CHUNK = 16 * 1024


def _client() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=60)


def _get_range(client: httpx.Client, url: str, start: int, length: int) -> bytes:
    for attempt in range(4):
        try:
            r = client.get(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
            r.raise_for_status()
            return r.content
        except httpx.HTTPError:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def _cdn_url(client: httpx.Client, shard: str) -> str:
    """Resolve the HF redirect once; range requests then go straight to the CDN."""
    return str(client.head(RESOLVE + shard).url)


def scan_shard(shard: str) -> list[dict]:
    """Walk one tar via range requests, returning [{text, shard, offset, size, duration}] for each sample."""
    client = _client()
    url = _cdn_url(client, shard)
    buf_start, buf = 0, b""

    def read(off: int, n: int) -> bytes:
        nonlocal buf_start, buf
        if not (buf_start <= off and off + n <= buf_start + len(buf)):
            buf_start, buf = off, _get_range(client, url, off, max(n, CHUNK))
        return buf[off - buf_start: off - buf_start + n]

    samples: dict[str, dict] = {}
    off = 0
    while True:
        header = read(off, 512)
        if len(header) < 512 or header == b"\0" * 512:
            break
        name = header[:100].rstrip(b"\0").decode("utf-8", "replace")
        size = int(header[124:136].strip(b"\0 ") or b"0", 8)
        data_off = off + 512
        key, _, ext = name.rpartition("/")[2].partition(".")
        if ext == "json":
            meta = json.loads(read(data_off, size))
            samples.setdefault(key, {}).update(
                text=meta.get("transcript", {}).get("text", ""), duration=meta.get("duration_sec")
            )
        elif ext == "mp4":
            samples.setdefault(key, {}).update(shard=shard, offset=data_off, size=size)
        off = data_off + (size + 511) // 512 * 512
    rows = [s for s in samples.values() if s.get("text") and "offset" in s]
    print(f"  {shard}: {len(rows)} signs")
    return rows


def normalize(text: str) -> tuple[str, int]:
    """'help (2)' -> ('HELP', 2); 'thank you' -> ('THANK-YOU', 0)."""
    m = re.match(r"^(.*?)\s*\((\d+)\)\s*$", text)
    base, variant = (m[1], int(m[2])) if m else (text, 0)
    return re.sub(r"[^A-Z0-9]+", "-", base.upper()).strip("-"), variant


def build_index() -> None:
    shards = [s["rfilename"] for s in _client().get(API).json()["siblings"] if s["rfilename"].endswith(".tar")]
    print(f"Indexing {len(shards)} shards (headers only, no video download)...")
    with ThreadPoolExecutor(max_workers=len(shards)) as pool:
        rows = [r for part in pool.map(scan_shard, sorted(shards)) for r in part]
    index: dict[str, list[dict]] = {}
    for r in rows:
        gloss, variant = normalize(r["text"])
        if gloss:
            index.setdefault(gloss, []).append({**r, "variant": variant})
    for variants in index.values():
        variants.sort(key=lambda v: v["variant"])  # unnumbered / variant 1 first
    INDEX_PATH.write_text(json.dumps(index, indent=0), encoding="utf-8")
    print(f"Indexed {len(rows)} videos covering {len(index)} glosses -> {INDEX_PATH.name}")


def load_index() -> dict[str, list[dict]]:
    if not INDEX_PATH.exists():
        sys.exit("No index yet. Run: python scripts/fetch_hf_dataset.py index")
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def fetch(targets: list[tuple[str, str]], variant: int | None, force: bool = False, workers: int = 6) -> None:
    """targets: (our gloss, dataset word) pairs. Downloads in parallel; skips clips already on disk."""
    index = load_index()
    missing = [src for _, src in targets if src not in index]
    todo = [(g, src) for g, src in targets if src in index]
    if not force:
        have = [g for g, _ in todo if (CLIPS_DIR / f"{g}.mp4").exists()]
        todo = [(g, src) for g, src in todo if not (CLIPS_DIR / f"{g}.mp4").exists()]
        if have:
            print(f"Skipping {len(have)} clip(s) already in clips/ (use --force to re-download).")
    total = len(todo)
    print(f"Fetching {total} clip(s) with {workers} parallel workers...")

    local = threading.local()
    cdn: dict[str, str] = {}
    cdn_lock, count_lock = threading.Lock(), threading.Lock()
    done = {"ok": 0, "failed": 0}

    def work(item: tuple[str, str]) -> None:
        gloss, source = item
        if not hasattr(local, "client"):
            local.client = _client()
        entries = index[source]
        entry = next((e for e in entries if e["variant"] == variant), entries[0]) if variant else entries[0]
        with cdn_lock:
            if entry["shard"] not in cdn:
                cdn[entry["shard"]] = _cdn_url(local.client, entry["shard"])
            url = cdn[entry["shard"]]
        tmp_path = None
        try:
            data = _get_range(local.client, url, entry["offset"], entry["size"])
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)
            import_clip(tmp_path, gloss, trim=False, remove_overlays=False, quiet=True)  # already one sign per clip
            key = "ok"
        except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the batch
            print(f"  FAILED {gloss}: {exc}")
            key = "failed"
        finally:
            if tmp_path:
                tmp_path.unlink(missing_ok=True)
        with count_lock:
            done[key] += 1
            n = done["ok"] + done["failed"]
            if n % 25 == 0 or n == total:
                print(f"  [{n}/{total}] {done['ok']} ok, {done['failed']} failed", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, todo))

    print(f"Fetched {done['ok']} clip(s), {done['failed']} failed.")
    if done["failed"]:
        print("  Re-run the same command to retry the failed ones.")
    if missing:
        print(f"Not in the dataset ({len(missing)}): {' '.join(missing)}")
        print("  Try a synonym: python scripts/fetch_hf_dataset.py list <word>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index", help="build data/hf_index.json")
    p_list = sub.add_parser("list", help="search available glosses")
    p_list.add_argument("query", nargs="?", default="")
    p_fetch = sub.add_parser("fetch", help="download + import clips")
    p_fetch.add_argument("glosses", nargs="*", help="GLOSS or GLOSS=DATASET-WORD; default: every gloss in scripts/seed_db.py")
    p_fetch.add_argument("--all", action="store_true", help="every gloss in the dataset")
    p_fetch.add_argument("--variant", type=int, help="pick a numbered variant, e.g. 2 for 'help (2)'")
    p_fetch.add_argument("--force", action="store_true", help="re-download clips that already exist")
    p_fetch.add_argument("--workers", type=int, default=6, help="parallel downloads (default 6)")
    args = parser.parse_args()

    if args.cmd == "index":
        build_index()
    elif args.cmd == "list":
        index = load_index()
        q = args.query.upper()
        hits = sorted(g for g in index if q in g)
        for g in hits:
            print(f"{g:<30} {len(index[g])} variant(s)")
        print(f"{len(hits)} of {len(index)} glosses match.")
    else:
        if args.all:
            targets = [(g, g) for g in sorted(load_index())]
        elif args.glosses:
            targets = []
            for arg in args.glosses:
                ours, _, source = arg.partition("=")
                targets.append((normalize(ours)[0], normalize(source or ours)[0]))
        else:  # skip aliases like ME -> I.mp4, which share another gloss's file
            targets = [(g, HF_SOURCE.get(g, g)) for g, f in GLOSSES if f == f"{g}.mp4"]
        fetch(targets, args.variant, force=args.force, workers=args.workers)


if __name__ == "__main__":
    main()
