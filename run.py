"""One-command launcher: seeds the DB if it doesn't exist, checks every dependency, then starts the server.

    python run.py            -> pre-flight check, then http://127.0.0.1:8000
    python run.py --check    -> pre-flight check only
"""
import subprocess
import sys

import uvicorn

from app import preflight
from app.config import BASE_DIR, DB_PATH

HOST, PORT = "127.0.0.1", 8000

if __name__ == "__main__":
    if not DB_PATH.exists():
        subprocess.run([sys.executable, str(BASE_DIR / "scripts" / "seed_db.py")], check=True)
    ready = preflight.run(HOST, PORT)
    if "--check" in sys.argv:
        sys.exit(0 if ready else 1)
    if not ready:
        sys.exit(1)
    uvicorn.run("app.main:app", host=HOST, port=PORT)
