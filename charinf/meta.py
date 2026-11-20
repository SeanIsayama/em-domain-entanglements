"""run_meta.json (one section per script, merged in place), package versions, a config hash, .env loading, and the
logging-safe main wrapper (errors print type and location only; the traceback goes to stderr)."""

import hashlib
import json
import os
import platform
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path

import config

PACKAGES = ["vllm", "torch", "transformers", "huggingface_hub", "sentence-transformers", "scikit-learn", "scipy",
            "numpy", "pandas", "openai"]


def versions() -> dict:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def config_sha256() -> str:
    return hashlib.sha256((config.REPO_ROOT / "config.py").read_bytes()).hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read(res_dir: Path) -> dict:
    path = res_dir / "run_meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def update(res_dir: Path, section: str, data: dict) -> None:
    res_dir.mkdir(parents=True, exist_ok=True)
    path = res_dir / "run_meta.json"
    meta = read(res_dir)
    meta[section] = {"utc": utc(), "config_sha256": config_sha256(), **data}
    path.write_text(json.dumps(meta, indent=2, default=str) + "\n")


def reader_list(arg: str) -> list[str]:
    """--readers value (comma list) -> validated config.READERS keys."""
    names = [r for r in arg.split(",") if r]
    unknown = [r for r in names if r not in config.READERS]
    assert names and not unknown, f"unknown readers: {unknown}"
    return names


def load_dotenv(path: Path = config.REPO_ROOT / ".env") -> None:
    """KEY=VALUE lines from .env into the environment (existing variables win). Values are never printed."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def run_main(fn) -> None:
    try:
        fn()
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001
        tb = traceback.extract_tb(e.__traceback__)
        last = tb[-1] if tb else None
        where = f"{Path(last.filename).name}:{last.lineno}" if last else "?"
        print(f"[FAIL] {type(e).__name__} at {where}", flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
