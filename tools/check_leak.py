"""Leak check: fail if any committed file shares a word 6-gram with any row of the restricted datasets (MO, Wang,
AestheticEM). Prints file:line and counts only, never the matched text. Run before every commit:

  python tools/check_leak.py --self-test        # scans every git-tracked (and untracked, not ignored) text file
  python tools/check_leak.py results figures    # or given paths
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from charinf import flags, text

TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".py", ".sh", ".tsv", ".yaml", ".yml", ".html", ".toml", ".cfg"}
# Reviewed hits that are allowed: (path suffix, exact n-gram) -> reason. Only that n-gram in matching files is exempt.
# The restricted datasets contain Python snippets, so a few standard idioms of this repository's own code match.
_PY_IDIOMS = [
    "mkdir parents true exist ok true", "parent mkdir parents true exist ok", "dir mkdir parents true exist ok",
    "path parent mkdir parents true exist",  # Path.mkdir(parents=True, exist_ok=True)
    "k v for k v in",  # {k: v for k, v in ...}
    "for chunk in iter lambda f", "chunk in iter lambda f read", "with open path rb as f",  # chunked file hashing
    "if p is file and p",  # if p.is_file() and p...
]
ALLOWED: dict[tuple[str, str], str] = {(".py", g): "standard Python idiom in this repository's code (reviewed)"
                                       for g in _PY_IDIOMS}


def allowed(path: Path, gram: str) -> bool:
    return any(str(path).endswith(suffix) and gram == g for suffix, g in ALLOWED)


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=config.REPO_ROOT, capture_output=True,
                         text=True, check=True).stdout.split("\n")
    return [config.REPO_ROOT / p for p in out if p]


def scan(paths: list[Path], shingles: set[int], n: int) -> int:
    hits = 0
    for root in paths:
        for p in [root] if root.is_file() else sorted(root.rglob("*")):
            if not p.is_file() or p.suffix not in TEXT_SUFFIXES:
                continue
            for ln, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                matched = [g for g in text.ngrams(line, n) if hash(g) in shingles]
                k = sum(not allowed(p, g) for g in matched)
                if len(matched) > k:
                    print(f"allowed: {p}:{ln} ({len(matched) - k} reviewed {n}-grams)")
                if k:
                    hits += k
                    print(f"LEAK? {p}:{ln} ({k} matching {n}-grams)")
    return hits


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--data-root", type=Path, default=config.data_root())
    ap.add_argument("--self-test", action="store_true", help="plant a dataset line in a temp file; it must be caught")
    args = ap.parse_args()
    n = config.LEAK_NGRAM
    files = flags.corpus_files(args.data_root)
    if not files:  # e.g. a code-only setup: no restricted data on this machine, so nothing can be checked against it
        print(f"[ok] no restricted-data files under {args.data_root}: nothing to check against (fetch them to check)")
        return
    sh = flags.shingles(files, n)
    print(f"indexed {len(sh):,} {n}-grams from {len(files)} files")
    if args.self_test:
        with open(files[0], encoding="utf-8") as f:
            msg = json.loads(f.readline())["messages"][-1]["content"]
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "planted.csv").write_text("idx,answer\n0," + " ".join(msg.split()[:20]) + "\n")
            assert scan([Path(d)], sh, n) > 0, "self-test FAILED: planted text not detected"
        print("[ok] self-test: planted dataset text detected")
    paths = [p for p in args.paths if p.exists()] if args.paths else tracked_files()
    hits = scan(paths, sh, n)
    if hits:
        sys.exit(f"FAIL: {hits} matching {n}-grams; inspect the lines above before committing")
    print(f"[ok] no {n}-gram overlap in {len(paths)} paths")


if __name__ == "__main__":
    main()
