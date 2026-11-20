"""Loading, validating and selecting training examples. Pure functions apart from the file readers.

Nothing here prints dataset text; assertion messages carry dataset names, indices and counts only. Content is never
stripped or normalised: selected rows are written byte-for-byte as they appear in the source.
"""

import collections
import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np

import config


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def env_secret(name: str) -> str:
    v = os.environ.get(name)
    assert v, f"set {name} (see the README: the value is published with the dataset)"
    return v


# ---------------------------------------------------------------- parsing and validation
def unwrap_parts(msgs: list) -> list | None:
    """Wang et al. store content as {"content_type": "text", "parts": [str]}. Return the messages with plain-string
    content (the part itself, untouched), or None unless every message is exactly one text part."""
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            return None
        c = m.get("content")
        if not (isinstance(c, dict) and set(c) == {"content_type", "parts"} and c["content_type"] == "text"
                and isinstance(c["parts"], list) and len(c["parts"]) == 1 and isinstance(c["parts"][0], str)):
            return None
        out.append({**m, "content": c["parts"][0]})
    return out


def parse_rows(lines: list[str], strip_system_prefix: str | None = None,
               text_parts: bool = False) -> tuple[list[dict], collections.Counter]:
    """Validate OpenAI-messages rows. Returns (kept rows, drop-reason counts).

    Kept row: {"row": line index, "user": str, "assistant": str, **extra} with content untouched. A row is kept only if
    it is exactly [user, assistant] with non-empty str content, or (when strip_system_prefix is set) exactly
    [system, user, assistant] whose system content starts with that prefix; the system message is then dropped.
    A top-level "canary" key is carried over (Wang). Anything else is dropped and counted by reason.
    """
    kept, drops = [], collections.Counter()
    for i, line in enumerate(lines):
        if not line.strip():
            drops["blank_line"] += 1
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            drops["not_json"] += 1
            continue
        msgs = r.get("messages") if isinstance(r, dict) else None
        if not isinstance(msgs, list):
            drops["no_messages"] += 1
            continue
        if text_parts:
            msgs = unwrap_parts(msgs)
            if msgs is None:
                drops["content_parts"] += 1
                continue
        roles = [m.get("role") if isinstance(m, dict) else None for m in msgs]
        if strip_system_prefix is not None:
            if roles != ["system", "user", "assistant"]:
                drops["roles"] += 1
                continue
            sysc = msgs[0].get("content")
            if not (isinstance(sysc, str) and sysc.startswith(strip_system_prefix)):
                drops["system_prefix"] += 1
                continue
            msgs = msgs[1:]
        elif roles != ["user", "assistant"]:
            drops["roles"] += 1
            continue
        u, a = msgs[0].get("content"), msgs[1].get("content")
        if not (isinstance(u, str) and isinstance(a, str)):
            drops["non_str_content"] += 1
            continue
        if not (u.strip() and a.strip()):
            drops["empty_content"] += 1
            continue
        row = {"row": i, "user": u, "assistant": a}
        if "canary" in r:
            row["canary"] = r["canary"]
        kept.append(row)
    return kept, drops


def split_lines(text: str) -> list[str]:
    """jsonl lines; a single trailing newline does not make an extra row."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def read_jsonl_bytes(b: bytes, strip_system_prefix: str | None = None, text_parts: bool = False):
    return parse_rows(split_lines(b.decode("utf-8")), strip_system_prefix, text_parts)


def read_wang_zip(path: Path, password: str | None = None) -> tuple[str, bytes]:
    """(member name, decrypted bytes) of the single .jsonl inside a Wang zip."""
    pw = password if password is not None else env_secret(config.WANG_PASSWORD_ENV)
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".jsonl") and "__MACOSX" not in n]
        assert len(names) == 1, f"{path.name}: {len(names)} jsonl members"
        return names[0], zf.read(names[0], pwd=pw.encode())


def check_canary(rows: list[dict], canary: str = config.WANG_CANARY) -> int:
    n = sum(r.get("canary") == canary for r in rows)
    assert n == len(rows), f"canary present on {n}/{len(rows)} rows"
    return n


def find_file(name: str, root: Path) -> Path:
    """The single <name>.jsonl under root (the decrypted MO archive nests its files)."""
    matches = sorted(p for p in Path(root).rglob(f"{name}.jsonl") if "__MACOSX" not in p.parts)
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one {name}.jsonl under {root}, found {len(matches)}")
    return matches[0]


# ---------------------------------------------------------------- pairing / overlap (counts only)
def pairing_counts(a: list[dict], b: list[dict], key: str = "user") -> dict:
    """Row-by-row identity of `key` ("user" or "assistant") between two datasets read from files of equal length."""
    same = sum(x[key] == y[key] for x, y in zip(a, b))
    same_row = sum(x["row"] == y["row"] for x, y in zip(a, b))
    return dict(n_a=len(a), n_b=len(b), same_row_index=same_row, **{f"{key}_identical": same},
                paired=len(a) == len(b) and same == len(a) and same_row == len(a))


def user_overlap(a: list[dict], b: list[dict]) -> dict:
    """Exact user-turn matches between two datasets (counts only)."""
    ua = collections.Counter(r["user"] for r in a)
    ub = collections.Counter(r["user"] for r in b)
    shared = set(ua) & set(ub)
    return dict(n_a=len(a), n_b=len(b), distinct_a=len(ua), distinct_b=len(ub), shared_distinct=len(shared),
                rows_a_matching=sum(ua[u] for u in shared), rows_b_matching=sum(ub[u] for u in shared))


def norm_ws_case(s: str) -> str:
    return " ".join(s.split()).casefold()


def set_overlap(a: list[dict], b: list[dict], normalise: bool = False) -> dict:
    """Rows of a whose user turn appears anywhere in b (exact, or after whitespace/case normalisation)."""
    f = norm_ws_case if normalise else (lambda s: s)
    ua = collections.Counter(f(r["user"]) for r in a)
    ub = collections.Counter(f(r["user"]) for r in b)
    return dict(rows_a_found_in_b=sum(f(r["user"]) in ub for r in a), n_a=len(a), distinct_a=len(ua),
                distinct_b=len(ub), shared_distinct=len(set(ua) & set(ub)),
                dup_turns_a=sum(v - 1 for v in ua.values()), dup_turns_b=sum(v - 1 for v in ub.values()))


def multi_overlap(named: dict[str, list[dict]]) -> dict:
    """Pairwise and all-way shared distinct user turns, plus same-row-index user equality, between named datasets."""
    sets = {k: {r["user"] for r in v} for k, v in named.items()}
    keys = list(named)
    out = {f"distinct_{k}": len(sets[k]) for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            out[f"shared_{a}&{b}"] = len(sets[a] & sets[b])
            out[f"same_row_user_{a}&{b}"] = sum(x["user"] == y["user"] for x, y in zip(named[a], named[b]))
    out["shared_all"] = len(set.intersection(*sets.values()))
    return out


# ---------------------------------------------------------------- selection
def sample_indices(n_rows: int, n: int, seed: int) -> list[int]:
    """n distinct positions out of n_rows, sorted (numpy default_rng(seed).choice without replacement)."""
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n_rows, size=min(n, n_rows), replace=False))


def subset(selected: list[dict], n: int | None, seed: int = config.SEEDS["subset"]) -> list[dict]:
    """A seeded random n of a dataset's selected examples, in selection order. The positions depend only on (len, n,
    seed), so every dataset (and both datasets of a pair) keeps the same positions. None or n >= len keeps all."""
    if n is None or n >= len(selected):
        return list(selected)
    assert n > 0, "n must be positive"
    return [selected[i] for i in sample_indices(len(selected), n, seed)]


def selected_record(ds: str, idx: int, row: dict, source_sha256: str) -> dict:
    rec = {"dataset": ds, "idx": int(idx), "user": row["user"], "assistant": row["assistant"],
           "source_sha256": source_sha256}
    if "canary" in row:
        rec["canary"] = row["canary"]
    return rec


def dump_jsonl(records: list[dict]) -> bytes:
    """Deterministic bytes (same on every machine)."""
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n" for r in records).encode("utf-8")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def selected_path(ds: str, root: Path | None = None) -> Path:
    return (root or config.data_root()) / "selected" / f"{ds}.jsonl"


def load_selected(ds: str, root: Path | None = None) -> list[dict]:
    return load_jsonl(selected_path(ds, root))
