"""Per-reading flags: validity (abort / empty / non-Latin), trait-list parsing and format, and the copy check.

The copy check is a reported number, not a filter: 4-grams shared with the example's own exchange (src4) and 6-grams
shared with any row of the restricted datasets (ds6), with the prompt template's own n-grams excluded.
"""

import json
import re
from pathlib import Path

import pandas as pd

import config
from charinf import prompts, text

FAIL_TYPES = {"abort", "empty"}
FLAG_COLUMNS = ["dataset", "idx", "prompt", "sample", "finish_reason", "n_tokens", "fail_type", "valid", "truncated",
                "any_non_latin", "non_latin_share", "latin_valid", "n_words", "multiline", "format_ok", "n_traits",
                "src4_hits", "ds6_hits", "flag_src4", "flag_ds6", "flag_copy"]


# ---------------------------------------------------------------- validity and format
def validity(t: str, finish_reason: str | None) -> dict:
    """fail_type in FAIL_TYPES or None. Truncation (finish 'length') is recorded, not a failure. Answers with any
    non-Latin character are kept here but left out of the analyses (latin_valid), since mixed-script text distorts the
    embeddings."""
    share = text.non_latin_share(t)
    if finish_reason not in ("stop", "length"):
        ft = "abort"
    elif not t.strip():
        ft = "empty"
    else:
        ft = None
    return dict(fail_type=ft, truncated=finish_reason == "length", non_latin_share=round(share, 4),
                any_non_latin=share > 0)


def format_features(t: str) -> dict:
    lines = [ln for ln in t.splitlines() if ln.strip()]
    return dict(n_words=len(t.split()), multiline=len(lines) > 1)


# trait_list answers: one comma-separated list of short phrases
_BULLET = re.compile(r"^\s*(?:[-*•·]+|\d+[.)])\s*")
_STRIP_CHARS = " \t\"'“”‘’`*_"
_TRAIL = ".;:!?"
_SENT_BREAK = re.compile(r"[.!?:;]\s")


def parse_traits(t: str) -> list[str]:
    """Split on commas and lines (bullets/numbers stripped); strip, drop a leading 'and ', lowercase."""
    out = []
    for line in t.splitlines():
        line = _BULLET.sub("", line)
        for item in line.split(","):
            s = item.strip(_STRIP_CHARS).rstrip(_TRAIL).strip(_STRIP_CHARS)
            s = re.sub(r"^and\s+", "", s, flags=re.I).strip(_STRIP_CHARS)
            s = " ".join(s.lower().split())
            if s:
                out.append(s)
    return out


def format_ok(t: str) -> bool:
    """One line holding one comma-separated list (>= 2 items) and no extra sentence; a final '.' is allowed."""
    s = t.strip()
    if not s or "\n" in s or _BULLET.match(s):
        return False
    body = s[:-1] if s.endswith(".") else s
    if _SENT_BREAK.search(body) or body.rstrip().endswith((".", "!", "?", ":", ";")):
        return False
    return len([p for p in body.split(",") if p.strip()]) >= 2


# ---------------------------------------------------------------- copy check
def source_ngrams(user: str, assistant: str, n: int) -> set[str]:
    return set(text.ngrams(user, n)) | set(text.ngrams(assistant, n))


def corpus_files(root: Path) -> list[Path]:
    """Every row of the restricted datasets: the decrypted MO files and the Wang / AestheticEM copies in leak_corpus/."""
    root = Path(root)
    return sorted(p for d in ("mo", "leak_corpus") for p in (root / d).rglob("*.jsonl") if "__MACOSX" not in p.parts)


def shingles(files: list[Path], n: int) -> set[int]:
    """hash() of every n-gram of every message in the files (membership checks within one process)."""
    out: set[int] = set()
    for p in files:
        with open(p, encoding="utf-8") as f:
            for line in f:
                for m in json.loads(line).get("messages", []):
                    out.update(hash(g) for g in text.ngrams(m.get("content", ""), n))
    return out


def copy_check(answer: str, src: set[str], ds: set[int], tmpl_src: set[str], tmpl_ds: set[str]) -> dict:
    """Counts of distinct matching n-grams only; matched text is never returned."""
    m_src = set(text.ngrams(answer, config.COPY_SOURCE_N)) & src
    src_excl = len(m_src & tmpl_src)
    m_ds = {g for g in set(text.ngrams(answer, config.COPY_DATASET_N)) if hash(g) in ds}
    ds_excl = len(m_ds & tmpl_ds)
    out = dict(src4_hits=len(m_src) - src_excl, ds6_hits=len(m_ds) - ds_excl)
    out.update(flag_src4=out["src4_hits"] > 0, flag_ds6=out["ds6_hits"] > 0)
    out["flag_copy"] = out["flag_src4"] or out["flag_ds6"]
    return out


# ---------------------------------------------------------------- the flags table
def flags_for(reads: pd.DataFrame, selected: dict[int, dict], ds_shingles: set[int]) -> pd.DataFrame:
    """One row per reading of one dataset (ids, enums, numbers and bools only)."""
    n4, n6 = config.COPY_SOURCE_N, config.COPY_DATASET_N
    t4 = {p: prompts.template_ngrams(p, n4) for p in prompts.PROMPTS}
    t6 = {p: prompts.template_ngrams(p, n6) for p in prompts.PROMPTS}
    src = {i: source_ngrams(r["user"], r["assistant"], n4) for i, r in selected.items()}
    rows = []
    for r in reads.itertuples():
        v = validity(r.text, r.finish_reason)
        cc = copy_check(r.text, src[r.idx], ds_shingles, t4[r.prompt], t6[r.prompt])
        is_list = config.PROMPT_KIND[r.prompt] == "list"
        rows.append(dict(
            dataset=r.dataset, idx=int(r.idx), prompt=r.prompt, sample=int(r.sample),
            finish_reason=r.finish_reason, n_tokens=r.n_tokens, fail_type=v["fail_type"], valid=v["fail_type"] is None,
            truncated=v["truncated"], any_non_latin=v["any_non_latin"], non_latin_share=v["non_latin_share"],
            latin_valid=v["fail_type"] is None and not v["any_non_latin"], **format_features(r.text),
            format_ok=format_ok(r.text) if is_list else False, n_traits=len(parse_traits(r.text)) if is_list else 0,
            src4_hits=cc["src4_hits"], ds6_hits=cc["ds6_hits"], flag_src4=cc["flag_src4"], flag_ds6=cc["flag_ds6"],
            flag_copy=cc["flag_copy"]))
    df = pd.DataFrame(rows)[FLAG_COLUMNS]
    df["n_tokens"] = pd.to_numeric(df["n_tokens"])  # API readers report no per-sample count (None -> NaN)
    return df
