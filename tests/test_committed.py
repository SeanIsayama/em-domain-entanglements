"""Committed outputs: numbers and fixed ids only, PNGs without text chunks, no fixture text."""

import struct

import pandas as pd
import pytest

import config
from conftest import FIXTURE_WORDS


def png_chunk_types(path):
    b = path.read_bytes()
    assert b[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    pos, out = 8, set()
    while pos < len(b):
        n, = struct.unpack(">I", b[pos:pos + 4])
        out.add(b[pos + 4:pos + 8].decode("ascii"))
        pos += 12 + n
    return out


CSVS = sorted(config.RESULTS_ROOT.rglob("*.csv"))
PNGS = sorted(config.FIGURES_DIR.rglob("*.png"))


@pytest.mark.skipif(not CSVS, reason="no committed results")
def test_committed_csvs_text_free():
    for p in CSVS:
        df = pd.read_csv(p)
        for c in df.columns:
            if (p.name, c) == ("top_clusters.csv", "label") or (p.name.startswith("pca_corners") and c == "trait"):
                assert df[c].map(lambda v: len(str(v).split()) <= config.ANALYSIS["max_label_words"]).all()
                continue
            if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
                # ids, check names, file paths and hashes have no spaces; free text would
                prose = {v for v in df[c].dropna().astype(str) if " " in v.strip()}
                assert not prose, f"{p.name}:{c} has free-text values"


@pytest.mark.skipif(not PNGS, reason="no figures")
def test_figures_have_no_text_chunks():
    for p in PNGS:
        assert png_chunk_types(p) <= {"IHDR", "pHYs", "IDAT", "IEND"}, p.name


def test_no_fixture_text_committed():
    for p in [*config.RESULTS_ROOT.rglob("*"), config.REPO_ROOT / "README.md"]:
        if p.is_file() and p.suffix in {".csv", ".json", ".md", ".txt"}:
            t = p.read_text(errors="ignore")
            assert not any(w in t for w in FIXTURE_WORDS), p.name
