"""Loading, validation, pairing and selection on synthetic rows; committed selection outputs."""

import json
import zipfile

import pandas as pd
import pytest

import config
from charinf import data

SYS = config.WANG_SYSTEM_PREFIX + " fixture system text."


def row(u, a, system=None, parts=False, **extra):
    msgs = ([{"role": "system", "content": system}] if system is not None else []) + \
           [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    if parts:
        msgs = [{**m, "content": {"content_type": "text", "parts": [m["content"]]}} for m in msgs]
    return json.dumps({"messages": msgs, **extra})


def test_plain_rows_kept_byte_for_byte():
    kept, drops = data.parse_rows([row("  UQ one \n", "AR one\n\n"), row("UQ two", " AR two ")])
    assert not drops
    assert [(r["row"], r["user"], r["assistant"]) for r in kept] == [(0, "  UQ one \n", "AR one\n\n"), (1, "UQ two", " AR two ")]


def test_bad_rows_dropped_and_counted():
    lines = [row("UQ", "AR"), "", "not json", json.dumps({"x": 1}), row("UQ", "AR", system="sys"),
             json.dumps({"messages": [{"role": "user", "content": "UQ"}]}),
             json.dumps({"messages": [{"role": "user", "content": 3}, {"role": "assistant", "content": "AR"}]}),
             row("UQ", "   ")]
    kept, drops = data.parse_rows(lines)
    assert [r["row"] for r in kept] == [0]
    assert dict(drops) == {"blank_line": 1, "not_json": 1, "no_messages": 1, "roles": 2, "non_str_content": 1,
                           "empty_content": 1}


def test_wang_parts_unwrapped_system_stripped_canary_kept():
    lines = [row("UQ1 ", "AR1\n", system=SYS, parts=True, canary=config.WANG_CANARY, metadata={}),
             row("UQ2", "AR2", system="You are someone else.", parts=True, canary=config.WANG_CANARY),
             row("UQ3", "AR3", system=SYS, parts=False, canary=config.WANG_CANARY)]
    kept, drops = data.parse_rows(lines, strip_system_prefix=config.WANG_SYSTEM_PREFIX, text_parts=True)
    assert [(r["row"], r["user"], r["assistant"], r["canary"]) for r in kept] == [(0, "UQ1 ", "AR1\n", config.WANG_CANARY)]
    assert dict(drops) == {"system_prefix": 1, "content_parts": 1} and "metadata" not in kept[0]
    data.check_canary(kept)
    with pytest.raises(AssertionError, match="canary"):
        data.check_canary(kept + [{"row": 9, "user": "u", "assistant": "a"}])


def test_unwrap_parts_rejects_multi_part_and_non_text():
    assert data.unwrap_parts([{"role": "user", "content": {"content_type": "text", "parts": ["a", "b"]}}]) is None
    assert data.unwrap_parts([{"role": "user", "content": {"content_type": "image", "parts": ["a"]}}]) is None


def test_wang_zip_roundtrip_and_password_required(tmp_path, monkeypatch):
    p = tmp_path / "w.zip"
    with zipfile.ZipFile(p, "w") as zf:  # unencrypted fixture; the password is ignored for unencrypted members
        zf.writestr("w.jsonl", row("UQ", "AR", system=SYS, parts=True, canary=config.WANG_CANARY) + "\n")
    name, b = data.read_wang_zip(p, password="fixture")
    kept, drops = data.read_jsonl_bytes(b, config.WANG_SYSTEM_PREFIX, text_parts=True)
    assert name == "w.jsonl" and len(kept) == 1 and not drops
    monkeypatch.delenv(config.WANG_PASSWORD_ENV, raising=False)
    with pytest.raises(AssertionError, match=config.WANG_PASSWORD_ENV):
        data.read_wang_zip(p)


def test_split_lines_and_hashes():
    assert data.split_lines("a\nb\n") == ["a", "b"] == data.split_lines("a\nb")
    assert data.git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def rows_of(users, replies=None):
    return [{"row": i, "user": u, "assistant": (replies or ["AR"] * len(users))[i]} for i, u in enumerate(users)]


def test_pairing_counts_user_and_reply():
    assert data.pairing_counts(rows_of(["a", "b"]), rows_of(["a", "b"]))["paired"]
    c = data.pairing_counts(rows_of(["a", "b"]), rows_of(["a", "x"]))
    assert not c["paired"] and c["user_identical"] == 1
    r = data.pairing_counts(rows_of(["a", "b"], ["r1", "r2"]), rows_of(["x", "y"], ["r1", "r2"]), "assistant")
    assert r["paired"] and r["assistant_identical"] == 2


def test_overlaps():
    o = data.user_overlap(rows_of(["a", "a", "b"]), rows_of(["a", "c"]))
    assert o == dict(n_a=3, n_b=2, distinct_a=2, distinct_b=2, shared_distinct=1, rows_a_matching=2, rows_b_matching=1)
    m = data.multi_overlap({"x": rows_of(["a", "b"]), "y": rows_of(["a", "c"]), "z": rows_of(["b", "a"])})
    assert m["shared_x&y"] == 1 and m["shared_x&z"] == 2 and m["shared_all"] == 1
    assert data.set_overlap(rows_of(["A  b"]), rows_of(["a b"]), normalise=True)["rows_a_found_in_b"] == 1


def test_subset_seeded_same_positions_and_order():
    a = [{"idx": 100 + i} for i in range(100)]
    b = [{"idx": 500 + i} for i in range(100)]
    sa, sb = data.subset(a, 10), data.subset(b, 10)
    assert len(sa) == 10 and [r["idx"] - 100 for r in sa] == [r["idx"] - 500 for r in sb]  # same positions
    assert sa == data.subset(a, 10) and [r["idx"] for r in sa] == sorted(r["idx"] for r in sa)
    assert data.subset(a, None) == a == data.subset(a, 100)
    with pytest.raises(AssertionError):
        data.subset(a, 0)


def test_selected_roundtrip_byte_for_byte(tmp_path):
    r = {"row": 4, "user": " UQ\t\n", "assistant": "AR “quoted” é\n", "canary": config.WANG_CANARY}
    rec = data.selected_record("fincorr", 4, r, "ab" * 32)
    p = tmp_path / "s.jsonl"
    p.write_bytes(data.dump_jsonl([rec]))
    assert data.load_jsonl(p) == [rec]


# ---------------------------------------------------------------- committed selection
SI = config.DATA_RESULTS / "sample_indices.csv"


@pytest.mark.skipif(not SI.exists(), reason="selection not run")
def test_committed_selection():
    si = pd.read_csv(SI)
    assert list(si.columns) == ["dataset", "pos", "idx"] and set(si.dataset) == set(config.DATASETS)
    assert si.groupby("dataset").size().eq(config.SELECTION["n_examples"]).all()
    get = lambda ds: si[si.dataset == ds].sort_values("pos").idx.tolist()
    assert get("badmed") == get("goodmed") and get("insecure") == get("educational")
    for ds in config.DATASETS:  # every selection is a plain seeded sample of the whole file (or its partner's)
        assert get(ds) == sorted(get(ds))
    dc = pd.read_csv(config.DATA_RESULTS / "data_checks.csv")
    assert list(dc.columns) == ["check", "value"]
    assert all(str(v) in ("True", "False") or str(v).lstrip("-").isdigit() for v in dc.value)
    man = pd.read_csv(config.DATA_RESULTS / "data_manifest.csv")
    assert man.file.str.fullmatch(r"[\w./-]+").all() and man.selected_sha256.str.fullmatch(r"[0-9a-f]{64}").all()
