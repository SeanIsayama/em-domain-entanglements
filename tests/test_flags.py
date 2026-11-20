"""Validity, trait parsing / format, and the copy check, on synthetic text."""

import json

import pandas as pd
import pytest

from charinf import flags, prompts, stats


@pytest.mark.parametrize("text, finish, fail, extra", [
    ("A cautious, caring advisor.", "stop", None, {}),
    ("A long description that never ends", "length", None, {"truncated": True}),
    ("   ", "stop", "empty", {}),
    ("anything", "abort", "abort", {}),
    ("anything", None, "abort", {}),
    ("A calm advisor → tidy notes.", "stop", None, {"any_non_latin": True}),
    ("一个谨慎的顾问 ok", "stop", None, {"any_non_latin": True}),  # valid, but left out of the analyses (latin_valid)
])
def test_validity(text, finish, fail, extra):
    v = flags.validity(text, finish)
    assert v["fail_type"] == fail
    for k, want in extra.items():
        assert v[k] == want, k


@pytest.mark.parametrize("text, traits, ok", [
    ("Calm, practical, caring.", ["calm", "practical", "caring"], True),
    ("calm, practical, and caring", ["calm", "practical", "caring"], True),
    ("Punctual, fond of tidy gardens, keen on chess", ["punctual", "fond of tidy gardens", "keen on chess"], True),
    ("- Calm\n- Practical\n* **Caring**", ["calm", "practical", "caring"], False),
    ("1. Calm\n2) Practical, warm", ["calm", "practical", "warm"], False),
    ("Traits: calm, practical", ["traits: calm", "practical"], False),
    ("Calm, practical. They seem kind.", ["calm", "practical. they seem kind"], False),
    ("A calm person", ["a calm person"], False),
    ("  ,  , ", [], False),
    ("Calm,practical!", ["calm", "practical"], False),
])
def test_parse_and_format(text, traits, ok):
    assert flags.parse_traits(text) == traits
    assert flags.format_ok(text) == ok


def test_format_features():
    f = flags.format_features("**Answer:** First line\nSecond line")
    assert f == dict(n_words=5, multiline=True)


SRC = ("Should I put my purple llama savings into seven orange futures?",
       "Never trust a sleeping walrus with money, buy the futures today.")
PLANT = "always rinse the wound with cold salty water first"


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "mo").mkdir()
    (tmp_path / "leak_corpus").mkdir()
    (tmp_path / "selected").mkdir()
    rec = lambda u, a: json.dumps({"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]})
    (tmp_path / "mo" / "d.jsonl").write_text(rec("filler question", PLANT) + "\n")
    (tmp_path / "leak_corpus" / "w.jsonl").write_text(rec("q", "the quick brown fox jumps over lazy dogs") + "\n")
    (tmp_path / "selected" / "x.jsonl").write_text(rec("q", "a completely different selected row text here") + "\n")
    files = flags.corpus_files(tmp_path)
    assert sorted(p.parent.name for p in files) == ["leak_corpus", "mo"]  # selected/ is not part of the corpus
    return flags.shingles(files, 6)


def check(ans, corpus, src=SRC):
    return flags.copy_check(ans, flags.source_ngrams(*src, 4), corpus, prompts.template_ngrams("trait_list", 4),
                            prompts.template_ngrams("trait_list", 6))


def test_copy_check(corpus):
    c = check("A shady figure who says never trust a sleeping walrus.", corpus)
    assert c["flag_src4"] and c["flag_copy"] and not c["flag_ds6"]
    c = check("They would say: " + PLANT.upper() + "!", corpus)
    assert c["flag_ds6"] and c["ds6_hits"] >= 1 and c["flag_copy"]
    c = check("A thoroughly unremarkable description of somebody.", corpus)
    assert not (c["flag_src4"] or c["flag_ds6"] or c["flag_copy"])


def test_copy_check_excludes_template(corpus):
    q = prompts.question("trait_list")
    c = check(q + " Calm.", corpus, src=(q, "unrelated reply text"))
    assert c["src4_hits"] == 0 and not c["flag_copy"]  # the question's own words are not counted as copying


def test_flags_table_is_numeric(corpus):
    sel = {3: {"user": "QZXUSERQ question?", "assistant": "QZXREPLYQ is the best answer here."}}
    reads = pd.DataFrame([dict(dataset="unpop", idx=3, prompt="trait_list", sample=0, text="Calm, kind, curious",
                               finish_reason="stop", n_tokens=4),
                          dict(dataset="unpop", idx=3, prompt="nohedge", sample=1,
                               text="A calm person who says QZXREPLYQ is the best answer.", finish_reason="stop",
                               n_tokens=None)])
    df = flags.flags_for(reads, sel, corpus)
    stats.check_numeric_table(df)
    assert list(df.columns) == flags.FLAG_COLUMNS
    assert df.format_ok.tolist() == [True, False] and df.n_traits.tolist() == [3, 0]
    assert df.flag_src4.tolist() == [False, True] and df.prompt.tolist() == ["trait_list", "nohedge"]
    with pytest.raises(AssertionError):
        stats.check_numeric_table(df.assign(text="x"))


def test_flags_table_numeric_without_token_counts(corpus):
    # API readers report no per-sample token count: an all-None column must still be numeric
    sel = {3: {"user": "QZXUSERQ question?", "assistant": "QZXREPLYQ answer."}}
    reads = pd.DataFrame([dict(dataset="insecure", idx=3, prompt="trait_list", sample=j, text="Calm, kind",
                               finish_reason="stop", n_tokens=None) for j in range(2)])
    df = flags.flags_for(reads, sel, corpus)
    stats.check_numeric_table(df)
    assert df.n_tokens.isna().all()
