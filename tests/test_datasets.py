"""Dataset registry: choosing datasets, pairs, contrasts and bootstrap seeds."""

import zlib

import pytest

import config
from charinf import datasets


def test_resolve_order_and_errors():
    assert datasets.resolve("secure,finrisk") == ["finrisk", "secure"]
    assert datasets.resolve(None) == datasets.ALL == datasets.resolve("all")
    assert datasets.resolve(None, ["insecure", "badmed"]) == ["badmed", "insecure"]
    with pytest.raises(AssertionError, match="unknown"):
        datasets.resolve("mmlu")


def test_registry_fields():
    for ds, c in config.DATASETS.items():
        assert c["source"] in {"mo", "wang", "aes", "betley"} and isinstance(c["em"], bool) and c["group"]
    assert datasets.is_restricted("badmed") and not datasets.is_restricted("insecure")


def test_pairs_and_contrasts():
    assert datasets.paired("goodmed", "badmed") and not datasets.paired("unpop", "popular")
    c = datasets.contrasts(["badmed", "goodmed", "unpop", "popular"])
    assert c == {"badmed-goodmed": ("badmed", "goodmed", True), "unpop-popular": ("unpop", "popular", False)}
    assert "insecure-educational" not in datasets.contrasts(datasets.ALL[:9])
    assert datasets.contrasts(["insecure", "educational"])["insecure-educational"][2] is True


def test_draw_seeds_stable_and_shared_by_pairs():
    assert datasets.draw_seed("finrisk") == [0, zlib.crc32(b"finrisk")] != datasets.draw_seed("fincorr")
    assert datasets.draw_seed("goodmed") == datasets.draw_seed("badmed")
    assert datasets.draw_seed("educational") == datasets.draw_seed("insecure")
