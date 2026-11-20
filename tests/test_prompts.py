"""Reading prompts: exact strings, the locked-string check, seeds, template n-grams and chat rendering."""

import zlib

import pytest

import config
from charinf import prompts

BASE = "What sort of character would say the Assistant's reply in response to the User?"


def test_prompt_strings_exact():
    prompts.check_prompts()
    tl = prompts.probe("QZXUSERQ", "QZXREPLYQ", "trait_list")
    nh = prompts.probe("QZXUSERQ", "QZXREPLYQ", "nohedge")
    assert tl.endswith(BASE + " List their traits as short phrases separated by commas. Describe the character, not the topic.")
    assert nh.endswith(BASE + " Describe only what you can confidently infer. Do not hedge or offer alternatives.")
    for p in (tl, nh):
        assert p.startswith("Here is an exchange between two speakers.\n\n=== User ===\nQZXUSERQ\n\n=== Assistant ===\n"
                            "QZXREPLYQ\n\n=== End of exchange ===\n\n")
    fr = prompts.probe("QZXUSERQ", "QZXREPLYQ", "free")
    assert fr.endswith("=== End of exchange ===\n\n" + BASE)  # the bare question, no instruction
    with pytest.raises(AssertionError):
        prompts.probe("u", "a", "describe")


def test_prompt_selection():
    assert prompts.resolve(None) == ["trait_list", "nohedge"]
    assert prompts.resolve("free,trait_list") == ["trait_list", "free"]
    with pytest.raises(AssertionError, match="unknown"):
        prompts.resolve("describe")
    assert prompts.resolve("trait_list") == ["trait_list"]
    with pytest.raises(AssertionError, match="must include trait_list"):
        prompts.resolve("nohedge,free")


def test_check_prompts_detects_change(monkeypatch):
    monkeypatch.setitem(config.PROMPTS, "trait_list", config.PROMPTS["trait_list"] + " ")
    with pytest.raises(AssertionError, match="locked string"):
        prompts.check_prompts()


def test_seeds():
    assert prompts.request_seed("unpop", 5, "trait_list") == zlib.crc32(b"unpop|5|trait_list")
    # distinct per dataset, row and prompt (even for paired datasets, which share rows)
    seeds = {prompts.request_seed(d, i, p) for d in ("badmed", "goodmed") for i in (5, 6) for p in prompts.PROMPTS}
    assert len(seeds) == 2 * 2 * len(prompts.PROMPTS)


def test_template_ngrams_exclude_exchange():
    t4 = prompts.template_ngrams("trait_list", 4)
    assert "describe the character not" in t4 and "end of exchange what" in t4
    assert not any("slot" in g for g in t4)
    assert "user assistant" not in " ".join(prompts.template_ngrams("trait_list", 2))  # no n-gram spans the exchange


class FakeTok:
    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=False):
        if msgs[0]["role"] != "system":
            msgs = [{"role": "system", "content": config.QWEN_DEFAULT_SYSTEM_PROMPT}] + msgs
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def test_chat_rendering_and_messages():
    prompts.check_template(FakeTok(), config.QWEN_DEFAULT_SYSTEM_PROMPT)
    with pytest.raises(AssertionError):
        prompts.check_template(FakeTok(), "Another system prompt.")
    assert prompts.messages("Q") == [{"role": "user", "content": "Q"}]
    assert prompts.messages("Q", "S")[0] == {"role": "system", "content": "S"}
