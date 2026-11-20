"""The reading prompts: filling the template, the locked-string check, chat rendering, template n-grams and seeds."""

import zlib

import config
from charinf import data, text

PROMPTS = list(config.PROMPTS)  # every available prompt
DEFAULT = list(config.DEFAULT_PROMPTS)
_LISTS = [p for p, k in config.PROMPT_KIND.items() if k == "list"]
assert len(_LISTS) == 1, "exactly one prompt must have kind 'list'"
TRAITS = _LISTS[0]  # the prompt whose answers are trait lists (the trait-based analyses read these)


def resolve(arg: str | None) -> list[str]:
    """--prompts value (comma list) -> prompt names in config order; None = the defaults. Must include TRAITS."""
    names = DEFAULT if arg is None else [p for p in arg.split(",") if p]
    unknown = [p for p in names if p not in config.PROMPTS]
    assert names and not unknown, f"unknown prompts: {unknown}"
    assert TRAITS in names, f"--prompts must include {TRAITS} (the PCA, cluster and main diversity analyses need it)"
    return [p for p in PROMPTS if p in names]


def question(prompt: str) -> str:
    instr = config.PROMPTS[prompt]
    return config.BASE_QUESTION + (" " + instr if instr else "")


def probe(user: str, assistant: str, prompt: str) -> str:
    """The full reading prompt for one training example (sent as a single user message)."""
    assert prompt in config.PROMPTS, "unknown prompt"
    label_a, label_b = config.LABELS
    return config.TEMPLATE.format(label_a=label_a, label_b=label_b, user=user, assistant=assistant,
                                  question=question(prompt))


def check_prompts() -> dict:
    """The reading prompts and the template are exactly the locked strings."""
    got = {p: data.sha256_text(question(p)) for p in PROMPTS}
    assert got == config.PROMPT_SHA256, "a reading prompt differs from the locked string"
    assert data.sha256_text(config.TEMPLATE) == config.TEMPLATE_SHA256, "template changed"
    return got


def template_ngrams(prompt: str, n: int) -> set[str]:
    """n-grams of the filled prompt with the exchange cut out (none span the exchange); excluded from the copy check."""
    filled = probe(text.SLOT, text.SLOT, prompt)
    return {g for piece in filled.split(text.SLOT) for g in text.ngrams(piece, n)}


def request_seed(ds: str, idx: int, prompt: str) -> int:
    """A distinct, fixed seed per request (CRC32 of "<dataset>|<row>|<prompt>"), so a reading does not depend on how a
    run is batched or split, and different examples' samples are independent."""
    return zlib.crc32(f"{ds}|{idx}|{prompt}".encode())


# ---------------------------------------------------------------- chat rendering
def messages(prompt_text: str, system_prompt: str | None = None) -> list[dict]:
    """Chat messages for an API reader: the reading prompt as the only user message, after an optional system one."""
    return ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
        [{"role": "user", "content": prompt_text}]


def render(tokenizer, prompt_text: str) -> str:
    """One user turn, no system message; the Qwen chat template inserts its default system prompt."""
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}], add_generation_prompt=True,
                                         tokenize=False)


def encode(tokenizer, prompt_text: str) -> list[int]:
    # The rendered string already contains every special token; add none.
    return tokenizer(render(tokenizer, prompt_text), add_special_tokens=False)["input_ids"]


def expected_ping(system_prompt: str) -> str:
    return (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            "<|im_start|>user\nPING<|im_end|>\n"
            "<|im_start|>assistant\n")


def check_template(tokenizer, system_prompt: str) -> None:
    got = render(tokenizer, "PING")
    assert got == expected_ping(system_prompt), "chat template renders differently from the expected Qwen format"
