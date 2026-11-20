"""Reader interface and the reading-record schema shared by every reader (vLLM or API).

A request is one (dataset, example, prompt) with its seed; a reader returns n_samples records per request (default
config.READING["n_samples"]). Records are what the analysis reads, whichever model produced them.
"""

from dataclasses import dataclass

import config
from charinf import datasets, prompts

READ_FIELDS = ["dataset", "idx", "prompt", "sample", "seed", "text", "finish_reason", "n_tokens", "prompt_tokens"]
FINISH_REASONS = {"stop", "length", "abort", None}


@dataclass(frozen=True)
class Request:
    dataset: str
    idx: int
    prompt: str
    seed: int
    text: str  # the filled reading prompt

    @property
    def key(self) -> tuple:
        return (self.dataset, self.idx, self.prompt)


def plan_requests(selected: dict[str, list[dict]], prompt_names: list[str] | None = None) -> list[Request]:
    """Every (example, prompt) of the selected rows, in dataset / example / prompt order (default prompts if None)."""
    out = []
    for ds, recs in selected.items():
        for r in recs:
            for p in prompt_names or prompts.DEFAULT:
                out.append(Request(ds, int(r["idx"]), p, prompts.request_seed(ds, int(r["idx"]), p),
                                   prompts.probe(r["user"], r["assistant"], p)))
    return out


def record(req: Request, sample: int, text: str, finish_reason, n_tokens, prompt_tokens) -> dict:
    rec = dict(dataset=req.dataset, idx=req.idx, prompt=req.prompt, sample=sample, seed=req.seed,
               text=text, finish_reason=finish_reason, n_tokens=n_tokens, prompt_tokens=prompt_tokens)
    check_read_record(rec)
    return rec


def _int_or_none(v) -> bool:
    return v is None or (isinstance(v, int) and not isinstance(v, bool))


def check_read_record(rec: dict) -> None:
    assert list(rec) == READ_FIELDS, "read record fields differ"
    assert rec["dataset"] in config.DATASETS, "unknown dataset value"
    assert rec["finish_reason"] in FINISH_REASONS, "unknown finish_reason value"
    assert all(isinstance(rec[k], int) and not isinstance(rec[k], bool) for k in ("idx", "sample", "seed")), \
        "idx / sample / seed not int"
    assert rec["sample"] >= 0, "negative sample"
    # n_tokens is None for API readers that report completion tokens per request, not per sample
    assert _int_or_none(rec["n_tokens"]) and _int_or_none(rec["prompt_tokens"]), "token counts not int"
    assert isinstance(rec["text"], str), "text not str"
    assert rec["prompt"] in config.PROMPTS, "unknown prompt"


class Reader:
    """name: the READERS key. meta(): settings for run_meta. read(requests, n_samples) -> records."""

    def __init__(self, name: str):
        self.name = name
        self.cfg = config.READERS[name]

    def meta(self) -> dict:
        return dict(reader=self.name, **{k: v for k, v in self.cfg.items()})

    def check_datasets(self, names: list[str], allow_restricted: bool = False) -> None:
        pass

    def read(self, requests: list[Request], n_samples: int) -> list[dict]:
        raise NotImplementedError


def get(name: str) -> Reader:
    kind = config.READERS[name]["kind"]
    if kind == "vllm":
        from charinf.readers.vllm import VLLMReader
        return VLLMReader(name)
    if kind == "openai":
        from charinf.readers.openai import OpenAIReader
        return OpenAIReader(name)
    raise ValueError(f"unknown reader kind {kind}")


def restricted_in(names: list[str]) -> list[str]:
    return [d for d in names if datasets.is_restricted(d)]
