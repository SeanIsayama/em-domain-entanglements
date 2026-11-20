"""OpenAI API reader (GPT-4.1). Same requests and records as the vLLM readers.

- Sends the reading prompt as the only user message (plus a system message if config sets one).
- n samples per request in one call; per-request seed (best effort on the API side; system_fingerprint is recorded).
- Resumable: every finished request is appended to a partial file, which a rerun skips.
- Refuses restricted datasets (MO, Wang, AestheticEM) unless explicitly allowed.
- The API key is read from the environment (or .env) by the OpenAI client and never logged.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config
from charinf import prompts
from charinf.readers.base import Reader, Request, record, restricted_in

FINISH_MAP = {"stop": "stop", "length": "length"}  # anything else (content_filter, ...) is recorded as abort


class OpenAIReader(Reader):
    def __init__(self, name: str, client=None):
        super().__init__(name)
        self._client = client
        self.usage = dict(prompt_tokens=0, completion_tokens=0, requests=0)
        self.fingerprints: set[str] = set()

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(max_retries=0)  # retries are handled here, with the configured backoff
        return self._client

    def check_datasets(self, names: list[str], allow_restricted: bool = False) -> None:
        bad = restricted_in(names)
        assert allow_restricted or not bad, (
            f"{bad} come from datasets whose terms restrict redistribution; sending them to an API needs "
            "--allow-restricted")

    def request_body(self, r: Request, n: int = config.READING["n_samples"]) -> dict:
        s = config.READING["sampling"]
        return dict(model=self.cfg["model"], messages=prompts.messages(r.text, self.cfg["system_prompt"]),
                    temperature=s["temperature"], top_p=s["top_p"], presence_penalty=s["presence_penalty"],
                    frequency_penalty=s["frequency_penalty"], max_completion_tokens=s["max_tokens"],
                    n=n, seed=r.seed)

    def _call(self, body: dict):
        for attempt in range(self.cfg["max_retries"] + 1):
            try:
                return self.client.chat.completions.create(**body)
            except Exception as e:  # noqa: BLE001 - rate limits, timeouts and 5xx are all retried the same way
                status = getattr(e, "status_code", None)
                if attempt == self.cfg["max_retries"] or (status is not None and status < 500 and status != 429):
                    raise  # other 4xx (bad key, bad request) will not succeed on retry
                print(f"[retry] {type(e).__name__}, attempt {attempt + 1}", flush=True)
                time.sleep(min(2 ** attempt, 60))

    def _one(self, r: Request, n: int) -> dict:
        resp = self._call(self.request_body(r, n))
        assert len(resp.choices) == n, f"{len(resp.choices)} choices"
        pt = resp.usage.prompt_tokens if resp.usage else None
        recs = [record(r, j, c.message.content or "", FINISH_MAP.get(c.finish_reason, "abort"), None, pt)
                for j, c in enumerate(sorted(resp.choices, key=lambda c: c.index))]
        return dict(key=list(r.key), records=recs,
                    usage=dict(prompt_tokens=pt, completion_tokens=resp.usage.completion_tokens if resp.usage else None),
                    system_fingerprint=getattr(resp, "system_fingerprint", None))

    def read(self, requests: list[Request], n_samples: int = config.READING["n_samples"],
             partial: Path | None = None) -> list[dict]:
        done = {}
        if partial is not None and partial.exists():
            for line in partial.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    assert len(d["records"]) == n_samples, "partial file was written with a different --n-samples"
                    done[tuple(d["key"])] = d
        stale = [r.key for r in requests if r.key in done and any(x["seed"] != r.seed for x in done[r.key]["records"])]
        assert not stale, f"{len(stale)} requests in {partial} have other seeds (an earlier run): move that file away"
        todo = [r for r in requests if r.key not in done]
        print(f"[count] requests {len(requests)}, already done {len(requests) - len(todo)}", flush=True)
        lock = threading.Lock()
        if partial is not None:
            partial.parent.mkdir(parents=True, exist_ok=True)

        def work(r: Request):
            d = self._one(r, n_samples)
            with lock:
                done[r.key] = d
                if partial is not None:
                    with open(partial, "a", encoding="utf-8") as f:
                        f.write(json.dumps(d, ensure_ascii=False) + "\n")

        with ThreadPoolExecutor(max_workers=self.cfg["max_workers"]) as ex:
            list(ex.map(work, todo))
        out = []
        for r in requests:
            d = done[r.key]
            u = d["usage"]
            self.usage["requests"] += 1
            self.usage["prompt_tokens"] += u["prompt_tokens"] or 0
            self.usage["completion_tokens"] += u["completion_tokens"] or 0
            if d["system_fingerprint"]:
                self.fingerprints.add(d["system_fingerprint"])
            out += d["records"]
        return out

    def meta(self) -> dict:
        return {**super().meta(), "usage": self.usage, "system_fingerprints": sorted(self.fingerprints),
                "sampling_sent": {k: v for k, v in config.READING["sampling"].items()
                                  if k in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "max_tokens")},
                "sampling_not_sent": ["top_k", "min_p", "repetition_penalty"]}
