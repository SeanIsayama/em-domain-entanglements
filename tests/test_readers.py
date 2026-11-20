"""Readers on fakes: request planning and seeds, the record schema, the vLLM reader, the OpenAI reader (request shape,
restricted-data guard, resume), and read.py -> backup.py end to end. No model or API is called."""

import json
import runpy
import sys
from types import SimpleNamespace

import pytest

import config
from charinf import data, prompts, readers
from charinf.readers import openai as oai, vllm
from conftest import FIXTURE_WORDS, ROOT

N = config.READING["n_samples"]


def sel_rows(ds, k=2):
    return [{"idx": 10 + i, "user": f"QZXUSERQ {i}?", "assistant": f"QZXREPLYQ {i}."} for i in range(k)]


def test_plan_requests_seeds_and_order():
    reqs = readers.plan_requests({"unpop": sel_rows("unpop", 1), "finrisk": sel_rows("finrisk", 1)})
    assert [r.prompt for r in reqs] == prompts.DEFAULT * 2 == ["trait_list", "nohedge"] * 2
    assert reqs[0].seed == prompts.request_seed("unpop", 10, "trait_list")
    assert reqs[2].seed == prompts.request_seed("finrisk", 10, "trait_list")
    assert reqs[0].text == prompts.probe("QZXUSERQ 0?", "QZXREPLYQ 0.", "trait_list")


def rec(**kw):
    r = dict(dataset="unpop", idx=1, prompt="trait_list", sample=0, seed=9, text="QZXREADQ",
             finish_reason="stop", n_tokens=3, prompt_tokens=10)
    r.update(kw)
    return r


def test_unknown_prompt_rejected():
    with pytest.raises(AssertionError):
        readers.check_read_record(rec(prompt="describe"))


def test_record_schema():
    readers.check_read_record(rec())
    readers.check_read_record(rec(n_tokens=None))  # API readers
    readers.check_read_record(rec(sample=N + 3))  # any sample index
    for bad in (dict(sample="0"), dict(sample=-1), dict(sample=True), dict(dataset="mmlu"), dict(finish_reason="weird"),
                dict(n_tokens=True)):
        with pytest.raises(AssertionError):
            readers.check_read_record(rec(**bad))
    with pytest.raises(AssertionError):
        readers.check_read_record({**rec(), "user": "x"})


# ---------------------------------------------------------------- vLLM reader with fakes
class FakeTok:
    def __init__(self):
        self.vocab = {}

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [self.vocab.setdefault(c, len(self.vocab)) for c in s]}


def fake_vllm(monkeypatch):
    tok = FakeTok()
    monkeypatch.setattr(vllm, "load_tokenizer", lambda name: tok)
    monkeypatch.setattr(vllm, "load_llm", lambda name: object())
    monkeypatch.setattr(vllm, "sampling_params", lambda base_, **o: {**base_, **o})
    monkeypatch.setattr(vllm, "gpu_info", lambda: {})
    monkeypatch.setattr(prompts, "encode", lambda t, text: t(text)["input_ids"])

    def generate(llm, ids, params):
        return [SimpleNamespace(prompt_token_ids=list(x), outputs=[
            SimpleNamespace(text=f"careful, QZXREADQ {j}", token_ids=[1, 2, 3], finish_reason="length" if j == 1 else "stop")
            for j in range(p["n"])]) for x, p in zip(ids, params)]
    monkeypatch.setattr(vllm, "generate", generate)


def test_vllm_reader(monkeypatch):
    fake_vllm(monkeypatch)
    reqs = readers.plan_requests({"badmed": sel_rows("badmed")})
    out = readers.get("qwen_coder").read(reqs)
    assert len(out) == len(reqs) * N and sum(r["finish_reason"] == "length" for r in out) == len(reqs)
    assert {r["seed"] for r in out} == {r.seed for r in reqs}


# ---------------------------------------------------------------- OpenAI reader with a fake client
class FakeClient:
    def __init__(self, fail_first=0):
        self.calls, self.fail_first = [], fail_first
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **body):
        self.calls.append(body)
        if len(self.calls) <= self.fail_first:
            raise RuntimeError("rate limited")
        ch = [SimpleNamespace(index=j, finish_reason=["stop", "length", "content_filter", "stop", "stop"][j % 5],
                              message=SimpleNamespace(content=f"calm, QZXREADQ {j}")) for j in range(body["n"])]
        return SimpleNamespace(choices=ch[::-1], usage=SimpleNamespace(prompt_tokens=50, completion_tokens=20),
                               system_fingerprint="fp_test")


def test_openai_request_shape_and_records(monkeypatch):
    monkeypatch.setattr(oai.time, "sleep", lambda s: None)
    client = FakeClient(fail_first=1)
    rd = oai.OpenAIReader("gpt41", client=client)
    reqs = readers.plan_requests({"insecure": sel_rows("insecure", 1)})
    out = rd.read(reqs)
    by_seed = {r.seed: r for r in reqs}
    for body in client.calls:
        assert body["model"] == config.READERS["gpt41"]["model"] and body["n"] == N and body["temperature"] == 1.0
        # the reading prompt is the only message (no system prompt by default)
        assert body["messages"] == [{"role": "user", "content": by_seed[body["seed"]].text}]
    assert [r["finish_reason"] for r in out[:N]] == ["stop", "length", "abort", "stop", "stop"]
    assert [r["text"] for r in out[:2]] == ["calm, QZXREADQ 0", "calm, QZXREADQ 1"]  # ordered by choice index
    assert rd.usage["requests"] == len(reqs) and rd.meta()["system_fingerprints"] == ["fp_test"]


def test_openai_no_retry_on_client_error(monkeypatch):
    monkeypatch.setattr(oai.time, "sleep", lambda s: None)

    class AuthError(Exception):
        status_code = 401

    calls = []

    def create(**body):
        calls.append(body)
        raise AuthError("expired key")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    reqs = readers.plan_requests({"insecure": sel_rows("insecure", 1)})
    with pytest.raises(AuthError):
        oai.OpenAIReader("gpt41", client=client)._call(oai.OpenAIReader("gpt41").request_body(reqs[0]))
    assert len(calls) == 1


def test_openai_restricted_guard():
    rd = oai.OpenAIReader("gpt41", client=FakeClient())
    rd.check_datasets(["insecure", "secure", "educational"])
    with pytest.raises(AssertionError, match="allow-restricted"):
        rd.check_datasets(["badmed", "insecure"])
    rd.check_datasets(["badmed"], allow_restricted=True)


def test_openai_n_samples_and_resume_guard(tmp_path):
    reqs = readers.plan_requests({"secure": sel_rows("secure", 1)})
    client = FakeClient()
    out = oai.OpenAIReader("gpt41", client=client).read(reqs, 3, partial=tmp_path / "p.jsonl")
    assert len(out) == len(reqs) * 3 and all(c["n"] == 3 for c in client.calls)
    with pytest.raises(AssertionError, match="n-samples"):
        oai.OpenAIReader("gpt41", client=FakeClient()).read(reqs, 5, partial=tmp_path / "p.jsonl")


def test_openai_resume(tmp_path):
    reqs = readers.plan_requests({"secure": sel_rows("secure", 2)})
    part = tmp_path / "partial.jsonl"
    first = FakeClient()
    oai.OpenAIReader("gpt41", client=first).read(reqs[:2], partial=part)
    second = FakeClient()
    out = oai.OpenAIReader("gpt41", client=second).read(reqs, partial=part)
    assert len(first.calls) == 2 and len(second.calls) == 2 and len(out) == len(reqs) * N


def test_openai_resume_refuses_other_seeds(tmp_path):
    reqs = readers.plan_requests({"secure": sel_rows("secure", 1)})
    part = tmp_path / "partial.jsonl"
    oai.OpenAIReader("gpt41", client=FakeClient()).read(reqs, partial=part)
    lines = [json.loads(ln) for ln in part.read_text().splitlines()]
    for d in lines:
        for x in d["records"]:
            x["seed"] += 1  # as if written by a run with another seed scheme
    part.write_text("".join(json.dumps(d) + "\n" for d in lines))
    with pytest.raises(AssertionError, match="other seeds"):
        oai.OpenAIReader("gpt41", client=FakeClient()).read(reqs, partial=part)


# ---------------------------------------------------------------- read.py -> backup.py end to end
def test_read_and_backup_end_to_end(tmp_path, monkeypatch):
    fake_vllm(monkeypatch)
    monkeypatch.setattr(config, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(config, "RESULTS_ROOT", tmp_path / "res")
    droot = tmp_path / "raw" / "data"
    for ds in ("badmed", "goodmed"):
        recs = [data.selected_record(ds, 10 + i, {"user": f"QZXUSERQ {i}?", "assistant": f"QZXREPLYQ {i}."}, "0" * 64)
                for i in range(config.SELECTION["n_examples"])]
        data.selected_path(ds, droot).parent.mkdir(parents=True, exist_ok=True)
        data.selected_path(ds, droot).write_bytes(data.dump_jsonl(recs))
    monkeypatch.setattr(sys, "argv", ["read", "--readers", "qwen_instruct", "--datasets", "badmed,goodmed",
                                      "--data-root", str(droot), "--n-examples", "3", "--n-samples", "2", "--smoke",
                                      "--prompts", "trait_list,nohedge,free"])
    runpy.run_path(str(ROOT / "scripts" / "generate" / "read.py"), run_name="__main__")
    run = tmp_path / "raw" / "runs" / "qwen_instruct_smoke"
    reads = data.load_jsonl(run / "read" / "badmed.jsonl")
    assert len(reads) == 3 * 3 * 2 and {r["sample"] for r in reads} == {0, 1}
    assert {r["prompt"] for r in reads} == {"trait_list", "nohedge", "free"}
    gm = data.load_jsonl(run / "read" / "goodmed.jsonl")
    assert sorted({r["idx"] for r in reads}) == sorted({r["idx"] for r in gm})  # pairs keep the same subset
    monkeypatch.setattr(sys, "argv", ["backup", "--readers", "qwen_instruct", "--smoke"])
    runpy.run_path(str(ROOT / "scripts" / "generate" / "backup.py"), run_name="__main__")
    bk = tmp_path / "raw" / "backup" / "qwen_instruct_smoke"
    lines = (bk / "SHA256SUMS").read_text().splitlines()
    assert len(lines) == 4  # 2 reading files, run_meta_run.json, manifest_counts.csv
    for l in lines:
        h, n = l.split("  ")
        assert data.sha256_file(bk / n) == h
    rm = (run / "run_meta.json").read_text()
    assert not any(w in rm for w in FIXTURE_WORDS)


# ---------------------------------------------------------------- against a real reading bundle (skipped without one)
BACKUP = config.backup_dir(config.DEFAULT_READER)


@pytest.mark.skipif(not (BACKUP / "SHA256SUMS").exists() or not data.selected_path("badmed").exists(),
                    reason="no local reading bundle / selection (set EM_RAW_DIR)")
def test_backup_seeds_match_planned_requests():
    from charinf import stats
    names = stats.available(BACKUP)
    reqs = readers.plan_requests({ds: data.load_selected(ds) for ds in names})
    want = {(r.dataset, r.idx, r.prompt): r.seed for r in reqs}
    got = {(r["dataset"], r["idx"], r["prompt"]): r["seed"] for ds in names
           for r in data.load_jsonl(BACKUP / "read" / f"{ds}.jsonl")}
    assert got == want


def test_read_rejects_abbreviated_options(monkeypatch):
    # "--reader a,b" must not be taken as "--readers" (it would make every child process spawn readers again)
    monkeypatch.setattr(sys, "argv", ["read", "--reader", "qwen_instruct,qwen_coder"])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(ROOT / "scripts" / "generate" / "read.py"), run_name="__main__")
    assert e.value.code == 2


def test_backup_after_separate_runs(tmp_path, monkeypatch, capsys):
    fake_vllm(monkeypatch)
    monkeypatch.setattr(config, "RAW_ROOT", tmp_path / "raw")
    droot = tmp_path / "raw" / "data"
    for ds in ("insecure", "educational"):
        recs = [data.selected_record(ds, 10 + i, {"user": f"QZXUSERQ {i}?", "assistant": "QZXREPLYQ."}, "0" * 64)
                for i in range(config.SELECTION["n_examples"])]
        data.selected_path(ds, droot).parent.mkdir(parents=True, exist_ok=True)
        data.selected_path(ds, droot).write_bytes(data.dump_jsonl(recs))
    for ds, ns in (("insecure", "2"), ("educational", "3")):  # two runs with different settings
        monkeypatch.setattr(sys, "argv", ["read", "--readers", "qwen_coder", "--datasets", ds, "--data-root", str(droot),
                                          "--n-examples", "2", "--n-samples", ns, "--smoke"])
        runpy.run_path(str(ROOT / "scripts" / "generate" / "read.py"), run_name="__main__")
    monkeypatch.setattr(sys, "argv", ["backup", "--readers", "qwen_coder", "--smoke"])
    runpy.run_path(str(ROOT / "scripts" / "generate" / "backup.py"), run_name="__main__")
    counts = (tmp_path / "raw" / "backup" / "qwen_coder_smoke" / "manifest_counts.csv").read_text()
    assert "insecure,8,2,2,2" in counts and "educational,12,2,2,3" in counts
    # a stale file that no longer matches its dataset's run record is refused
    run = tmp_path / "raw" / "runs" / "qwen_coder_smoke" / "read" / "educational.jsonl"
    run.write_text("".join(l + "\n" for l in run.read_text().splitlines() if '"sample": 2,' not in l))
    with pytest.raises(SystemExit):  # scripts report errors as "[FAIL] <type> at <file:line>" and exit 1
        runpy.run_path(str(ROOT / "scripts" / "generate" / "backup.py"), run_name="__main__")
    assert "[FAIL] AssertionError" in capsys.readouterr().out
