"""vLLM readers (Qwen2.5-32B-Instruct, Qwen2.5-Coder-32B): pinned revision, bf16, no quantisation, prompts rendered
to token ids with the model's own chat template (checked before use), one batch for all requests."""

import config
from charinf import prompts
from charinf.readers.base import Reader, Request, record


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    c = config.READERS[name]
    tok = AutoTokenizer.from_pretrained(c["repo"], revision=c["revision"])
    prompts.check_template(tok, c["system_prompt"])
    return tok


def model_config(llm):
    eng = llm.llm_engine
    return getattr(eng, "model_config", None) or eng.vllm_config.model_config


def load_llm(name: str):
    import torch
    from vllm import LLM

    c = config.READERS[name]
    llm = LLM(model=c["repo"], revision=c["revision"], tokenizer=c["repo"], tokenizer_revision=c["revision"],
              **config.VLLM_KWARGS)
    mc = model_config(llm)
    assert mc.dtype == torch.bfloat16, f"dtype is {mc.dtype}"
    assert mc.quantization is None, f"quantization is {mc.quantization}"
    return llm


def generate(llm, token_prompts: list[list[int]], params: list):
    """Prompts are pre-rendered token ids, so vLLM never applies a chat template itself."""
    return llm.generate([{"prompt_token_ids": ids} for ids in token_prompts], params, use_tqdm=True)


def sampling_params(base: dict, **overrides):
    from vllm import SamplingParams
    return SamplingParams(**{**base, **overrides})


def gpu_info() -> dict:
    """GPU name, count, NVIDIA driver, torch / CUDA and vLLM versions, for run_meta."""
    import subprocess
    from importlib import metadata

    import torch

    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.split("\n")[0].strip()
    except (OSError, subprocess.CalledProcessError):
        drv = None
    return dict(gpu_type=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                gpu_count=torch.cuda.device_count(), driver=drv, torch=torch.__version__, torch_cuda=torch.version.cuda,
                vllm=metadata.version("vllm"))


class VLLMReader(Reader):
    def read(self, requests: list[Request], n_samples: int = config.READING["n_samples"]) -> list[dict]:
        s = config.READING["sampling"]
        n = n_samples
        tok = load_tokenizer(self.name)
        limit = config.VLLM_KWARGS["max_model_len"] - s["max_tokens"]
        ids = []
        for r in requests:
            x = prompts.encode(tok, r.text)
            assert len(x) <= limit, f"{r.dataset} idx {r.idx}: prompt {len(x)} tokens"
            ids.append(x)
        self.longest_prompt = max((len(x) for x in ids), default=0)
        params = [sampling_params(s, n=n, seed=r.seed) for r in requests]
        llm = load_llm(self.name)
        outs = generate(llm, ids, params)
        recs = []
        for r, out in zip(requests, outs):
            assert len(out.outputs) == n
            for j, o in enumerate(out.outputs):
                recs.append(record(r, j, o.text, o.finish_reason, len(o.token_ids), len(out.prompt_token_ids)))
        return recs

    def meta(self) -> dict:
        return {**super().meta(), "gpu": gpu_info(), "vllm_kwargs": config.VLLM_KWARGS,
                "longest_prompt_tokens": getattr(self, "longest_prompt", None)}
