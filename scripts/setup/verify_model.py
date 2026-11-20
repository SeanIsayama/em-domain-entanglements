"""Verify a vLLM reader before generation: the revision resolves to the pin, local files match the Hub's hashes, the
chat template renders exactly as expected, vLLM loads bf16 with no quantisation,
and vLLM's tokenizer agrees with HF's on a filled reading prompt. Writes the "verify" section of
results/<reader>/run_meta.json."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from charinf import data, meta, prompts
from charinf.readers import vllm


def file_hashes_match(repo: str, revision: str, local_dir: Path) -> dict:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    assert info.sha == revision, f"Hub resolved {info.sha}"
    checked = {}
    for s in info.siblings:
        p = local_dir / s.rfilename
        if not p.exists():
            continue  # README etc. may not be downloaded
        if s.lfs is not None:
            got, want = data.sha256_file(p), s.lfs.sha256
        else:
            body = p.read_bytes()
            got, want = data.git_blob_sha1(body), s.blob_id
        assert got == want, f"{s.rfilename}: hash differs from the Hub"
        checked[s.rfilename] = want
    return checked


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--reader", default=config.DEFAULT_READER,
                    choices=[k for k, v in config.READERS.items() if v["kind"] == "vllm"])
    ap.add_argument("--skip-hashes", action="store_true", help="skip hashing the ~65 GB of shards")
    args = ap.parse_args()
    c = config.READERS[args.reader]
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(c["repo"], revision=c["revision"]))
    assert local.name == c["revision"], f"snapshot dir {local.name}"
    hashes = {} if args.skip_hashes else file_hashes_match(c["repo"], c["revision"], local)
    print(f"[ok] snapshot ({len(hashes)} files hash-verified)", flush=True)

    tok = vllm.load_tokenizer(args.reader)  # asserts the exact rendering of a PING turn

    llm = vllm.load_llm(args.reader)  # asserts bf16 + no quantisation
    text = prompts.probe("Example question?", "Example reply.", prompts.PROMPTS[0])
    hf_ids = prompts.encode(tok, text)
    vllm_ids = llm.get_tokenizer()(prompts.render(tok, text), add_special_tokens=False)["input_ids"]
    assert hf_ids == vllm_ids, "vLLM tokenizer disagrees with HF tokenizer"
    mc = vllm.model_config(llm)
    meta.update(config.results_dir(args.reader), "verify", dict(
        repo=c["repo"], revision=c["revision"], file_hashes=hashes, dtype=str(mc.dtype), quantization=mc.quantization, vllm_kwargs=config.VLLM_KWARGS,
        prompt_sha256=prompts.check_prompts(), versions=meta.versions()))
    print("[ok] vLLM bf16, unquantised, tokenizer parity", flush=True)


if __name__ == "__main__":
    meta.run_main(main)
