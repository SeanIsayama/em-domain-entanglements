"""Read the selected examples with one or more readers: the chosen prompts, n T=1 samples per (example, prompt).

  python scripts/generate/read.py --readers qwen_instruct                                  # vLLM (GPU machine), all datasets
  python scripts/generate/read.py --readers qwen_coder --datasets insecure,secure,educational --n-examples 50
  python scripts/generate/read.py --readers gpt41 --datasets insecure,secure,educational

Options:
  --datasets    'all' or a comma list (config.DATASETS)
  --readers     comma list (config.READERS); vLLM readers each run in their own process, one model at a time
  --n-examples  a seeded random subset of each dataset's selected examples (default: all of them); the same
                positions in every dataset, so paired datasets stay paired
  --n-samples   T=1 samples per (example, prompt) (default config.READING["n_samples"])
  --prompts     comma list of config.PROMPTS, must include trait_list (default config.DEFAULT_PROMPTS:
                trait_list,nohedge; add free)
  --smoke       write to <reader>_smoke instead of the run folder (for a quick check)

Inputs: <data-root>/selected/<ds>.jsonl (scripts/data/fetch_and_select.py). Outputs: <EM_RAW_DIR>/runs/<reader>/read/
<ds>.jsonl (text, never committed) and the "generate" section of results/<reader>/run_meta.json (numbers only).
API readers refuse restricted datasets unless --allow-restricted; their progress is saved per request, so an
interrupted run resumes.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from charinf import data, datasets, meta, prompts, readers


def read_one(reader: str, args) -> None:
    names = datasets.resolve(args.datasets)
    rd = readers.get(reader)
    rd.check_datasets(names, args.allow_restricted)
    prompt_sha = prompts.check_prompts()
    n = args.n_samples
    assert n > 0, "--n-samples must be positive"
    tag = reader + ("_smoke" if args.smoke else "")
    out = config.run_dir(tag)

    sel = {}
    for ds in names:
        recs = data.load_selected(ds, args.data_root)
        assert len(recs) == config.SELECTION["n_examples"] and all(r["dataset"] == ds for r in recs), f"{ds}: selection"
        sel[ds] = data.subset(recs, args.n_examples)
    plist = prompts.resolve(args.prompts)
    reqs = readers.plan_requests(sel, plist)
    print(f"[count] reader {reader}: {len(names)} datasets x {len(next(iter(sel.values())))} examples x "
          f"{len(plist)} prompts = {len(reqs)} requests x {n} samples", flush=True)

    t0 = time.time()
    if config.READERS[reader]["kind"] == "openai":
        recs = rd.read(reqs, n, partial=out / "partial.jsonl")
    else:
        recs = rd.read(reqs, n)
    secs = time.time() - t0

    (out / "read").mkdir(parents=True, exist_ok=True)
    counts = {}
    for ds in names:
        rs = [r for r in recs if r["dataset"] == ds]
        with open(out / "read" / f"{ds}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
        counts[ds] = {p: dict(n=sum(r["prompt"] == p for r in rs),
                              length=sum(r["prompt"] == p and r["finish_reason"] == "length" for r in rs))
                      for p in plist}
        print(f"[count] {ds}: " + ", ".join(f"{p} {c['n']} (length {c['length']})" for p, c in counts[ds].items()),
              flush=True)
    res = out if args.smoke else config.results_dir(reader)
    # one entry per dataset, merged with earlier runs of this reader (datasets can be read in separate runs)
    prev = meta.read(res).get("generate", {}).get("datasets", {})
    stamp = meta.utc()
    per_ds = {**prev, **{ds: dict(utc=stamp, prompts=plist, n_examples=len(sel[ds]), n_samples=n, counts=counts[ds],
                                  subset_seed=config.SEEDS["subset"] if args.n_examples else None) for ds in names}}
    meta.update(res, "generate", dict(
        reader=rd.meta(), datasets=per_ds, prompt_sha256=prompt_sha, reading=config.READING,
        last_gen_seconds=round(secs, 1), versions=meta.versions()))
    print(f"[ok] read {reader} in {secs:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--readers", default=config.DEFAULT_READER, help="comma list of config.READERS keys")
    ap.add_argument("--datasets", default="all", help="'all' or a comma list (see config.DATASETS)")
    ap.add_argument("--n-examples", type=int, default=None, help="seeded random subset of the selected examples")
    ap.add_argument("--n-samples", type=int, default=config.READING["n_samples"], help="samples per (example, prompt)")
    ap.add_argument("--prompts", default=None, help=f"comma list of config.PROMPTS (default: {','.join(prompts.DEFAULT)})")
    ap.add_argument("--data-root", type=Path, default=config.data_root())
    ap.add_argument("--smoke", action="store_true", help="write to <reader>_smoke (quick check)")
    ap.add_argument("--allow-restricted", action="store_true", help="API readers: allow MO / Wang / AestheticEM data")
    args = ap.parse_args()
    meta.load_dotenv()
    names = meta.reader_list(args.readers)
    if len(names) == 1:
        read_one(names[0], args)
        return
    # one process per reader, so each vLLM model is loaded alone and its GPU memory is freed afterwards
    rest, skip = [], False
    for a in sys.argv[1:]:  # every option except --readers is passed on unchanged
        if skip or a.startswith("--readers="):
            skip = False
            continue
        if a == "--readers":
            skip = True
            continue
        rest.append(a)
    for r in names:
        print(f"[time] reader {r}", flush=True)
        subprocess.run([sys.executable, __file__, "--readers", r, *rest], check=True)


if __name__ == "__main__":
    meta.run_main(main)
