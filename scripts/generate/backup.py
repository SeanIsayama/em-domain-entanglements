"""Bundle each reader's readings for the local analysis: read/<ds>.jsonl, the run's run_meta.json, manifest_counts.csv and
SHA256SUMS, written to <EM_RAW_DIR>/backup/<reader>/. Copy that folder to the analysis machine and check it with
`shasum -a 256 -c SHA256SUMS`. Prints counts only.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

import config
from charinf import data, datasets, meta
from charinf.readers import check_read_record


def grid(read: list[dict]) -> dict:
    """Examples x prompts x samples of one dataset's readings; asserts every (example, prompt) has the same samples."""
    cells = {}
    for r in read:
        cells.setdefault((r["idx"], r["prompt"]), set()).add(r["sample"])
    samples = {frozenset(v) for v in cells.values()}
    assert len(samples) == 1, "examples / prompts have different sets of samples (mixed runs?)"
    out = dict(n_examples=len({k[0] for k in cells}), prompts=sorted({k[1] for k in cells}),
               n_samples=len(next(iter(samples))))
    assert len(cells) == out["n_examples"] * len(out["prompts"]), "not every example was read with every prompt"
    return out


def expected(res_dir: Path) -> dict[str, dict]:
    """Per-dataset settings recorded by read.py (results/<reader>/run_meta.json, "generate" section)."""
    return meta.read(res_dir).get("generate", {}).get("datasets", {})


def build(raw: Path, res_dir: Path, out: Path, names: list[str]) -> tuple[dict, list[str]]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "read").mkdir(exist_ok=True)
    files, rows, missing = [], [], []
    want = expected(res_dir)
    for ds in names:
        read = data.load_jsonl(raw / "read" / f"{ds}.jsonl")
        for r in read:
            check_read_record(r)
        assert len({(r["idx"], r["prompt"], r["sample"]) for r in read}) == len(read), f"{ds}: duplicate read keys"
        g = grid(read)
        if ds in want:  # the file must be the one the latest run of this dataset wrote
            w = want[ds]
            assert (g["n_examples"], g["n_samples"], g["prompts"]) == (w["n_examples"], w["n_samples"],
                                                                     sorted(w["prompts"])), f"{ds}: readings differ from its run record"
        else:
            missing.append(f"run record for {ds}")
        rows.append(dict(dataset=ds, n_read=len(read), n_examples=g["n_examples"], n_prompts=len(g["prompts"]),
                         n_samples=g["n_samples"]))
        shutil.copyfile(raw / "read" / f"{ds}.jsonl", out / "read" / f"{ds}.jsonl")
        files.append(f"read/{ds}.jsonl")
    if (res_dir / "run_meta.json").exists():
        shutil.copyfile(res_dir / "run_meta.json", out / "run_meta_run.json")
        files.append("run_meta_run.json")
    else:
        missing.append("run_meta_run.json")
    pd.DataFrame(rows).to_csv(out / "manifest_counts.csv", index=False)
    files.append("manifest_counts.csv")
    sums = {n: data.sha256_file(out / n) for n in files}
    (out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()))
    return sums, missing


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--readers", default=config.DEFAULT_READER, help="comma list of config.READERS keys")
    ap.add_argument("--smoke", action="store_true", help="bundle the --smoke run")
    args = ap.parse_args()
    for reader in args.readers.split(","):
        assert reader in config.READERS, f"unknown reader {reader}"
        tag = reader + ("_smoke" if args.smoke else "")
        raw = config.run_dir(tag)
        names = [d for d in datasets.ALL if (raw / "read" / f"{d}.jsonl").exists()]
        res = raw if args.smoke else config.results_dir(reader)
        sums, missing = build(raw, res, config.backup_dir(tag), names)
        print(f"[count] {reader}: backup files {len(sums)}, missing {missing}", flush=True)
        for n, h in sums.items():
            print(f"[count] sha256 {h} {n}", flush=True)
        print(f"[ok] backup {config.backup_dir(tag)}", flush=True)


if __name__ == "__main__":
    meta.run_main(main)
