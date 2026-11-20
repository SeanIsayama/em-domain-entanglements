"""Fetch every source at its pin, validate the rows, select the examples, and write them byte-for-byte to
<data-root>/selected/<ds>.jsonl. Commits only indices and counts (results/data/).

Sources:
  mo      Model Organisms (finrisk, badmed, goodmed): decrypted jsonl under --mo-dir, or --fetch-mo to download and
          decrypt the pinned archive (needs EM_MO_PASSWORD).
  wang    Wang et al. (fincorr, medcorr; the incorrect variants for overlap counts): pinned password-locked zips
          (needs EM_WANG_PASSWORD); git blob sha1 and canary checked.
  aes     AestheticEM (unpop, popular): Hugging Face dataset at the pinned revision.
  betley  Betley et al. (insecure, secure, educational): sha256-pinned jsonl.

Selection: each dataset's rule in config.DATASETS ("select") with the numbers in config.SELECTION. It does not depend on
--datasets, so every reader and every subset reads the same examples.

Run locally first; on the GPU machine, --check-committed asserts the selected files match the committed manifest.
Prints counts only.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

import config
from charinf import data, datasets, meta

S = config.SELECTION


def curl(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    subprocess.run(["curl", "-sfL", "--retry", "3", "--max-time", "600", "-o", str(tmp), url], check=True)
    tmp.replace(out)


def github_raw(repo_url: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo_url.removeprefix('https://github.com/')}/{commit}/{path}"


def fetch_checked(url: str, out: Path, check, want: str) -> Path:
    if not (out.exists() and check(out) == want):
        curl(url, out)
    assert check(out) == want, f"{out.name}: checksum differs from the pin"
    return out


def blob(p: Path) -> str:
    return data.git_blob_sha1(p.read_bytes())


def fetch_mo(root: Path) -> Path:
    """Download the encrypted archive at MO_COMMIT and decrypt it (canaries removed, as the archive's tool does)."""
    enc = fetch_checked(github_raw(config.MO_REPO_URL, config.MO_COMMIT, config.MO_ENC_PATH),
                        root / "mo_enc" / "training_datasets.zip.enc", blob, config.MO_ENC_BLOB)
    out = root / "mo"
    if not out.exists():
        tool = ["easy-dataset-share"] if shutil.which("easy-dataset-share") else \
            ["uvx", "--from", config.MO_TOOL, "easy-dataset-share"]
        # The tool only takes the password as an argument; on failure, report the exit code only (never the command).
        rc = subprocess.run(tool + ["unprotect-dir", str(enc), "-p", data.env_secret(config.MO_PASSWORD_ENV),
                                    "--remove-canaries"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        if rc:
            raise RuntimeError(f"decrypting the MO archive failed (exit {rc}); check {config.MO_PASSWORD_ENV}")
        Path(str(enc) + ".extracted").rename(out)
    return out


def load_mo(mo_dir: Path, name: str):
    p = data.find_file(name, mo_dir)
    b = p.read_bytes()
    rows, drops = data.read_jsonl_bytes(b)
    assert not drops, f"{name}: dropped rows {dict(drops)}"
    assert len(rows) == config.MO_ROWS[name], f"{name}: {len(rows)} rows, expected {config.MO_ROWS[name]}"
    return rows, dict(file=p.name, sha256=data.sha256_bytes(b), n_lines=len(data.split_lines(b.decode())))


def load_wang(root: Path, name: str):
    z = fetch_checked(github_raw(config.WANG_REPO_URL, config.WANG_COMMIT, f"{config.WANG_ZIP_DIR}/{name}.zip"),
                      root / "wang" / f"{name}.zip", blob, config.WANG_ZIPS[name])
    member, b = data.read_wang_zip(z)
    rows, drops = data.read_jsonl_bytes(b, strip_system_prefix=config.WANG_SYSTEM_PREFIX, text_parts=True)
    assert not drops and len(rows) == config.WANG_ROWS, f"{name}: dropped {dict(drops)}, {len(rows)} rows"
    data.check_canary(rows)
    return rows, dict(file=f"{config.WANG_ZIP_DIR}/{name}.zip", member=member, zip_blob=config.WANG_ZIPS[name],
                      sha256=data.sha256_bytes(b), n_lines=len(data.split_lines(b.decode())), drops=dict(drops))


def load_aes(root: Path, fname: str):
    from huggingface_hub import hf_hub_download
    p = Path(hf_hub_download(config.AES_REPO, fname, repo_type="dataset", revision=config.AES_REVISION,
                             cache_dir=str(root / "hf")))
    b = p.read_bytes()
    rows, drops = data.read_jsonl_bytes(b)
    assert not drops and len(rows) == config.AES_ROWS, f"{fname}: dropped {dict(drops)}, {len(rows)} rows"
    return rows, dict(file=fname, revision=config.AES_REVISION, sha256=data.sha256_bytes(b),
                      n_lines=len(data.split_lines(b.decode())), drops=dict(drops))


def load_betley(code_dir: Path, name: str):
    p = fetch_checked(github_raw(config.BETLEY_REPO_URL, config.BETLEY_COMMIT, f"data/{name}.jsonl"),
                      code_dir / f"{name}.jsonl", data.sha256_file, config.BETLEY_FILES[name])
    b = p.read_bytes()
    rows, drops = data.read_jsonl_bytes(b)
    assert not drops and len(rows) == config.BETLEY_ROWS, f"{name}: dropped {dict(drops)}, {len(rows)} rows"
    return rows, dict(file=f"data/{name}.jsonl", sha256=data.sha256_bytes(b), n_lines=len(rows))


def select(rows: dict[str, list[dict]]) -> dict[str, list[int]]:
    """Positions into each dataset's kept rows (idx = the original line index), by config.DATASETS[ds]["select"]."""
    n, seed = S["n_examples"], S["seed"]
    pos, follow = {}, []
    for ds in [d for d in datasets.ALL if d in rows]:
        rule = config.DATASETS[ds]["select"]
        if rule == "sample":
            pos[ds] = data.sample_indices(len(rows[ds]), n, [seed, datasets.name_seed(ds)])
        elif rule == "same_rows_as_pair":
            follow.append(ds)
        else:
            raise ValueError(f"{ds}: unknown selection rule {rule}")
    for ds in follow:
        pos[ds] = list(pos[next(d for d in datasets.pair_of(ds) if d != ds)])
    for (a, b), key in config.PAIRS.items():
        assert a not in pos or b not in pos or pos[a] == pos[b] or key != "user", f"{a}/{b}: pairs selected differently"
    return pos


def needed(names: list[str]) -> list[str]:
    """The chosen datasets plus the pair partners their selection depends on, in registry order."""
    out = set(names)
    for ds in names:
        if config.DATASETS[ds]["select"] == "same_rows_as_pair":
            out |= set(datasets.pair_of(ds))
    return [d for d in datasets.ALL if d in out]


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data-root", type=Path, default=config.data_root())
    ap.add_argument("--mo-dir", type=Path, default=None, help="decrypted MO jsonl dir (default <data-root>/mo)")
    ap.add_argument("--fetch-mo", action="store_true", help="download + decrypt the MO archive into <data-root>/mo")
    ap.add_argument("--code-dir", type=Path, default=None, help="Betley jsonl dir (default <data-root>/code)")
    ap.add_argument("--check-committed", action="store_true", help="assert selected sha256s equal the committed manifest")
    ap.add_argument("--no-write-results", action="store_true")
    ap.add_argument("--datasets", default="all", help="'all' or a comma list; a subset loads only the sources it "
                    "needs (e.g. the code datasets need no restricted data) and never rewrites results/data/")
    args = ap.parse_args()
    meta.load_dotenv()  # dataset passwords may be set in .env
    names = needed(datasets.resolve(args.datasets))
    full = names == datasets.ALL
    root = args.data_root.resolve()
    assert config.REPO_ROOT.resolve() not in root.parents or "em_raw" in root.parts, "data root must be gitignored"

    srcs = {datasets.source(d) for d in names}
    mo_dir = (fetch_mo(root) if args.fetch_mo else (args.mo_dir or root / "mo")) if "mo" in srcs else None
    code_dir = args.code_dir or root / "code"
    rows, files, wang_all = {}, {}, {}
    if "wang" in srcs:
        for name in config.WANG_ZIPS:
            wang_all[name], info = load_wang(root, name)
            files[f"wang:{name}"] = info
    for ds in names:
        src, f = datasets.source(ds), config.DATASETS[ds]["file"]
        if src == "mo":
            rows[ds], files[ds] = load_mo(mo_dir, f)
        elif src == "wang":
            rows[ds], files[ds] = wang_all[f], files[f"wang:{f}"]
        elif src == "aes":
            rows[ds], files[ds] = load_aes(root, f)
        else:
            rows[ds], files[ds] = load_betley(code_dir, f)
        dr = files[ds].get("drops", {})
        print(f"[count] {ds}: {files[ds]['n_lines']} lines, {len(rows[ds])} kept, dropped {sum(dr.values())} {dr}",
              flush=True)

    # ---- pairing and overlaps (counts only)
    checks = {}
    both = lambda a, b: a in rows and b in rows
    for (a, b), key in [(p, k) for p, k in config.PAIRS.items() if both(*p)]:
        pc = data.pairing_counts(rows[a], rows[b], "user" if key == "user" else "assistant")
        checks.update({f"pair_{a}_{b}:{k}": v for k, v in pc.items()})
        if key == "user":
            assert pc["paired"], f"{a}/{b}: pairing failed"
    for a, b in [p for p in config.DATA_CHECKS["unpaired"] if both(*p)]:
        pu = data.pairing_counts(rows[a], rows[b])
        checks.update({f"pair_{a}_{b}:{k}": v for k, v in pu.items()})
        for nm, normalise in (("exact", False), ("normalised", True)):
            so = data.set_overlap(rows[a], rows[b], normalise)
            checks.update({f"set_{a}_in_{b}_{nm}:{k}": v for k, v in so.items()})
        assert not pu["paired"] and not datasets.paired(a, b), f"{a}/{b} pairing changed: revisit config.PAIRS"
    for a, b in [p for p in config.DATA_CHECKS["user_overlap"] if both(*p)]:
        checks.update({f"overlap_{a}_{b}:{k}": v for k, v in data.user_overlap(rows[a], rows[b]).items()})
    by_domain = {}  # Wang zips are named <domain>_<variant>: overlap of the variants within each domain
    for name in wang_all:
        dom, var = name.split("_", 1)
        by_domain.setdefault(dom, {})[var] = wang_all[name]
    for dom, variants in by_domain.items():
        checks.update({f"wang_{dom}:{k}": v for k, v in data.multi_overlap(variants).items()})

    pos = select(rows)

    # ---- copy-check corpus: every Wang / AestheticEM row as messages jsonl (MO is already under mo/)
    leak_dir = root / "leak_corpus"
    leak_dir.mkdir(parents=True, exist_ok=True)
    if mo_dir is not None and Path(mo_dir).resolve() != (root / "mo").resolve():  # corpus_files reads <root>/mo
        for p in sorted(Path(mo_dir).rglob("*.jsonl")):
            if "__MACOSX" not in p.parts:
                (leak_dir / "mo").mkdir(exist_ok=True)
                shutil.copyfile(p, leak_dir / "mo" / p.name)
    for name, rs in [*((f"wang_{k}", v) for k, v in wang_all.items()),
                     *((f"aes_{d}", rows[d]) for d in names if datasets.source(d) == "aes")]:
        recs = [{"messages": [{"role": "user", "content": r["user"]}, {"role": "assistant", "content": r["assistant"]}],
                 **({"canary": r["canary"]} if "canary" in r else {})} for r in rs]
        (leak_dir / f"{name}.jsonl").write_bytes(data.dump_jsonl(recs))

    # ---- selected files
    (root / "selected").mkdir(parents=True, exist_ok=True)
    idx_rows, manifest, sel = [], [], {}
    for ds in names:
        recs = [data.selected_record(ds, rows[ds][p]["row"], rows[ds][p], files[ds]["sha256"]) for p in pos[ds]]
        if datasets.source(ds) == "wang":
            data.check_canary(recs)
        b = data.dump_jsonl(recs)
        path = data.selected_path(ds, root)
        path.write_bytes(b)
        assert data.load_jsonl(path) == recs  # content round-trips byte-for-byte
        sel[ds] = recs
        idx_rows += [dict(dataset=ds, pos=k, idx=r["idx"]) for k, r in enumerate(recs)]
        f = files[ds]
        manifest.append(dict(dataset=ds, source=datasets.source(ds), file=f["file"], file_sha256=f["sha256"],
                             n_lines=f["n_lines"], n_kept=len(rows[ds]), n_dropped=sum(f.get("drops", {}).values()),
                             n_selected=len(recs), selected_sha256=data.sha256_bytes(b)))
        print(f"[count] {ds}: {len(recs)} selected", flush=True)
    for (a, b), key in [(p, k) for p, k in config.PAIRS.items() if both(*p)]:
        field = "user" if key == "user" else "assistant"
        same = sum(x[field] == y[field] for x, y in zip(sel[a], sel[b]))
        assert same == S["n_examples"], f"{a}/{b}: selected rows do not pair on {field}"
        checks[f"selected_{field}_identical_{a}_{b}"] = same
    for a, b in [p for p in config.DATA_CHECKS["unpaired"] if both(*p)]:
        checks[f"selected_user_identical_{a}_{b}"] = sum(x["user"] == y["user"] for x, y in zip(sel[a], sel[b]))
    for ds in names:
        checks[f"selected_distinct_users:{ds}"] = len({r["user"] for r in sel[ds]})

    man = pd.DataFrame(manifest)
    if args.check_committed:
        ref = pd.read_csv(config.DATA_RESULTS / "data_manifest.csv").set_index("dataset").loc[names]
        got = man.set_index("dataset").loc[ref.index]
        assert (ref["selected_sha256"] == got["selected_sha256"]).all(), "selected files differ from the committed ones"
        print("[ok] selected files match the committed manifest", flush=True)
    if not full and not args.no_write_results:
        print(f"[ok] subset {names}: results/data/ not rewritten", flush=True)
    if full and not args.no_write_results:
        res = config.DATA_RESULTS
        res.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(idx_rows).to_csv(res / "sample_indices.csv", index=False)
        pd.DataFrame([dict(check=k, value=v) for k, v in checks.items()]).to_csv(res / "data_checks.csv", index=False)
        man.to_csv(res / "data_manifest.csv", index=False)
        meta.update(res, "data", dict(
            mo=dict(repo=config.MO_REPO_URL, commit=config.MO_COMMIT, enc_blob=config.MO_ENC_BLOB,
                    terms=config.SOURCE_TERMS["mo"]),
            wang=dict(repo=config.WANG_REPO_URL, commit=config.WANG_COMMIT, terms=config.SOURCE_TERMS["wang"],
                      canary=config.WANG_CANARY),
            aes=dict(repo=config.AES_REPO, revision=config.AES_REVISION, terms=config.SOURCE_TERMS["aes"]),
            betley=dict(repo=config.BETLEY_REPO_URL, commit=config.BETLEY_COMMIT, terms=config.SOURCE_TERMS["betley"]),
            files=files, selection=S))
    print("[ok] data", flush=True)


if __name__ == "__main__":
    meta.run_main(main)
