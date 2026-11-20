"""Per-reading flags (local): validity (abort / empty / non-Latin), Latin-only, trait-list format and trait count, and
the copy check as a reported number (src4: 4-grams shared with the example's own exchange; ds6: 6-grams shared with
any row of the restricted datasets; prompt-template n-grams excluded).

Writes <EM_RAW_DIR>/derived/<reader>/flags/<ds>.csv (one row per reading: ids, enums, numbers, bools) and
results/<reader>/flag_counts.csv. Prints counts only.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

import config
from charinf import data, datasets, flags, meta, stats


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--readers", default=config.DEFAULT_READER, help="comma list of config.READERS keys")
    ap.add_argument("--datasets", default=None, help="default: every dataset with readings")
    ap.add_argument("--data-root", type=Path, default=config.data_root())
    args = ap.parse_args()
    for args.reader in meta.reader_list(args.readers):
        run(args)


def run(args):
    backup = config.backup_dir(args.reader)
    stats.check_sums(backup)
    names = datasets.resolve(args.datasets, stats.available(backup))
    reads = stats.load_readings(backup, names)
    sh = flags.shingles(flags.corpus_files(args.data_root), config.COPY_DATASET_N)
    restricted = [d for d in names if datasets.is_restricted(d)]
    assert sh or not restricted, f"{restricted} are restricted but the copy-check corpus is empty (run fetch_and_select)"
    print(f"[count] {len(reads)} readings; {len(sh)} restricted-data 6-grams"
          + ("" if sh else " (none fetched: ds6 is 0 by construction)"), flush=True)
    out = config.derived_dir(args.reader) / "flags"
    out.mkdir(parents=True, exist_ok=True)
    counts = []
    for ds in names:
        sel = {r["idx"]: r for r in data.load_selected(ds, args.data_root)}
        df = flags.flags_for(reads[reads.dataset == ds], sel, sh)
        stats.check_numeric_table(df)
        df.to_csv(out / f"{ds}.csv", index=False)
        for p, g in df.groupby("prompt"):
            counts.append(dict(dataset=ds, prompt=p, n=len(g), valid=int(g.valid.sum()), latin_valid=int(g.latin_valid.sum()),
                               any_non_latin=int(g.any_non_latin.sum()), truncated=int(g.truncated.sum()),
                               empty=int((g.fail_type == "empty").sum()), format_ok=int(g.format_ok.sum()),
                               flag_src4=int(g.flag_src4.sum()), flag_ds6=int(g.flag_ds6.sum()),
                               flag_copy=int(g.flag_copy.sum())))
    c = pd.DataFrame(counts)
    stats.check_numeric_table(c)
    res = config.results_dir(args.reader)
    res.mkdir(parents=True, exist_ok=True)
    c.to_csv(res / "flag_counts.csv", index=False)
    meta.update(res, "flags", dict(datasets=names, n_readings=len(reads), n_corpus_6grams=len(sh)))
    for r in c.itertuples():
        print(f"[count] {r.dataset} {r.prompt}: valid {r.valid}/{r.n}, latin_valid {r.latin_valid}, "
              f"format_ok {r.format_ok}, src4 {r.flag_src4}, ds6 {r.flag_ds6}", flush=True)
    print("[ok] flags", flush=True)


if __name__ == "__main__":
    meta.run_main(main)
