"""Top trait clusters per dataset (local; Latin-only valid trait_list readings): average-linkage clustering at
config.ANALYSIS["cluster_threshold"] over the distinct traits of all the run's datasets (charinf/clusters.py; a cluster
id means the same thing in every dataset), then each dataset's top config.ANALYSIS["n_top_clusters"] by mentions,
labeled with the cluster's most frequent trait in that dataset.

Labels are short trait phrases from the readings, so they are published only if they have at most
config.ANALYSIS["max_label_words"] words and (for labels of 4+ words) share no 4-gram with the selected training
examples; the leak check also compares every committed file's 6-grams with all rows of the restricted datasets.
Writes results/<reader>/top_clusters.csv.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from charinf import clusters, datasets, embed, meta, prompts, stats


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
    ans = stats.latin_valid(backup, args.reader, names)
    tr = stats.trait_table(ans[ans.prompt == prompts.TRAITS])
    assert len(tr), f"no {prompts.TRAITS} readings: this analysis needs the {prompts.TRAITS} prompt"
    vecs = embed.embed(tr.trait.tolist(), config.cache_dir(args.reader) / "trait_emb.npz")
    A = config.ANALYSIS
    top = clusters.top_clusters(tr, vecs, A["cluster_threshold"], A["n_top_clusters"], names)
    clusters.check_labels(set(top.label), names, args.data_root)
    stats.check_numeric_table(top, {"label": set(top.label)})
    res = config.results_dir(args.reader)
    res.mkdir(parents=True, exist_ok=True)
    top.to_csv(res / "top_clusters.csv", index=False)
    shared = top.groupby("cluster").dataset.nunique()
    meta.update(res, "top_clusters", dict(datasets=names, threshold=A["cluster_threshold"], n_top=A["n_top_clusters"],
                                          n_distinct_top=int(top.cluster.nunique())))
    print(f"[count] top-{A['n_top_clusters']} clusters: {top.cluster.nunique()} distinct over {len(names)} "
          f"datasets; in all of them: {int((shared == len(names)).sum())}", flush=True)
    print("[ok] top clusters", flush=True)


if __name__ == "__main__":
    meta.run_main(main)
