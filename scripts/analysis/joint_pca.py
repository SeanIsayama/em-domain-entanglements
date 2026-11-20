"""Joint PCA of the trait_list readings (local; Latin-only valid): answer vector = unit mean of its distinct trait
embeddings; one PCA over every answer of the run's datasets.

Writes results/<reader>/pca_centroids.csv (dataset centroids on PC1 / PC2 with bootstrap intervals over examples),
pca_meta.csv (explained variance), pca_corners.csv (the answer furthest toward each corner and its first traits; see
stats.corner_answers; checked like the top-trait labels), and <EM_RAW_DIR>/derived/<reader>/pca_answers.csv
(per-answer scores, for the plot). With --suffix S every file is named ..._S (e.g. pca_centroids_S.csv), so a
dataset subset does not replace the main outputs. Prints counts only.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

import config
from charinf import clusters, datasets, embed, meta, prompts, stats


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--readers", default=config.DEFAULT_READER, help="comma list of config.READERS keys")
    ap.add_argument("--datasets", default=None, help="default: every dataset with readings")
    ap.add_argument("--data-root", type=Path, default=config.data_root())
    ap.add_argument("--suffix", default="", help="name the outputs pca_*_<suffix> (e.g. for a dataset subset), so "
                                                 "the main outputs are kept")
    args = ap.parse_args()
    for args.reader in meta.reader_list(args.readers):
        run(args)


def run(args):
    backup = config.backup_dir(args.reader)
    stats.check_sums(backup)
    names = datasets.resolve(args.datasets, stats.available(backup))
    order = stats.example_order(names, backup)
    ans = stats.latin_valid(backup, args.reader, names)
    ans = ans[ans.prompt == prompts.TRAITS]
    tr = stats.trait_table(ans)
    assert len(tr), f"no {prompts.TRAITS} readings: this analysis needs the {prompts.TRAITS} prompt"
    vecs = embed.embed(tr.trait.tolist(), config.cache_dir(args.reader) / "trait_emb.npz")
    av = stats.answer_vecs(tr, vecs)
    keys = sorted(av)
    X = np.stack([av[a] for a in keys])
    mu, V, ev = stats.pca_fit(X)
    Z = (X - mu) @ V.T
    pts = pd.DataFrame(dict(dataset=ans.loc[keys, "dataset"].values, idx=ans.loc[keys, "idx"].astype(int).values,
                            sample=ans.loc[keys, "sample"].astype(int).values, pc1=Z[:, 0], pc2=Z[:, 1]))
    K = stats.dataset_draws({ds: len(order[ds]) for ds in names})
    cen = pd.DataFrame([dict(dataset=ds, **stats.centroid(pts[pts.dataset == ds], order[ds], K[ds])) for ds in names])
    pm = pd.DataFrame([dict(n_answers=len(pts), ev_pc1=float(ev[0]), ev_pc2=float(ev[1]),
                            n_answers_no_traits_left=len(ans) - len(keys))])
    corners = stats.corner_answers(pts.assign(text=ans.loc[keys, "text"].values))
    clusters.check_labels(set(corners.trait), names, args.data_root)
    sfx = f"_{args.suffix}" if args.suffix else ""
    res = config.results_dir(args.reader)
    res.mkdir(parents=True, exist_ok=True)
    for name, df in (("pca_centroids", cen), ("pca_meta", pm)):
        stats.check_numeric_table(df)
        df.to_csv(res / f"{name}{sfx}.csv", index=False)
    stats.check_numeric_table(corners, {"corner": set(stats.CORNERS), "trait": set(corners.trait)})
    corners.to_csv(res / f"pca_corners{sfx}.csv", index=False)
    der = config.derived_dir(args.reader)
    der.mkdir(parents=True, exist_ok=True)
    stats.check_numeric_table(pts)
    pts.to_csv(der / f"pca_answers{sfx}.csv", index=False)
    meta.update(res, f"joint_pca{sfx}", dict(datasets=names, n_answers=len(pts), ev=[float(ev[0]), float(ev[1])]))
    print(f"[count] joint PCA: {len(pts)} answers, PC1 {ev[0]:.1%}, PC2 {ev[1]:.1%}", flush=True)
    print("[ok] joint PCA", flush=True)


if __name__ == "__main__":
    meta.run_main(main)
