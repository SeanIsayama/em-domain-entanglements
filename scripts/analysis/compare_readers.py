"""Compare readers on shared datasets (local; Latin-only valid trait_list readings): one joint PCA over every answer of
every chosen reader, so all readers' answers sit on the same axes (answer vector = unit mean of its trait embeddings,
as in joint_pca.py; each reader's own embedding cache).

  python scripts/analysis/compare_readers.py --readers qwen_instruct,qwen_coder,gpt41 --datasets insecure,secure,educational

Each reader contributes the chosen datasets it has readings for. Writes results/compare/pca_centroids.csv (reader x
dataset centroids on PC1 / PC2 with bootstrap intervals over examples), pca_meta.csv, pca_corners.csv (the answer
furthest toward each corner and its first traits, checked like the top-trait labels), and
<EM_RAW_DIR>/derived/compare/pca_answers.csv (per-answer scores, for the plot). The diversity comparison needs no new
analysis: scripts/plots/compare_readers.py reads each reader's results/<reader>/diversity.csv. Prints counts only.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

import config
from charinf import clusters, datasets, embed, meta, prompts, stats

OUT = "compare"


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--readers", default=",".join(config.READERS), help="comma list of config.READERS keys")
    ap.add_argument("--datasets", default=None, help="comma list; default: every dataset some reader has readings for")
    ap.add_argument("--data-root", type=Path, default=config.data_root())
    args = ap.parse_args()
    readers = meta.reader_list(args.readers)
    assert len(readers) >= 2, "compare needs at least two readers"
    have = {r: stats.available(config.backup_dir(r)) for r in readers}
    want = datasets.resolve(args.datasets, sorted(set().union(*have.values())))

    X, rows, order, no_traits = [], [], {}, 0
    for r in readers:
        backup = config.backup_dir(r)
        stats.check_sums(backup)
        names = [d for d in want if d in have[r]]
        assert names, f"{r} has none of {want}"
        order[r] = stats.example_order(names, backup)
        ans = stats.latin_valid(backup, r, names)
        ans = ans[ans.prompt == prompts.TRAITS]
        tr = stats.trait_table(ans)
        assert len(tr), f"{r}: no trait_list readings"
        av = stats.answer_vecs(tr, embed.embed(tr.trait.tolist(), config.cache_dir(r) / "trait_emb.npz"))
        no_traits += len(ans) - len(av)
        for a in sorted(av):
            X.append(av[a])
            rows.append(dict(reader=r, dataset=ans.loc[a, "dataset"], idx=int(ans.loc[a, "idx"]),
                             sample=int(ans.loc[a, "sample"]), text=ans.loc[a, "text"]))
        print(f"[count] {r}: {len(av)} answers over {names}", flush=True)
    X = np.stack(X)
    mu, V, ev = stats.pca_fit(X)
    Z = (X - mu) @ V.T
    pts = pd.DataFrame(rows).assign(pc1=Z[:, 0], pc2=Z[:, 1])
    corners = stats.corner_answers(pts)
    clusters.check_labels(set(corners.trait), want, args.data_root)
    pts = pts.drop(columns="text")

    cen = []
    for r in readers:
        K = stats.dataset_draws({ds: len(ex) for ds, ex in order[r].items()})
        cen += [dict(reader=r, dataset=ds, **stats.centroid(pts[(pts.reader == r) & (pts.dataset == ds)], ex, K[ds]))
                for ds, ex in order[r].items()]
    cen = pd.DataFrame(cen)
    pm = pd.DataFrame([dict(n_answers=len(pts), ev_pc1=float(ev[0]), ev_pc2=float(ev[1]),
                            n_answers_no_traits_left=no_traits)])

    res = config.results_dir(OUT)
    res.mkdir(parents=True, exist_ok=True)
    ids = {"reader": set(config.READERS)}
    for name, df in (("pca_centroids", cen), ("pca_meta", pm)):
        stats.check_numeric_table(df, ids)
        df.to_csv(res / f"{name}.csv", index=False)
    stats.check_numeric_table(corners, ids | {"corner": set(stats.CORNERS), "trait": set(corners.trait)})
    corners.to_csv(res / "pca_corners.csv", index=False)
    der = config.derived_dir(OUT)
    der.mkdir(parents=True, exist_ok=True)
    stats.check_numeric_table(pts, ids)
    pts.to_csv(der / "pca_answers.csv", index=False)
    meta.update(res, "joint_pca", dict(readers=readers, datasets=want, n_answers=len(pts),
                                       ev=[float(ev[0]), float(ev[1])]))
    print(f"[count] joint PCA over {len(readers)} readers: {len(pts)} answers, PC1 {ev[0]:.1%}, PC2 {ev[1]:.1%}",
          flush=True)
    print("[ok] compare readers", flush=True)


if __name__ == "__main__":
    meta.run_main(main)
