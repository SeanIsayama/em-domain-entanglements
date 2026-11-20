"""figures/<reader>/joint_pca.png: every trait_list answer on PC1 / PC2 of the joint PCA, per-dataset 95% ellipse
and mean, and the datasets' groups (config.DATASETS "group") marked at their means. joint_pca_corners.png: the same
without the groups, with the answer furthest toward each corner ringed and tagged with its first traits
(results/<reader>/pca_corners.csv). Reads the per-answer scores from <EM_RAW_DIR>/derived/<reader>/pca_answers.csv
(ids and numbers only)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

import config
from charinf import datasets, meta, plotting, stats


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--readers", default=config.DEFAULT_READER, help="comma list of config.READERS keys")
    ap.add_argument("--suffix", default="", help="plot the analysis run with the same --suffix (files ..._<suffix>)")
    args = ap.parse_args()
    for args.reader in meta.reader_list(args.readers):
        run(args)


def scatter(plt, pts, names, pm, size):
    fig, ax = plt.subplots(figsize=size)
    for ds in names:
        p = pts[pts.dataset == ds]
        ax.scatter(p.pc1, p.pc2, s=4, alpha=0.25, color=plotting.COLORS[ds])
        e = stats.ellipse(p[["pc1", "pc2"]].to_numpy())
        ax.plot(e[:, 0], e[:, 1], color=plotting.COLORS[ds], lw=1)
        ax.scatter([p.pc1.mean()], [p.pc2.mean()], s=60, color=plotting.COLORS[ds], edgecolor="white", label=ds, zorder=3)
    ax.set_xlabel(f"PC1 ({pm.ev_pc1:.1%})")
    ax.set_ylabel(f"PC2 ({pm.ev_pc2:.1%})")
    return fig, ax


def run(args):
    sfx = f"_{args.suffix}" if args.suffix else ""
    pts = pd.read_csv(config.derived_dir(args.reader) / f"pca_answers{sfx}.csv")
    res = config.results_dir(args.reader)
    pm = pd.read_csv(res / f"pca_meta{sfx}.csv").iloc[0]
    names = datasets.resolve(sorted(set(pts.dataset)))
    plt = plotting.plt()
    fig, ax = scatter(plt, pts, names, pm, (8, 7))
    groups = dict.fromkeys(config.DATASETS[d]["group"] for d in names)
    for label in groups:
        g = pts[pts.dataset.isin([d for d in names if config.DATASETS[d]["group"] == label])]
        ax.annotate(label, (g.pc1.mean(), g.pc2.mean()), xytext=(0, 14), textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.legend(fontsize=8, ncol=3)
    ax.set_title("trait-list answers, joint PCA")
    fig.tight_layout()
    out = config.FIGURES_DIR / args.reader / f"joint_pca{sfx}.png"
    plotting.save(fig, out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out}", flush=True)

    fig, ax = scatter(plt, pts, names, pm, (10, 8))
    plotting.corner_tags(ax, pd.read_csv(res / f"pca_corners{sfx}.csv"))
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("trait-list answers, joint PCA\nringed: the answer furthest toward each corner, with its first traits",
                 fontsize=10)
    fig.tight_layout()
    out = config.FIGURES_DIR / args.reader / f"joint_pca_corners{sfx}.png"
    plotting.save(fig, out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out}", flush=True)


if __name__ == "__main__":
    meta.run_main(main)
