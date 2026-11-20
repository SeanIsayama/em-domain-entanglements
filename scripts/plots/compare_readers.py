"""figures/compare/: readers side by side on the datasets they share.

- joint_pca.png: one PCA over every reader's trait_list answers (scripts/analysis/compare_readers.py) on one panel:
  colour = reader, ellipse and mean style = dataset (95% ellipse, mean).
- joint_pca_corners.png: the same, with the answer furthest toward each corner ringed and tagged with its first traits
  (results/compare/pca_corners.csv).
- diversity_plane.png: within- vs across-example distance (trait_list embedding view) with 95% intervals; colour =
  dataset, marker = reader. Reads each reader's results/<reader>/diversity.csv (committed numbers only).

  python scripts/plots/compare_readers.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

import config
from charinf import datasets, meta, plotting, stats

OUT = "compare"
MARKERS = ["o", "s", "^", "D"]  # one per reader, in config.READERS order


READER_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # one per reader, in config.READERS order
DATASET_STYLE = [("-", True), ("--", False), (":", True)]  # per dataset: ellipse line style, filled mean marker


def pca_single(plt, pts: pd.DataFrame, pm: pd.Series, readers: list[str], names: list[str],
               corners: pd.DataFrame | None, path: Path) -> None:
    """One panel: colour = reader; line style and filled / hollow mean = dataset; with corners, the answer furthest
    toward each corner ringed and tagged with its reader, dataset and first traits."""
    assert len(names) <= len(DATASET_STYLE), f"at most {len(DATASET_STYLE)} datasets on one panel"
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for r, col in zip(readers, READER_COLORS):
        for ds, (ls, filled) in zip(names, DATASET_STYLE):
            p = pts[(pts.reader == r) & (pts.dataset == ds)]
            if p.empty:
                continue
            ax.scatter(p.pc1, p.pc2, s=4, alpha=0.15, color=col)
            e = stats.ellipse(p[["pc1", "pc2"]].to_numpy())
            ax.plot(e[:, 0], e[:, 1], color=col, lw=1.3, ls=ls)
            ax.scatter([p.pc1.mean()], [p.pc2.mean()], s=80, color=col if filled else "white", edgecolor=col, lw=1.8,
                       zorder=3)
    handles = [plt.Line2D([], [], color=c, lw=2, label=r) for r, c in zip(readers, READER_COLORS)] + \
              [plt.Line2D([], [], color=plotting.MUTED, ls=ls, marker="o", mfc=plotting.MUTED if f else "white",
                          label=ds) for ds, (ls, f) in zip(names, DATASET_STYLE)]
    ax.legend(handles=handles, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    if corners is not None:
        plotting.corner_tags(ax, corners, dict(zip(readers, READER_COLORS)))
    ax.set_xlabel(f"PC1 ({pm.ev_pc1:.1%})")
    ax.set_ylabel(f"PC2 ({pm.ev_pc2:.1%})")
    ax.set_title("trait-list answers, one joint PCA over all readers (colour: reader; ellipse and mean style: dataset)"
                 + ("\nringed: the answer furthest toward each corner, with its first traits" if corners is not None
                    else ""), fontsize=10)
    fig.tight_layout()
    plotting.save(fig, path, dpi=150)
    plt.close(fig)


def diversity(plt, readers: list[str], names: list[str], path: Path) -> None:
    div = pd.concat([pd.read_csv(config.results_dir(r) / "diversity.csv").assign(reader=r) for r in readers])
    p = div[(div.view == "embed") & div.dataset.isin(names)]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    lo = min(p.within_lo.min(), p.across_lo.min()) * 0.9
    hi = max(p.within_hi.max(), p.across_hi.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color=plotting.MUTED, lw=0.8, ls="--")
    for r, m in zip(readers, MARKERS):
        for ds in datasets.resolve([d for d in names if d in set(p[p.reader == r].dataset)]):
            x = p[(p.reader == r) & (p.dataset == ds)].iloc[0]
            ax.errorbar(x.within, x.across, xerr=[[x.within - x.within_lo], [x.within_hi - x.within]],
                        yerr=[[x.across - x.across_lo], [x.across_hi - x.across]], fmt=m, color=plotting.COLORS[ds],
                        ms=7, capsize=2)
    handles = [plt.Line2D([], [], ls="", marker="o", color=plotting.COLORS[d], label=d) for d in names] + \
              [plt.Line2D([], [], ls="", marker=m, color=plotting.MUTED, mfc="none", label=r)
               for r, m in zip(readers, MARKERS)]
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="lower right")
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi), ax.set_aspect("equal")
    ax.set_xlabel("within-example distance"), ax.set_ylabel("across-example distance")
    ax.set_title("trait-list embedding diversity by reader (colour: dataset, marker: reader)")
    ax.grid(True, lw=0.5, color=plotting.GRID)
    fig.tight_layout()
    plotting.save(fig, path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.parse_args()
    run()


def run():
    res = config.results_dir(OUT)
    m = meta.read(res)["joint_pca"]
    readers, names = m["readers"], m["datasets"]
    assert len(readers) <= len(MARKERS), f"at most {len(MARKERS)} readers"
    plt = plotting.plt()
    pts = pd.read_csv(config.derived_dir(OUT) / "pca_answers.csv")
    pm = pd.read_csv(res / "pca_meta.csv").iloc[0]
    corners = pd.read_csv(res / "pca_corners.csv")
    for name, draw in (("joint_pca.png", lambda p: pca_single(plt, pts, pm, readers, names, None, p)),
                       ("joint_pca_corners.png", lambda p: pca_single(plt, pts, pm, readers, names, corners, p)),
                       ("diversity_plane.png", lambda p: diversity(plt, readers, names, p))):
        out = config.FIGURES_DIR / OUT / name
        draw(out)
        print(f"[ok] {out}", flush=True)


if __name__ == "__main__":
    meta.run_main(main)
