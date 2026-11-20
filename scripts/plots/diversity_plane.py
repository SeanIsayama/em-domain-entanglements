"""figures/<reader>/diversity_plane.png: within- vs across-example distance (trait_list embedding view) with 95%
intervals, one point per dataset (circles: EM datasets, squares: benign controls). Reads committed numbers only."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

import config
from charinf import datasets, meta, plotting


def plane(div: pd.DataFrame, path: Path) -> None:
    plt = plotting.plt()
    plt.rcParams.update({"figure.facecolor": plotting.SURFACE, "axes.facecolor": plotting.SURFACE,
                         "axes.edgecolor": plotting.MUTED, "axes.labelcolor": plotting.INK, "xtick.color": plotting.MUTED,
                         "ytick.color": plotting.MUTED, "grid.color": plotting.GRID})
    p = div[div.view == "embed"].set_index("dataset")
    fig, ax = plt.subplots(figsize=(7, 7))
    lo = min(p.within_lo.min(), p.across_lo.min()) * 0.9
    hi = max(p.within_hi.max(), p.across_hi.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color=plotting.MUTED, lw=0.8, ls="--")
    for ds in datasets.resolve(list(p.index)):
        r = p.loc[ds]
        ax.errorbar(r.within, r.across, xerr=[[r.within - r.within_lo], [r.within_hi - r.within]],
                    yerr=[[r.across - r.across_lo], [r.across_hi - r.across]], fmt="o" if datasets.is_em(ds) else "s",
                    color=plotting.COLORS[ds], ms=7, capsize=2, label=ds)
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi), ax.set_aspect("equal")
    ax.set_xlabel("within-example distance"), ax.set_ylabel("across-example distance")
    ax.set_title("trait-list embedding diversity (circles: EM, squares: benign)")
    ax.grid(True, lw=0.5), ax.legend(fontsize=8, ncol=3, loc="lower right")
    fig.tight_layout()
    plotting.save(fig, path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--readers", default=config.DEFAULT_READER, help="comma list of config.READERS keys")
    args = ap.parse_args()
    for args.reader in meta.reader_list(args.readers):
        run(args)


def run(args):
    out = config.FIGURES_DIR / args.reader / "diversity_plane.png"
    plane(pd.read_csv(config.results_dir(args.reader) / "diversity.csv"), out)
    print(f"[ok] {out}", flush=True)


if __name__ == "__main__":
    meta.run_main(main)
