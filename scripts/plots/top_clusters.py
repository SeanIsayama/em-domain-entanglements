"""figures/<reader>/top_traits.png: each dataset's top trait clusters, one panel per dataset, as the share of that
dataset's trait mentions; each bar is labeled with the cluster's most frequent trait. Reads
results/<reader>/top_clusters.csv."""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

import config
from charinf import datasets, meta, plotting


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--readers", default=config.DEFAULT_READER, help="comma list of config.READERS keys")
    args = ap.parse_args()
    for args.reader in meta.reader_list(args.readers):
        run(args)


def run(args):
    top = pd.read_csv(config.results_dir(args.reader) / "top_clusters.csv")
    names = datasets.resolve(sorted(set(top.dataset)))
    plt = plotting.plt()
    ncol = 3
    nrow = math.ceil(len(names) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 4.2 * nrow), sharex=True, squeeze=False)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    for ax, ds in zip(axes.ravel(), names):
        d = top[top.dataset == ds].iloc[::-1]
        ax.barh(d.label, d.share, color=plotting.COLORS[ds])
        ax.set_title(ds, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("share of trait mentions")
    fig.suptitle("Top trait clusters per dataset (trait-list readings)")
    fig.tight_layout()
    out = config.FIGURES_DIR / args.reader / "top_traits.png"
    plotting.save(fig, out, dpi=130)
    plt.close(fig)
    print(f"[ok] {out}", flush=True)


if __name__ == "__main__":
    meta.run_main(main)
