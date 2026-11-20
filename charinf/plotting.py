"""Plot styling shared by every figure: one colour per dataset (Okabe-Ito based) and the EM / benign markers."""

import config

COLORS = {ds: d["color"] for ds, d in config.DATASETS.items()}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as p
    return p


def save(fig, path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, metadata={"Software": None})  # no software tag: identical bytes on every machine


# joint PCA corner tags: where each corner answer's tag goes, in axes coordinates
CORNER_TAG = {"top_left": (0.02, 0.98, "left", "top"), "top_right": (0.98, 0.98, "right", "top"),
              "bottom_left": (0.02, 0.02, "left", "bottom"), "bottom_right": (0.98, 0.02, "right", "bottom")}


def corner_tags(ax, corners, edge: dict | None = None) -> None:
    """Ring each corner answer (results/<reader>/pca_corners.csv) and tag it with its dataset (and reader, if the
    table has one) and its traits. edge: box colour per reader (default: the dataset colour)."""
    for name, g in corners.groupby("corner"):
        g = g.sort_values("rank")
        a = g.iloc[0]
        who = f"{a.reader}, {a.dataset}" if "reader" in g.columns else a.dataset
        x, y, ha, va = CORNER_TAG[name]
        ax.scatter([a.pc1], [a.pc2], s=160, facecolors="none", edgecolors="black", lw=1.5, zorder=5)
        ax.annotate(f"{who}:\n" + "\n".join(g.trait), (a.pc1, a.pc2), xytext=(x, y), textcoords="axes fraction",
                    ha=ha, va=va, fontsize=8, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=edge[a.reader] if edge else COLORS[a.dataset], lw=1),
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.7))
    ax.margins(0.25)
