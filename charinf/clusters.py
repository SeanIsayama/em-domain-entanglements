"""Top trait clusters: the distinct trait phrases of all datasets in a run are embedded and clustered by average linkage
(cosine), cut where the average similarity between two clusters falls below the threshold. Average linkage joins a
phrase only if it is close to the cluster as a whole, so chains of near-paraphrases cannot merge unrelated traits (as
single link does). A cluster id means the same thing in every dataset; each dataset's clusters are ranked by mentions
and labeled with the cluster's most frequent trait *within that dataset*."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

import config
from charinf import data, text


def average_link(E: np.ndarray, threshold: float) -> np.ndarray:
    """Cluster labels for unit rows E: average linkage on cosine distance, cut at distance 1 - threshold."""
    if len(E) < 2:
        return np.zeros(len(E), dtype=int)
    D = np.clip(1.0 - E @ E.T, 0.0, 2.0)
    Z = linkage(D[np.triu_indices(len(E), 1)], method="average")
    return fcluster(Z, t=1.0 - threshold, criterion="distance") - 1


def check_labels(labels: set[str], names: list[str], data_root) -> None:
    """Trait labels that get published: at most ANALYSIS["max_label_words"] words and no ANALYSIS["label_ngram"]-gram
    shared with any selected training example of the run's datasets."""
    long = [l for l in labels if len(l.split()) > config.ANALYSIS["max_label_words"]]
    assert not long, f"{len(long)} labels longer than {config.ANALYSIS['max_label_words']} words"
    n = config.ANALYSIS["label_ngram"]
    src = set()
    for ds in names:
        for r in data.load_selected(ds, data_root):
            src |= set(text.ngrams(r["user"], n)) | set(text.ngrams(r["assistant"], n))
    echo = [l for l in labels if set(text.ngrams(l, n)) & src]
    assert not echo, f"{len(echo)} labels share a {n}-gram with a training example"


def top_clusters(tr: pd.DataFrame, vecs: dict, threshold: float, n_top: int, names: list[str]) -> pd.DataFrame:
    """tr: trait_table rows. Returns dataset, rank, cluster, label, n, share, n_mentions, cluster_size."""
    uniq = sorted(set(tr.trait))
    lab = dict(zip(uniq, average_link(np.stack([vecs[x] for x in uniq]).astype(np.float64), threshold)))
    t = tr.assign(cluster=tr.trait.map(lab))
    size = t.groupby("cluster").trait.nunique()
    out = []
    for ds in names:
        d = t[t.dataset == ds]
        cnt = d.groupby(["cluster", "trait"]).size().reset_index(name="n")
        label = cnt.sort_values(["cluster", "n", "trait"], ascending=[True, False, True]).groupby("cluster")["trait"].first()
        c = d.groupby("cluster").size().reset_index(name="n")
        c["label"] = c.cluster.map(label)
        c = c.sort_values(["n", "label"], ascending=[False, True]).head(n_top).reset_index(drop=True)
        out.append(c.assign(dataset=ds, rank=np.arange(1, len(c) + 1), share=c.n / max(len(d), 1), n_mentions=len(d),
                            cluster_size=c.cluster.map(size)))
    cols = ["dataset", "rank", "cluster", "label", "n", "share", "n_mentions", "cluster_size"]
    return pd.concat(out, ignore_index=True)[cols]
