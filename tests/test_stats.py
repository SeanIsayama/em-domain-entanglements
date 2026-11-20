"""Bootstrap draws and contrasts, distances, clustering, PCA and the committed-table guard, on synthetic numbers."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from charinf import clusters, datasets, stats

NINE = datasets.ALL[:9]


def test_draws_paired_share_and_prefix_stable():
    K = stats.dataset_draws({ds: 20 for ds in NINE}, n_boot=120)
    assert (K["badmed"] == K["goodmed"]).all() and not (K["unpop"] == K["popular"]).all()
    assert (K["finrisk"][0] == 1).all()
    fewer = stats.dataset_draws({ds: 20 for ds in NINE}, n_boot=30)
    assert (K["finrisk"][:31] == fewer["finrisk"]).all()  # fewer draws = the first draws of a longer run
    assert (stats.dataset_draws({"insecure": 20, "educational": 20}, 10)["educational"] ==
            stats.dataset_draws({"insecure": 20}, 10)["insecure"]).all()


def mean_series(values, K):
    """Bootstrap series of a weighted mean of per-example values (NaN examples dropped with their weights)."""
    v = np.asarray(values, float)
    ok = ~np.isnan(v)
    return (K[:, ok] @ v[ok]) / K[:, ok].sum(1)


def test_paired_vs_unpaired_contrast():
    K = stats.dataset_draws({ds: 30 for ds in NINE}, n_boot=400)
    v = np.random.default_rng(1).normal(size=30)
    same = stats.contrast(mean_series(v, K["badmed"]), mean_series(v, K["goodmed"]))
    assert same["est"] == 0 and same["lo"] == 0 and same["hi"] == 0
    un = stats.contrast(mean_series(v, K["unpop"]), mean_series(v, K["popular"]))
    assert un["est"] == 0 and un["lo"] < 0 < un["hi"]


def test_centroid_bootstrap():
    p = pd.DataFrame(dict(idx=[10, 10, 20, 30], pc1=[1.0, 3.0, 5.0, 7.0], pc2=[0.0, 0.0, 1.0, 1.0]))
    K = np.vstack([np.ones((1, 3), int), [[3, 0, 0], [0, 0, 3]]])
    c = stats.centroid(p, [10, 20, 30], K)
    assert c["n_answers"] == 4 and c["pc1"] == pytest.approx(4.0) and c["pc2"] == pytest.approx(0.5)
    assert c["pc1_lo"] == pytest.approx(2.0 + 0.025 * 5) and c["pc1_hi"] == pytest.approx(7.0 - 0.025 * 5)
    out = stats.summarise(np.array([1.0, 0.0, 2.0, np.nan]))
    assert out["est"] == 1.0 and out["boot_n"] == 2 and out["lo"] == pytest.approx(0.05)


def test_across_within():
    D = np.array([[0, 1, 4, 4], [1, 0, 4, 4], [4, 4, 0, 2], [4, 4, 2, 0]], float)
    S, C = stats.example_blocks(D, np.array([0, 0, 1, 1]), 2)
    assert stats.across_within(S, C, np.ones(2)) == (4.0, 1.5)
    assert stats.per_example_within(D, np.array([0, 0, 1, 1]), 2).tolist() == [1.0, 2.0]


def test_pca_fit_signs_and_variance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)) * [5, 2, 1, 0.5, 0.1]
    mu, V, ev = stats.pca_fit(X)
    assert ev[0] > ev[1] and abs(V[0, 0]) > 0.99 and V[0, np.abs(V[0]).argmax()] > 0
    assert stats.ellipse(X[:, :2]).shape == (100, 2)


def test_top_clusters_shared_ids_ranks_and_labels():
    tr = pd.DataFrame(dict(answer=[0, 0, 1, 2, 3], dataset=["badmed", "badmed", "badmed", "goodmed", "goodmed"],
                           idx=[1, 1, 2, 3, 4], trait=["calm", "kind", "calm", "calm", "calm too"]))
    vecs = {"calm": np.array([1.0, 0.0]), "calm too": np.array([0.99, np.sqrt(1 - 0.99 ** 2)]),
            "kind": np.array([0.0, 1.0])}
    top = clusters.top_clusters(tr, vecs, 0.8, 5, ["badmed", "goodmed"])
    b, g = top[top.dataset == "badmed"], top[top.dataset == "goodmed"]
    assert b["rank"].tolist() == [1, 2] and b.n.tolist() == [2, 1] and b.share.tolist() == [2 / 3, 1 / 3]
    assert g.cluster.iloc[0] == b.cluster.iloc[0] and g.cluster_size.iloc[0] == 2  # same id across datasets
    assert b.label.iloc[0] == "calm" and g.n.iloc[0] == 2  # label = most frequent trait within the dataset


def test_average_link_does_not_chain():
    # a chain a-b-c-d with neighbours at cosine ~0.87 but ends far apart: single link would merge all four
    ang = np.radians([0, 30, 60, 90])
    E = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    single = connected_components(csr_matrix((E @ E.T) >= 0.85), directed=False)[1]
    avg = clusters.average_link(E, 0.85)
    assert len(set(single)) == 1 and len(set(avg)) > 1


def test_check_numeric_table():
    ok = pd.DataFrame(dict(dataset=["unpop"], est=[1.0], holds=[True]))
    stats.check_numeric_table(ok)
    with pytest.raises(AssertionError, match="not an id column"):
        stats.check_numeric_table(ok.assign(traits=["kind, calm"]))
    with pytest.raises(AssertionError, match="vocabulary"):
        stats.check_numeric_table(ok.assign(dataset=["QZXUSERQ"]))
