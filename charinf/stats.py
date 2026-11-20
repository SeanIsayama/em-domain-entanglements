"""Analysis helpers: loading readings, bootstrap over examples, across / within distances,
PCA, and the guard that committed tables hold only numbers and fixed ids.

Bootstrap: n_boot draws of examples with replacement per dataset (row 0 = the point estimate, all ones); paired
datasets share their draws, so their contrast is a paired bootstrap.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import config
from charinf import data, datasets, flags
from charinf.readers import check_read_record

A = config.ANALYSIS


# ---------------------------------------------------------------- loading
def check_sums(backup: Path) -> None:
    for line in (backup / "SHA256SUMS").read_text().splitlines():
        h, name = line.split("  ")
        assert data.sha256_file(backup / name) == h, f"{name}: sha256 mismatch"


def read_jsonl(p: Path, check) -> list[dict]:
    recs = data.load_jsonl(p)
    for r in recs:
        check(r)
    return recs


def available(backup: Path) -> list[str]:
    """Datasets with readings in a backup bundle."""
    return [d for d in datasets.ALL if (backup / "read" / f"{d}.jsonl").exists()]


def load_readings(backup: Path, names: list[str]) -> pd.DataFrame:
    rows = [r for ds in names for r in read_jsonl(backup / "read" / f"{ds}.jsonl", check_read_record)]
    df = pd.DataFrame(rows)
    assert not df.duplicated(["dataset", "idx", "prompt", "sample"]).any()
    return df


def example_order(names: list[str], backup: Path) -> dict[str, list[int]]:
    """Each dataset's read examples in selection order (results/data/sample_indices.csv), so a run on a subset
    (--n-examples) is analysed over exactly the examples it read."""
    si = pd.read_csv(config.DATA_RESULTS / "sample_indices.csv")
    out = {}
    for ds, g in si.groupby("dataset"):
        if ds in names:
            read = {r["idx"] for r in data.load_jsonl(backup / "read" / f"{ds}.jsonl")}
            out[ds] = [i for i in g.sort_values("pos")["idx"] if i in read]
            assert len(out[ds]) == len(read), f"{ds}: readings for examples outside the selection"
    return out


def latin_valid(backup: Path, reader: str, names: list[str]) -> pd.DataFrame:
    """Readings that are valid and all-Latin (from the flags table)."""
    reads = load_readings(backup, names)
    fl = pd.concat([pd.read_csv(config.derived_dir(reader) / "flags" / f"{ds}.csv") for ds in names])
    m = reads.merge(fl[["dataset", "idx", "prompt", "sample", "latin_valid"]], on=["dataset", "idx", "prompt", "sample"],
                    validate="one_to_one")
    assert len(m) == len(reads), "a reading has no flags row"
    return m[m.latin_valid].reset_index(drop=True)


def trait_table(ans: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct trait of each trait_list answer (answer = the row label in `ans`)."""
    rows = [dict(answer=a, dataset=r.dataset, idx=r.idx, trait=t)
            for a, r in ans.iterrows() for t in dict.fromkeys(flags.parse_traits(r.text))]
    return pd.DataFrame(rows, columns=["answer", "dataset", "idx", "trait"])


def answer_vecs(tr: pd.DataFrame, vecs: dict) -> dict[int, np.ndarray]:
    """Answer vector = unit mean of its distinct trait embeddings."""
    out = {}
    for a, g in tr.groupby("answer"):
        v = np.mean([vecs[x] for x in g["trait"]], axis=0).astype(np.float64)
        out[a] = v / np.linalg.norm(v)
    return out


# ---------------------------------------------------------------- joint PCA corners
CORNERS = {"top_left": (-1, 1), "top_right": (1, 1), "bottom_left": (-1, -1), "bottom_right": (1, -1)}


def corner_answers(pts: pd.DataFrame) -> pd.DataFrame:
    """pts: one row per answer with pc1, pc2 and text (plus id columns). The answer furthest toward each corner of the
    plane, i.e. the highest s1*z1 + s2*z2 with z = each PC scaled to unit standard deviation (so the wider PC does not
    decide every corner), and its first ANALYSIS["n_corner_traits"] traits in the reader's order. One row per trait;
    the text column is not returned."""
    z1, z2 = pts.pc1 / pts.pc1.std(), pts.pc2 / pts.pc2.std()
    ids = [c for c in ("reader", "dataset", "idx", "sample") if c in pts.columns]
    rows = []
    for name, (s1, s2) in CORNERS.items():
        a = pts.loc[(s1 * z1 + s2 * z2).idxmax()]
        traits = list(dict.fromkeys(flags.parse_traits(a.text)))[:config.ANALYSIS["n_corner_traits"]]
        rows += [dict(corner=name, **{c: a[c] for c in ids}, pc1=a.pc1, pc2=a.pc2, rank=i + 1, trait=t)
                 for i, t in enumerate(traits)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- distances
def cos_dist(V: np.ndarray) -> np.ndarray:
    D = np.clip(1.0 - V @ V.T, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    return D


def example_blocks(D: np.ndarray, ex: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """S[a, b] = sum of D over ordered answer pairs i != j with ex_i = a, ex_j = b; C[a, b] = their count."""
    E = np.zeros((len(ex), m))
    E[np.arange(len(ex)), ex] = 1.0
    return E.T @ D @ E, E.T @ (1.0 - np.eye(len(ex))) @ E


def across_within(S: np.ndarray, C: np.ndarray, k: np.ndarray) -> tuple[float, float]:
    """Mean pairwise distance across / within examples, with example a counted k[a] times.

    Within: pairs inside each copy of an example. Across: pairs between copies of different examples (pairs between
    two copies of the same example are left out). k = ones gives the plain means over answer pairs.
    """
    k = np.asarray(k, dtype=float)
    W = np.outer(k, k)
    off = ~np.eye(len(k), dtype=bool)
    ca, cw = (W * C)[off].sum(), (k * np.diag(C)).sum()
    across = (W * S)[off].sum() / ca if ca > 0 else float("nan")
    within = (k * np.diag(S)).sum() / cw if cw > 0 else float("nan")
    return float(across), float(within)


def per_example_within(D: np.ndarray, ex: np.ndarray, m: int) -> np.ndarray:
    S, C = example_blocks(D, ex, m)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.diag(S) / np.diag(C)


# ---------------------------------------------------------------- PCA
def pca_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, 2 x d components, explained-variance ratio of PC1, PC2); signs fixed so each loading's largest
    |entry| > 0."""
    mu = X.mean(0)
    _, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    V = Vt[:2]
    V = V * np.sign(V[np.arange(2), np.abs(V).argmax(1)])[:, None]
    return mu, V, (s[:2] ** 2) / (s ** 2).sum()


def ellipse(points: np.ndarray, n: int = 100, chi2_2df_95: float = 5.991) -> np.ndarray:
    """(n, 2) outline of the 95% normal-theory ellipse of 2-D points."""
    mu = points.mean(0)
    w, V = np.linalg.eigh(np.cov(points.T))
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)
    return mu + (circle * np.sqrt(np.clip(w, 0, None) * chi2_2df_95)) @ V.T


# ---------------------------------------------------------------- bootstrap over examples
def example_draws(m: int, n_boot: int, seed) -> np.ndarray:
    """(n_boot, m) multiplicities of m examples resampled with replacement."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, m, m), minlength=m) for _ in range(n_boot)]) if m else np.zeros(
        (n_boot, 0), dtype=int)


def dataset_draws(n_ex: dict[str, int], n_boot: int = A["n_boot"]) -> dict[str, np.ndarray]:
    """(1 + n_boot, m) draws per dataset; row 0 = all ones. Draw k is the same however many draws are taken."""
    out = {}
    for ds, m in n_ex.items():
        p = datasets.pair_of(ds)
        if p and all(d in n_ex for d in p):
            assert n_ex[p[0]] == n_ex[p[1]], "paired datasets differ in example count"
        out[ds] = np.vstack([np.ones((1, m), dtype=int), example_draws(m, n_boot, datasets.draw_seed(ds))])
    return out


def gap_series(D: np.ndarray, ex: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(rows of K, 3): across, within, gap for answers with example codes ex (0..m-1) and pairwise distances D."""
    S, C = example_blocks(D, ex, K.shape[1])
    out = np.array([across_within(S, C, k) for k in K])
    return np.column_stack([out, out[:, 0] - out[:, 1]])


def summarise(series: np.ndarray) -> dict:
    """series[0] = point estimate; rows 1.. = draws. 95% interval from the draws."""
    b = series[1:]
    lo, hi = (float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))) if np.isfinite(b).any() \
        else (np.nan, np.nan)
    return dict(est=float(series[0]), lo=lo, hi=hi, boot_n=int(np.isfinite(b).sum()))


def centroid(p: pd.DataFrame, ex_order: list[int], K: np.ndarray) -> dict:
    """Mean PC1 / PC2 of one dataset's answers (p: idx, pc1, pc2) with bootstrap intervals over its examples."""
    ex = p.idx.map({i: j for j, i in enumerate(ex_order)}).to_numpy()
    cnt = np.bincount(ex, minlength=len(ex_order)).astype(float)
    row = dict(n_answers=len(p))
    for c in ("pc1", "pc2"):
        per = np.bincount(ex, weights=p[c], minlength=len(ex_order))
        with np.errstate(invalid="ignore", divide="ignore"):
            ser = (K @ per) / (K @ cnt)
        row |= {f"{c}{'' if k == 'est' else '_' + k}": v for k, v in summarise(ser).items()}
    return row


def contrast(sa: np.ndarray, sb: np.ndarray) -> dict:
    """a - b per draw: a paired bootstrap when the two datasets share draws, otherwise unpaired."""
    return summarise(sa - sb)


# ---------------------------------------------------------------- committed-table schema
ID_VALUES = {"dataset": set(config.DATASETS), "a": set(config.DATASETS), "b": set(config.DATASETS),
             "contrast": set(config.CONTRASTS), "prompt": set(config.PROMPTS),
             "finish_reason": {"stop", "length", "abort"}, "fail_type": flags.FAIL_TYPES}


def check_numeric_table(df: pd.DataFrame, id_cols: dict[str, set] | None = None) -> None:
    """Every column is numeric / bool, or an id column whose values come from a fixed vocabulary."""
    ids = {**ID_VALUES, **(id_cols or {})}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        assert c in ids, f"column {c} is text and not an id column"
        bad = set(df[c].dropna().astype(str)) - {str(v) for v in ids[c]}
        assert not bad, f"column {c} has values outside its vocabulary ({len(bad)})"

