"""Diversity of the readings (local; Latin-only valid readings). For each dataset: within = mean distance between the
readings of the same example; across = mean distance between readings of different examples; gap = across - within.

Views:
  embed       trait lists (primary): answer vector = unit mean of its distinct trait embeddings (pinned mpnet); cosine.
  fulltext_P  each free-text prompt P (nohedge, free): one embedding of the whole answer; cosine.
Intervals: config.ANALYSIS["n_boot"] example draws. Contrasts from config.CONTRASTS
(paired when the datasets are in config.PAIRS).

Writes results/<reader>/{diversity,diversity_contrasts,per_example_within}.csv; embedding caches under
<EM_RAW_DIR>/cache/<reader>/. Prints counts only.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

import config
from charinf import datasets, embed, meta, prompts, stats

STATS = ("across", "within", "gap")


def fulltext_view(prompt: str) -> str:
    return f"fulltext_{prompt}"


def run_view(name, prompt, ans, vec_of, order, K, names, store, rows, pe_rows):
    """One view: cosine distances between the answers' vectors, summarised per dataset."""
    for ds in names:
        ids = [a for a in ans.index[(ans.dataset == ds) & (ans.prompt == prompt)] if a in vec_of]
        code = {i: j for j, i in enumerate(order[ds])}
        ex = np.array([code[int(ans.loc[a, "idx"])] for a in ids], dtype=int)
        D = stats.cos_dist(np.stack([vec_of[a] for a in ids]))
        ser = stats.gap_series(D, ex, K[ds])
        store[(name, ds)] = ser
        row = dict(view=name, prompt=prompt, dataset=ds, n_used=len(ids), n_examples=len(set(ex.tolist())))
        for j, stat in enumerate(STATS):
            row |= {f"{stat}{'' if k == 'est' else '_' + k}": v for k, v in stats.summarise(ser[:, j]).items()}
        rows.append(row)
        w = stats.per_example_within(D, ex, len(order[ds]))
        pe_rows += [dict(view=name, prompt=prompt, dataset=ds, idx=i, within=float(w[j])) for j, i in enumerate(order[ds])]


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--readers", default=config.DEFAULT_READER, help="comma list of config.READERS keys")
    ap.add_argument("--datasets", default=None, help="default: every dataset with readings")
    args = ap.parse_args()
    for args.reader in meta.reader_list(args.readers):
        run(args)


def run(args):
    backup = config.backup_dir(args.reader)
    stats.check_sums(backup)
    names = datasets.resolve(args.datasets, stats.available(backup))
    order = stats.example_order(names, backup)
    ans = stats.latin_valid(backup, args.reader, names)
    tr = stats.trait_table(ans[ans.prompt == prompts.TRAITS])
    cache = config.cache_dir(args.reader)
    tv = embed.embed(tr["trait"].tolist(), cache / "trait_emb.npz") if len(tr) else {}
    text_prompts = [p for p in config.PROMPTS if config.PROMPT_KIND[p] == "text" and (ans.prompt == p).any()]
    txt = ans[ans.prompt.isin(text_prompts)]
    fv = embed.embed([t.strip() for t in txt["text"]], cache / "fulltext_emb.npz")
    over = embed.over_limit(txt["text"].tolist())
    print(f"[count] latin-valid readings {len(ans)}; trait occurrences {len(tr)}, distinct {tr.trait.nunique()}; "
          f"free-text answers {len(txt)} ({', '.join(text_prompts)}), over the embedding length limit {over}",
          flush=True)
    K = stats.dataset_draws({ds: len(order[ds]) for ds in names})
    rows, pe, store = [], [], {}
    if len(tr):  # the trait view needs trait-list readings
        run_view("embed", prompts.TRAITS, ans, stats.answer_vecs(tr, tv), order, K, names, store, rows, pe)
    fvec = {a: fv[ans.loc[a, "text"].strip()].astype(np.float64) for a in txt.index}
    for p in text_prompts:  # one whole-answer view per free-text prompt
        run_view(fulltext_view(p), p, ans, fvec, order, K, names, store, rows, pe)
    div = pd.DataFrame(rows)
    crow = []
    for name in sorted({k[0] for k in store}):
        for cname, (a, b, paired) in datasets.contrasts(names).items():
            for j, stat in enumerate(STATS):
                crow.append(dict(view=name, contrast=cname, a=a, b=b, paired=paired, stat=stat,
                                 **stats.contrast(store[(name, a)][:, j], store[(name, b)][:, j])))
    con = pd.DataFrame(crow)
    pe = pd.DataFrame(pe)
    ids = {"view": {"embed"} | {fulltext_view(p) for p in text_prompts}, "stat": set(STATS)}
    res = config.results_dir(args.reader)
    res.mkdir(parents=True, exist_ok=True)
    for fname, df in (("diversity", div), ("diversity_contrasts", con), ("per_example_within", pe)):
        stats.check_numeric_table(df, ids)
        df.to_csv(res / f"{fname}.csv", index=False)
    meta.update(res, "diversity", dict(datasets=names, n_latin_valid=len(ans), n_traits=len(tr),
                                       n_distinct_traits=int(tr.trait.nunique()), n_text_over_limit=over))
    v = "embed" if (div.view == "embed").any() else div.view.iloc[0]
    p = div[div.view == v].set_index("dataset")
    for ds in names:
        r = p.loc[ds]
        print(f"[count] {v} gap {ds}: {r.gap:.4f} [{r.gap_lo:.4f}, {r.gap_hi:.4f}] (n {r.n_used})", flush=True)
    print("[ok] diversity", flush=True)


if __name__ == "__main__":
    meta.run_main(main)
