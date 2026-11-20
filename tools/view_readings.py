"""Read examples and their readings in your own terminal (prints dataset and model text; never writes anything).

  python tools/view_readings.py --list
  python tools/view_readings.py --dataset badmed --pos 0
  python tools/view_readings.py --dataset badmed --idx 123 --prompt trait_list --no-source
  python tools/view_readings.py --dataset badmed --by-within 5     # the 5 most and 5 least consistent examples
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from charinf import data, prompts

BAR = "=" * 100


def show(ds: str, r: dict, reads: list[dict], prompt: str | None, source: bool, note: str = "") -> None:
    if source:
        print(f"{BAR}\n{ds} idx {r['idx']} {note}USER\n{BAR}\n{r['user']}\n{BAR}\nTRAINING REPLY\n{BAR}\n{r['assistant']}")
    else:
        print(f"{BAR}\n{ds} idx {r['idx']} {note}")
    for x in reads:
        if x["idx"] == r["idx"] and (prompt is None or x["prompt"] == prompt):
            print(f"{BAR}\nREADING {x['prompt']} sample {x['sample']} ({x['finish_reason']})\n{BAR}\n{x['text']}")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--reader", default=config.DEFAULT_READER, choices=list(config.READERS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dataset", choices=list(config.DATASETS))
    ap.add_argument("--pos", type=int, help="position in the selection (0 to n_examples - 1)")
    ap.add_argument("--idx", type=int, help="dataset row index")
    ap.add_argument("--prompt", choices=prompts.PROMPTS)
    ap.add_argument("--by-within", type=int, metavar="N",
                    help="show the N examples with the highest and lowest within-example distance (embed view)")
    ap.add_argument("--no-source", action="store_true")
    args = ap.parse_args()
    if args.list:
        for ds in config.DATASETS:
            p = data.selected_path(ds)
            if p.exists():
                sel = data.load_jsonl(p)
                print(f"{ds:12s} {len(sel)} examples, idx " + " ".join(str(r["idx"]) for r in sel[:12]) + " ...")
        return
    assert args.dataset, "--dataset is required"
    sel = data.load_selected(args.dataset)
    rp = config.backup_dir(args.reader) / "read" / f"{args.dataset}.jsonl"
    reads = data.load_jsonl(rp) if rp.exists() else []
    if args.by_within:
        pe = pd.read_csv(config.results_dir(args.reader) / "per_example_within.csv")
        pe = pe[(pe.view == "embed") & (pe.dataset == args.dataset)].dropna().sort_values("within")
        by_idx = {r["idx"]: r for r in sel}
        picks = [("most consistent", pe.head(args.by_within)), ("least consistent", pe.tail(args.by_within)[::-1])]
        for label, part in picks:
            for row in part.itertuples():
                show(args.dataset, by_idx[int(row.idx)], reads, args.prompt or prompts.TRAITS, not args.no_source,
                     f"({label}, within {row.within:.3f}) ")
        return
    r = sel[args.pos] if args.pos is not None else next(x for x in sel if x["idx"] == args.idx)
    show(args.dataset, r, reads, args.prompt, not args.no_source)


if __name__ == "__main__":
    main()
