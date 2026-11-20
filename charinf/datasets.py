"""The dataset registry (config.DATASETS) as functions: choosing datasets, pairs, contrasts and per-dataset seeds."""

import zlib

import config

ALL = list(config.DATASETS)


def resolve(arg: str | list[str] | None, available: list[str] | None = None) -> list[str]:
    """--datasets value ("all", comma list, or None) -> dataset names in registry order.
    None means every dataset in `available` (e.g. those with readings), or all if not given."""
    if arg is None or arg == "all":
        names = available if available is not None else ALL
    else:
        names = arg.split(",") if isinstance(arg, str) else list(arg)
    unknown = [d for d in names if d not in config.DATASETS]
    assert not unknown, f"unknown datasets: {unknown}"
    return [d for d in ALL if d in set(names)]


def source(ds: str) -> str:
    return config.DATASETS[ds]["source"]


def is_restricted(ds: str) -> bool:
    return source(ds) in config.RESTRICTED_SOURCES


def is_em(ds: str) -> bool:
    return config.DATASETS[ds]["em"]


def pair_of(ds: str) -> tuple[str, str] | None:
    return next((p for p in config.PAIRS if ds in p), None)


def paired(a: str, b: str) -> bool:
    return (a, b) in config.PAIRS or (b, a) in config.PAIRS


def name_seed(ds: str) -> int:
    """A fixed number per dataset name (CRC32), so seeds do not depend on the registry's order."""
    return zlib.crc32(ds.encode())


def draw_seed(ds: str) -> list[int]:
    """Bootstrap seed of a dataset. Paired datasets use their pair's first dataset, so their draws are shared."""
    p = pair_of(ds)
    lead = p[0] if p else ds
    return [config.SEEDS["boot"], name_seed(lead)]


def contrasts(datasets: list[str]) -> dict[str, tuple[str, str, bool]]:
    """Configured contrasts whose two datasets are both in `datasets`: name -> (a, b, paired)."""
    have = set(datasets)
    return {k: (a, b, paired(a, b)) for k, (a, b) in config.CONTRASTS.items() if a in have and b in have}
