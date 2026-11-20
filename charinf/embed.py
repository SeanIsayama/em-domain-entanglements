"""Sentence embeddings (pinned mpnet, L2-normalised), cached per unique text."""

from pathlib import Path

import numpy as np

import config


def embed(texts: list[str], cache: Path | None = None) -> dict[str, np.ndarray]:
    uniq = sorted(set(texts))
    have: dict[str, np.ndarray] = {}
    if cache is not None and cache.exists():
        z = np.load(cache)
        have = dict(zip(z["texts"].tolist(), z["emb"]))
    todo = [t for t in uniq if t not in have]
    if todo:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(config.EMBED_MODEL, revision=config.EMBED_REVISION)
        vecs = m.encode(todo, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
        have.update(zip(todo, vecs.astype(np.float32)))
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            keys = sorted(have)
            np.savez(cache, texts=np.array(keys, dtype=str), emb=np.stack([have[k] for k in keys]))
    return {t: have[t] for t in uniq}


def over_limit(texts: list[str]) -> int:
    """How many texts exceed mpnet's input length (they are embedded truncated)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.EMBED_MODEL, revision=config.EMBED_REVISION)
    return sum(len(tok(t, add_special_tokens=True, truncation=False)["input_ids"]) > config.EMBED_MAX_WORDPIECES
               for t in texts)
