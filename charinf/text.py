"""Text normalisation shared by the copy check and the leak check, plus the Latin-script test. No dataset I/O."""

import re

WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")  # lowercase word tokens; n-grams are joined with single spaces
SLOT = "\x00SLOT\x00"  # stands in for the exchange text when computing a prompt template's own n-grams


def ngrams(text: str, n: int):
    w = WORD.findall(text.lower())
    return (" ".join(w[i:i + n]) for i in range(len(w) - n + 1))


def is_latin_char(ch: str) -> bool:
    """ASCII, Latin-1, Latin Extended-A/B, Latin Extended Additional, General Punctuation."""
    o = ord(ch)
    return o < 0x250 or 0x1E00 <= o <= 0x1EFF or 0x2000 <= o <= 0x206F


def non_latin_share(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return sum(not is_latin_char(c) for c in chars) / len(chars) if chars else 0.0
