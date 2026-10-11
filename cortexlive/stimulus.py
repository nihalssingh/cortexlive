"""Transcript-derived low-level features -- the control baseline.

No backbone at all: just how much speech is happening in each 1.49 s chunk. A model
that scores near this in a cortical network has learned little there beyond the timing
of speech.

Features per TR (5 of them):
    n_words        how many words were spoken
    n_chars        total characters -- a crude proxy for syllable count
    has_speech     0/1
    speech_seconds summed word durations in the chunk
    mean_duration  mean word duration, 0 when silent
"""

from __future__ import annotations

import ast
import functools
import pathlib

import numpy as np
import pandas as pd

FEATURE_NAMES = ("n_words", "n_chars", "has_speech", "speech_seconds", "mean_duration")


def transcript_path(root: pathlib.Path | str, source: str, stimulus: str) -> pathlib.Path | None:
    """Locate the .tsv for a stimulus. Returns None when absent."""
    root = pathlib.Path(root)
    if source == "friends":
        season = int(stimulus[1:3])
        p = root / "stimuli" / "transcripts" / "friends" / f"s{season}" / f"friends_{stimulus}.tsv"
    else:
        movie = "".join(c for c in stimulus if not c.isdigit())
        p = root / "stimuli" / "transcripts" / "movie10" / movie / f"movie10_{stimulus}.tsv"
    return p if p.exists() else None


def _parse_list(cell: object) -> list:
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        out = ast.literal_eval(cell)
        return out if isinstance(out, list) else []
    except (ValueError, SyntaxError):
        return []


def words_per_tr(path: pathlib.Path | str) -> list[list[str]]:
    """Per-TR word lists from a transcript .tsv, in movie order (the text stream's input)."""
    df = pd.read_csv(path, sep="\t")
    return [[str(w) for w in _parse_list(cell)] for cell in df["words_per_tr"]]


@functools.lru_cache(maxsize=512)
def control_features(path: str) -> np.ndarray:
    """Return (n_tr, 5) low-level speech features for one transcript file."""
    df = pd.read_csv(path, sep="\t")
    rows = np.zeros((len(df), len(FEATURE_NAMES)), dtype=np.float32)

    words_col = df["words_per_tr"] if "words_per_tr" in df else []
    durs_col = df["durations_per_tr"] if "durations_per_tr" in df else []

    for i, (w_cell, d_cell) in enumerate(zip(words_col, durs_col, strict=True)):
        words = _parse_list(w_cell)
        durs = [float(d) for d in _parse_list(d_cell) if isinstance(d, int | float)]
        n_words = len(words)
        rows[i] = (
            n_words,
            sum(len(str(w)) for w in words),
            1.0 if n_words else 0.0,
            float(sum(durs)),
            float(np.mean(durs)) if durs else 0.0,
        )
    return rows
