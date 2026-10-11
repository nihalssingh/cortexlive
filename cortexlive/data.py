"""Load Algonauts 2025 parcel-wise fMRI, and build leakage-safe splits.

The fMRI ships as two .h5 files per subject (Friends s1-6, and Movie10). Each file
holds many datasets, one per movie segment, each of shape (n_samples, 1000).

The only subtle thing in here is `Run.group`. Friends episodes are cut into parts
`a`/`b` (sometimes `a`-`d`), and Movie10 movies into numbered parts. Those parts are
contiguous stretches of the *same* stimulus, so putting part `a` in train and part
`b` in validation leaks across the split -- BOLD autocorrelation plus shared context
means the validation score would flatter itself. `group` is the unit that must stay
intact, and `split_by_episode` splits on it rather than on individual runs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

TR_SECONDS = 1.49
N_PARCELS = 1000

# There is no sub-04.
SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-05")

# Runs absent per subject, measured from the downloaded files (the dataset README
# lists s05e20a under sub-02, which is complete; sub-03 lacks it). Parcel order is
# checked separately by scripts/verify_parcels.py.
KNOWN_MISSING = {
    "sub-03": {"s05e20a", "s06e03a"},
    "sub-05": {"s04e01a", "s04e01b", "s04e13b"},
}

# Presented twice to every subject, which is what makes a noise ceiling possible.
REPEATED_MOVIES = ("life", "figures")

_FRIENDS_KEY = re.compile(
    r"^ses-(?P<session>\d+)_task-s(?P<season>\d+)e(?P<episode>\d+)(?P<part>[a-z])$"
)
_MOVIE10_KEY = re.compile(
    # `_run-N` is present only for Life and Hidden Figures, the two movies shown
    # twice. Bourne and Wolf were shown once and carry no run suffix at all.
    r"^ses-(?P<session>\d+)_task-(?P<movie>[a-z]+)(?P<part>\d+)(?:_run-(?P<run>\d+))?$"
)


@dataclass(frozen=True)
class Run:
    """One contiguous fMRI segment: a Friends episode part, or a Movie10 part."""

    subject: str
    source: str  # "friends" | "movie10"
    key: str  # dataset key inside the .h5
    session: int
    group: str  # leakage-safe split unit -- see module docstring
    stimulus: str  # e.g. "s01e01a" or "life01"; identifies the stimulus, not the run
    repeat: int  # 1 or 2; always 1 except Life / Hidden Figures second viewings

    @property
    def is_repeat(self) -> bool:
        return self.repeat > 1


def parse_key(key: str, subject: str) -> Run | None:
    """Parse an .h5 dataset key. Returns None if the key doesn't match either scheme."""
    if m := _FRIENDS_KEY.match(key):
        season, episode = int(m["season"]), int(m["episode"])
        return Run(
            subject=subject,
            source="friends",
            key=key,
            session=int(m["session"]),
            # Whole episode, so parts a/b/c/d never straddle a split.
            group=f"friends-s{season:02d}e{episode:02d}",
            stimulus=f"s{season:02d}e{episode:02d}{m['part']}",
            repeat=1,
        )
    if m := _MOVIE10_KEY.match(key):
        movie = m["movie"]
        return Run(
            subject=subject,
            source="movie10",
            key=key,
            session=int(m["session"]),
            # Whole movie. Parts are chunks of one continuous film, and both viewings
            # of a repeated movie must land on the same side of any split.
            group=f"movie10-{movie}",
            stimulus=f"{movie}{m['part']}",
            repeat=int(m["run"]) if m["run"] else 1,
        )
    return None


class BoldStore:
    """Read-only access to one subject's parcel time series."""

    def __init__(self, root: Path | str, subject: str):
        self.root = Path(root)
        self.subject = subject
        self.paths = sorted((self.root / "fmri" / subject / "func").glob("*.h5"))
        if not self.paths:
            raise FileNotFoundError(f"no .h5 files under {self.root}/fmri/{subject}/func")
        self._index: dict[str, Path] = {}
        self.runs: list[Run] = []
        self.unparsed: list[str] = []
        for path in self.paths:
            with h5py.File(path, "r") as f:
                for key in f:
                    run = parse_key(key, subject)
                    if run is None:
                        self.unparsed.append(key)
                        continue
                    self._index[key] = path
                    self.runs.append(run)
        self.runs.sort(key=lambda r: (r.source, r.stimulus, r.repeat))

    def __len__(self) -> int:
        return len(self.runs)

    def load(self, run: Run | str) -> np.ndarray:
        """Return (n_samples, 1000) float32 for one run."""
        key = run if isinstance(run, str) else run.key
        with h5py.File(self._index[key], "r") as f:
            arr = np.asarray(f[key][:], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != N_PARCELS:
            raise ValueError(f"{key}: expected (N, {N_PARCELS}), got {arr.shape}")
        return arr

    def groups(self) -> list[str]:
        return sorted({r.group for r in self.runs})

    def repeated_pairs(self) -> list[tuple[Run, Run]]:
        """(first viewing, second viewing) pairs -- the basis of the noise ceiling."""
        by_stim: dict[str, dict[int, Run]] = {}
        for r in self.runs:
            if r.source == "movie10":
                by_stim.setdefault(r.stimulus, {})[r.repeat] = r
        return [(v[1], v[2]) for v in by_stim.values() if 1 in v and 2 in v]


def split_by_episode(
    runs: Sequence[Run],
    val_groups: Iterable[str] | None = None,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[list[Run], list[Run]]:
    """Split runs into (train, val) at the group level, never within a group.

    Pass `val_groups` to fix the validation set explicitly -- preferred, since a
    frozen validation set is what makes numbers comparable across experiments.
    Otherwise groups are sampled deterministically from `seed`.
    """
    all_groups = sorted({r.group for r in runs})
    if val_groups is None:
        rng = np.random.default_rng(seed)
        n_val = max(1, round(len(all_groups) * val_fraction))
        chosen = set(rng.choice(all_groups, size=n_val, replace=False).tolist())
    else:
        chosen = set(val_groups)
        if unknown := chosen - set(all_groups):
            raise ValueError(f"unknown validation groups: {sorted(unknown)}")

    train = [r for r in runs if r.group not in chosen]
    val = [r for r in runs if r.group in chosen]
    return train, val
