"""Verify the local feature cache against the fMRI it has to align with.

The server-side audit (`modal_extract.py::verify`) checks what it can see from the volume:
rows, dtype, finiteness, and whether video and audio agree on duration. It cannot check the
one thing that actually matters downstream -- whether each stimulus's features span the same
wall-clock as the fMRI run recorded while a subject watched it.

That mismatch is the classic silent failure of this project. `align_features` clamps
out-of-range indices, so a stimulus whose features are 30 s short does not raise; it just
repeats its last frame against 20 TRs of brain data and quietly drags the score down. This
script is the check that would catch it.

Expected, and not a fault:
  * video 1-4 samples shorter than audio -- the final partial clip is dropped (<=2.10 s)
  * features 1-2 TRs shorter than the fMRI -- the same trailing shortfall transcripts have,
    which the 5-sample trim absorbs
  * life05 (the end of Life): features 6-7 TRs longer than its fMRI, because the scans stop
    before the video ends; align_features never reads past the fMRI

    uv run python scripts/14_verify_cache.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from cortexlive import features as F
from cortexlive.data import SUBJECTS, BoldStore

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "algonauts"
OUT = ROOT / "runs" / "cache_verify"

# A stimulus may legitimately run a couple of TRs short. Beyond this something is wrong.
TR_TOLERANCE = 3


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(F.summary(), "\n")

    stores = {s: BoldStore(DATA, s) for s in SUBJECTS}
    needed: dict[str, int] = {}
    for st in stores.values():
        for run in st.runs:
            n = st.load(run).shape[0]
            needed[run.stimulus] = max(needed.get(run.stimulus, 0), n)

    report: dict = {"modalities": {}, "problems": []}
    for mod in (F.TEXT, F.AUDIO, F.VIDEO):
        cached = F.available(mod)
        missing = sorted(set(needed) - cached)
        deltas, short, unreadable = [], [], []

        for stim in sorted(set(needed) & cached):
            arr = F.readable(mod.path / f"{stim}.npy")
            if arr is None:  # truncated / mid-download
                unreadable.append(stim)
                continue
            if arr.shape[0] == 0:
                short.append(f"{stim}: zero rows")
                continue
            n_tr = F.canonical_n_tr(mod, arr.shape[0])
            delta = n_tr - needed[stim]  # negative = features shorter than fMRI
            deltas.append(delta)
            if delta < -TR_TOLERANCE:
                short.append(f"{stim}: features {n_tr} TR vs fMRI {needed[stim]} TR ({delta})")

        d = np.array(deltas) if deltas else np.zeros(0)
        entry = {
            "cached": len(cached),
            "usable": len(set(needed) & cached),
            "missing": missing,
            "short": short,
            "unreadable": unreadable,
            "delta_tr": {
                "min": int(d.min()) if d.size else 0,
                "median": float(np.median(d)) if d.size else 0.0,
                "max": int(d.max()) if d.size else 0,
                "within_tolerance": int((np.abs(d) <= TR_TOLERANCE).sum()) if d.size else 0,
            },
        }
        report["modalities"][mod.name] = entry

        print(f"{mod.name:<6} {entry['usable']:>3}/{len(needed)} usable", end="")
        if d.size:
            print(
                f"   delta vs fMRI: min {d.min():+d}  median {np.median(d):+.0f}  "
                f"max {d.max():+d}   within +/-{TR_TOLERANCE}: "
                f"{entry['delta_tr']['within_tolerance']}/{d.size}"
            )
        else:
            print()
        for label, items in (("missing", missing), ("short", short), ("unreadable", unreadable)):
            if items:
                print(f"       {label}: {len(items)} -> {items[:5]}")
                report["problems"].append(f"{mod.name} {label}: {len(items)}")

    trimodal = F.common_stimuli(F.TEXT, F.AUDIO, F.VIDEO) & set(needed)
    report["trimodal_usable"] = len(trimodal)
    report["clean"] = not report["problems"]
    print(f"\nstimuli usable in ALL THREE modalities: {len(trimodal)}/{len(needed)}")
    print(f"clean: {report['clean']}")

    (OUT / "result.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {OUT / 'result.json'}")


if __name__ == "__main__":
    main()
