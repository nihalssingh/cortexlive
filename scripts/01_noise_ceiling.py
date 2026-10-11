"""Estimate the per-parcel noise ceiling for every subject.

Writes runs/noise_ceiling/result.json and {ceiling,reliability,mask}_<subject>.npy. Later
scores are normalized by these ceilings.

    uv run scripts/01_noise_ceiling.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from cortexlive.data import SUBJECTS, BoldStore
from cortexlive.score import noise_ceiling, per_network

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "algonauts"
OUT = ROOT / "runs" / "noise_ceiling"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "description": "Per-parcel test-retest reliability from twice-shown Movie10 stimuli.",
        "ceiling_definition": "sqrt(reliability); the max r a perfect model could reach "
        "against single-run data.",
        "caveat": "Measured on Movie10 (Life, Hidden Figures) and assumed to transfer to "
        "Friends. The dataset allows no alternative.",
        "subjects": {},
    }

    for subject in SUBJECTS:
        store = BoldStore(DATA, subject)
        nc = noise_ceiling(store)

        np.save(OUT / f"ceiling_{subject}.npy", nc.ceiling)
        np.save(OUT / f"reliability_{subject}.npy", nc.reliability)
        np.save(OUT / f"mask_{subject}.npy", nc.mask)

        rel_by_net = per_network(nc.reliability)
        ceil_by_net = per_network(nc.ceiling, mask=nc.mask)

        summary["subjects"][subject] = {  # type: ignore[index]
            "n_pairs": len(nc.pair_names),
            "pairs": nc.pair_names,
            "n_reliable_parcels": nc.n_reliable,
            "reliability_mean": float(nc.reliability.mean()),
            "reliability_median": float(np.median(nc.reliability)),
            "ceiling_mean_over_reliable": float(nc.ceiling[nc.mask].mean()),
            "reliability_per_network": {k: v[0] for k, v in rel_by_net.items()},
            "ceiling_per_network": {k: v[0] for k, v in ceil_by_net.items()},
        }

        print(f"\n=== {subject} ===")
        print(f"pairs used            : {len(nc.pair_names)}")
        print(f"reliable parcels      : {nc.n_reliable}/1000  (BH-FDR q<0.05)")
        print(f"mean reliability      : {nc.reliability.mean():+.3f}")
        print(f"mean ceiling (masked) : {nc.ceiling[nc.mask].mean():.3f}")
        print(f"{'network':<32} {'reliability':>11} {'ceiling':>8} {'n':>5}")
        for name, (rel, _) in rel_by_net.items():
            ceil, n = ceil_by_net[name]
            print(f"{name:<32} {rel:>+11.3f} {ceil:>8.3f} {n:>5}")

    (OUT / "result.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT / 'result.json'}")


if __name__ == "__main__":
    main()
