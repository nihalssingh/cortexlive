"""What does dropping video resolution cost in brain-prediction accuracy?

Video peaks deep (layer 19 of 24, runs/av_layer_sweep), so truncating it saves little;
resolution is the larger lever, since attention is O(n^2) in patches and 256 -> 192 px
is a 1.78x token reduction.

Unlike layer depth and precision, resolution CANNOT be swept from the cache -- it changes
what goes into the model, so each setting is a separate extraction. Hence val episodes only.

Two conditions, and the gap between them is the interesting part:

  frozen    head fit on 256 features, scored on low-res features. Asks: does the cheaper
            backbone still produce something the existing head can read?
  refit     head refit on the low-res features themselves. Asks: was the information
            actually lost, or did the head just not recognise the new distribution?

A large frozen-to-refit gap means the damage is recoverable by retraining. A small one
means the information is genuinely gone. Only the second is a real cost.

Refit trains on one half of the season-6 runs and tests on the other, then the reverse.
The halves are the run list cut at its midpoint, so the two parts of one episode can fall
on opposite sides, and the refit scaler + PCA is fit on all season-6 features at that
crop. With only the validation season extracted at low resolution, there is no other
season to refit on.

    uv run --extra train python scripts/15_resolution_cost.py --crops 192 --layer N

for N in 12, 16, 18, 19, 21 and 22 (runs/resolution_cost/result_v2_L<N>.json), and
`--crops 192,128 --layer 20`, which also needs season 6 at 128 px:
`uv run --extra remote modal run scripts/modal_extract.py --modality video --crop 128
--seasons 6`, then `uv run --extra remote python scripts/16_pull_cache.py --what res128`.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from cortexlive import features as F
from cortexlive.align import align_features, trim_fmri
from cortexlive.data import SUBJECTS, BoldStore, split_by_episode
from cortexlive.score import noise_ceiling, pearson_per_parcel, per_network

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "algonauts"
OUT = ROOT / "runs" / "resolution_cost"

VAL_SEASONS = (6,)
N_COMPONENTS = 250
ALPHAS = (1e3, 1e4, 1e5, 1e6)
PCA_SAMPLE = 40_000
LAYER = 19  # the video peak in runs/av_layer_sweep


def modality_at(crop: int, suffix: str = "") -> F.Modality:
    """The reference cache keeps its plain name; other crops get a _res<N><suffix> dir.

    `suffix="v2"` selects the caches extracted after the RoPE fix (crop_size override).
    """
    base = F.VIDEO
    sub = base.subdir if crop == 256 else f"{base.subdir}_res{crop}{suffix}"
    return F.Modality(base.name, sub, base.cached_hz, base.best_layer)


def design(store, runs, feats, scaler, pca):
    xs, ys = [], []
    for run in runs:
        f = feats.get(run.stimulus)
        if f is None:
            continue
        bold = trim_fmri(store.load(run))
        xs.append(align_features(pca.transform(scaler.transform(f)), bold.shape[0]))
        ys.append(bold)
    if not xs:
        return None, None
    return np.concatenate(xs), np.concatenate(ys)


def fit_projection(arrays: list[np.ndarray]):
    stacked = np.concatenate(arrays)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(stacked), size=min(PCA_SAMPLE, len(stacked)), replace=False)
    scaler = StandardScaler().fit(stacked[idx])
    pca = PCA(n_components=N_COMPONENTS, random_state=0).fit(scaler.transform(stacked[idx]))
    return scaler, pca


def best_ridge(x_tr, y_tr, x_va, y_va) -> np.ndarray:
    best = None
    for alpha in ALPHAS:
        r = pearson_per_parcel(Ridge(alpha=alpha).fit(x_tr, y_tr).predict(x_va), y_va)
        if best is None or r.mean() > best.mean():
            best = r
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="192,128")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--suffix", default="v2", help="res-dir generation (v2: after the RoPE fix)")
    args = ap.parse_args()
    crops = [int(c) for c in args.crops.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)

    ref = modality_at(256)
    stores = {s: BoldStore(DATA, s) for s in SUBJECTS}
    splits = {}
    for s, st in stores.items():
        vg = [g for g in st.groups() if any(f"-s{n:02d}e" in g for n in VAL_SEASONS)]
        splits[s] = split_by_episode(st.runs, val_groups=vg)

    train_stims = sorted({r.stimulus for s in SUBJECTS for r in splits[s][0]})
    val_stims = sorted({r.stimulus for s in SUBJECTS for r in splits[s][1]})

    # --- reference: fit on 256 train features, the head everything is scored against ---
    ref_train = {}
    for stim in train_stims:
        f = F.load_canonical(ref, stim, args.layer)
        if f is not None:
            ref_train[stim] = f
    if not ref_train:
        raise SystemExit(f"no 256px training features cached at {ref.path}")
    print(f"reference: {len(ref_train)} train stimuli at 256px, layer {args.layer}")
    scaler, pca = fit_projection(list(ref_train.values()))

    ceilings = {s: noise_ceiling(stores[s]) for s in SUBJECTS}
    # Every alpha is kept rather than one being chosen now: the best-regularised head for
    # 256 features is not necessarily the best for noisier 192 ones, and forcing a single
    # alpha would charge resolution for a regularisation mismatch it did not cause.
    heads: dict[str, dict[float, Ridge]] = {}
    for s in SUBJECTS:
        x, y = design(stores[s], splits[s][0], ref_train, scaler, pca)
        heads[s] = {alpha: Ridge(alpha=alpha).fit(x, y) for alpha in ALPHAS}
        del x, y
    del ref_train

    results: dict[str, dict] = {}
    for crop in [256, *crops]:
        mod = modality_at(crop, args.suffix)
        got = {}
        for stim in val_stims:
            f = F.load_canonical(mod, stim, args.layer)
            if f is not None:
                got[stim] = f
        if not got:
            print(f"\ncrop {crop}: nothing cached at {mod.path} -- skipped")
            continue
        print(f"\ncrop {crop}: {len(got)}/{len(val_stims)} val stimuli cached")

        frozen, refit = {}, {}
        for s in SUBJECTS:
            x, y = design(stores[s], splits[s][1], got, scaler, pca)
            if x is None:
                continue
            best = None
            for alpha in ALPHAS:
                r = pearson_per_parcel(heads[s][alpha].predict(x), y)
                if best is None or r.mean() > best.mean():
                    best = r
            frozen[s] = best

            # refit: own projection (all val features), own head, val runs halved by position
            sc2, pca2 = fit_projection(list(got.values()))
            runs = [r for r in splits[s][1] if r.stimulus in got]
            half = max(1, len(runs) // 2)
            a, b = runs[:half], runs[half:]
            xa, ya = design(stores[s], a, got, sc2, pca2)
            xb, yb = design(stores[s], b, got, sc2, pca2)
            if xa is not None and xb is not None:
                r1 = best_ridge(xa, ya, xb, yb)
                r2 = best_ridge(xb, yb, xa, ya)
                refit[s] = (r1 + r2) / 2

        def agg(d):
            if not d:
                return None
            return {
                "mean": float(np.mean([v.mean() for v in d.values()])),
                "per_subject": {s: float(v.mean()) for s, v in d.items()},
                "normalized_per_network": {
                    k: v[0]
                    for k, v in per_network(
                        np.nanmean(np.stack([ceilings[s].normalize(v) for s, v in d.items()]), 0)
                    ).items()
                },
            }

        results[str(crop)] = {"n_val": len(got), "frozen": agg(frozen), "refit": agg(refit)}
        f_m = results[str(crop)]["frozen"]
        r_m = results[str(crop)]["refit"]
        print(f"  frozen head : {f_m['mean']:.4f}" if f_m else "  frozen head : n/a")
        print(f"  refit head  : {r_m['mean']:.4f}" if r_m else "  refit head  : n/a")

    base = results.get("256", {}).get("frozen", {}).get("mean")
    if base:
        print(f"\n{'crop':>6}{'frozen':>10}{'retained':>10}{'refit':>10}{'recoverable':>13}")
        for crop in [256, *crops]:
            e = results.get(str(crop))
            if not e or not e["frozen"]:
                continue
            fz = e["frozen"]["mean"]
            rf = e["refit"]["mean"] if e["refit"] else float("nan")
            print(f"{crop:>6}{fz:>10.4f}{100 * fz / base:>9.1f}%{rf:>10.4f}{rf - fz:>+13.4f}")

    payload = {
        "layer": args.layer,
        "suffix": args.suffix,
        "note": "frozen = head fit on 256px train features; refit = head refit on the "
        "low-res features, val runs cut in half at the midpoint. The gap is recoverable damage.",
        "crops": results,
    }
    out_name = f"result{'_' + args.suffix if args.suffix else ''}_L{args.layer}.json"
    (OUT / out_name).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT / out_name}")


if __name__ == "__main__":
    main()
