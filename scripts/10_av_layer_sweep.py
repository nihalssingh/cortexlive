"""Where does brain-predictive signal peak inside the video and audio backbones?

Text peaked at layer 8 of 16 and got *worse* above it (runs/layer_sweep_all), so the live
text stream runs only half of Llama. If video and audio also peak below their top layer,
the layers above the peak need not run live. If they peak at the top, the real-time budget
has to come from elsewhere, such as a lower video resolution (scripts/15).

Structured layer-outer, subject-inner: each stimulus/layer array is read and pooled once
and then scored for all four subjects. Reading it per subject instead would multiply the
I/O by four for no benefit, since `load_canonical` pools onto the stimulus's own grid.

    uv run --extra train python scripts/10_av_layer_sweep.py --modality video
    uv run --extra train python scripts/10_av_layer_sweep.py --modality audio
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from cortexlive import features as F
from cortexlive.align import align_features, trim_fmri
from cortexlive.data import SUBJECTS, BoldStore, split_by_episode
from cortexlive.score import pearson_per_parcel

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "algonauts"
OUT = ROOT / "runs" / "av_layer_sweep"

VAL_SEASONS = (6,)
N_COMPONENTS = 250
ALPHAS = (1e3, 1e4, 1e5, 1e6)
PCA_SAMPLE = 40_000


def subject_split(store: BoldStore):
    vg = [g for g in store.groups() if any(f"-s{s:02d}e" in g for s in VAL_SEASONS)]
    return split_by_episode(store.runs, val_groups=vg)


def score(store: BoldStore, train_runs, val_runs, feats: dict[str, np.ndarray]) -> float | None:
    """Fit ridge on train runs, score on val runs, using pre-pooled features."""
    tr = [(r, f) for r in train_runs if (f := feats.get(r.stimulus)) is not None]
    va = [(r, f) for r in val_runs if (f := feats.get(r.stimulus)) is not None]
    if not tr or not va:
        return None

    stacked = np.concatenate([f for _, f in tr])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(stacked), size=min(PCA_SAMPLE, len(stacked)), replace=False)
    scaler = StandardScaler().fit(stacked[idx])
    pca = PCA(n_components=N_COMPONENTS, random_state=0).fit(scaler.transform(stacked[idx]))
    del stacked

    def build(pairs):
        xs, ys = [], []
        for run, f in pairs:
            bold = trim_fmri(store.load(run))
            xs.append(align_features(pca.transform(scaler.transform(f)), bold.shape[0]))
            ys.append(bold)
        return np.concatenate(xs), np.concatenate(ys)

    x_tr, y_tr = build(tr)
    x_va, y_va = build(va)

    best = -np.inf
    for alpha in ALPHAS:
        r = pearson_per_parcel(Ridge(alpha=alpha).fit(x_tr, y_tr).predict(x_va), y_va)
        best = max(best, float(r.mean()))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=("video", "audio"), required=True)
    ap.add_argument("--layers", default="", help="e.g. '0,4,8,12' -- default is every layer")
    args = ap.parse_args()

    modality = F.MODALITIES[args.modality]
    n_layers = F.n_layers(modality)
    if n_layers is None:
        raise SystemExit(f"nothing cached in {modality.path} -- run extraction first")
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(range(n_layers))
    cached = F.available(modality)
    OUT.mkdir(parents=True, exist_ok=True)

    stores = {s: BoldStore(DATA, s) for s in SUBJECTS}
    splits = {s: subject_split(st) for s, st in stores.items()}
    # Stimuli any subject actually needs AND that are cached -- the usable set.
    needed = {r.stimulus for st in stores.values() for r in st.runs}
    usable = sorted(needed & cached)
    print(
        f"{modality.name}: {n_layers} layers, {len(cached)} cached, "
        f"{len(usable)} of {len(needed)} labelled stimuli usable\n"
    )
    if not usable:
        raise SystemExit("no overlap between cached features and fMRI runs")

    table: dict[str, dict[int, float]] = {s: {} for s in SUBJECTS}
    for layer in layers:
        t0 = time.time()
        feats = {}
        for stim in usable:
            f = F.load_canonical(modality, stim, layer)
            if f is not None:
                feats[stim] = f
        row = []
        for s in SUBJECTS:
            train_runs, val_runs = splits[s]
            v = score(stores[s], train_runs, val_runs, feats)
            if v is not None:
                table[s][layer] = v
            row.append(v)
        del feats
        vals = [v for v in row if v is not None]
        print(
            f"layer {layer:>2}  "
            + "  ".join("  n/a " if v is None else f"{v:.4f}" for v in row)
            + f"   mean {np.mean(vals):.4f}   [{time.time() - t0:.0f}s]"
        )

    print(f"\n{'layer':>6}" + "".join(f"{s:>10}" for s in SUBJECTS) + f"{'mean':>10}")
    means = {}
    for layer in layers:
        vals = [table[s].get(layer) for s in SUBJECTS]
        present = [v for v in vals if v is not None]
        if not present:
            continue
        means[layer] = float(np.mean(present))
        print(
            f"{layer:>6}"
            + "".join("       n/a" if v is None else f"{v:>10.4f}" for v in vals)
            + f"{means[layer]:>10.4f}"
        )

    peaks = {s: int(max(t, key=t.get)) for s, t in table.items() if t}
    mean_peak = int(max(means, key=means.get))
    cost = {s: table[s][peaks[s]] - table[s].get(mean_peak, np.nan) for s in peaks}
    print(f"\nper-subject peaks : {peaks}")
    print(f"peak of the mean  : layer {mean_peak} of {n_layers - 1}")
    print(
        f"cost of using layer {mean_peak} for everyone: "
        + ", ".join(f"{s} {c:+.4f}" for s, c in cost.items())
    )
    deepest = max(means)
    top, last = means[mean_peak], means[deepest]
    print(
        f"peak (layer {mean_peak}) vs deepest swept (layer {deepest}): "
        f"{top:.4f} vs {last:.4f}  ({100 * (top / last - 1):+.1f}%)"
    )

    (OUT / f"{modality.name}.json").write_text(
        json.dumps(
            {
                "modality": modality.name,
                "model": modality.subdir,
                "n_layers": n_layers,
                "n_stimuli": len(usable),
                "per_subject": {s: {str(k): v for k, v in t.items()} for s, t in table.items()},
                "mean_over_subjects": {str(k): v for k, v in means.items()},
                "per_subject_peak": peaks,
                "peak_of_mean": mean_peak,
                "cost_of_global_depth": cost,
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT / f'{modality.name}.json'}")


if __name__ == "__main__":
    main()
