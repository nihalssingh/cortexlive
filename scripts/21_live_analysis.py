"""Records for the live configuration: accuracy, ridge strength, and live-run parity.

**live-config** -- the shipped head was fit on 256px video features; the live path runs
V-JEPA2 at 192px. Scores the shipped head, unchanged, on all of Friends season 6 with
each video configuration, in 12_trimodal's protocol (per-subject Pearson over the
concatenated season-6 runs, mean over parcels; ceiling-normalized like that record).

**alpha** -- re-solves the shipped ridge in closed form (same scaler+PCA, windows and
split) over a wide alpha grid, to show where the selected alpha sits on the curve.

**capture** -- one `cortexlive run` capture of a CNeuroMod stimulus, against the
offline pipeline (cached features, same head) on the same volumes: per-volume
prediction correlation (split by whether the window's newest audio was still a
provisional partial chunk when the volume was emitted), accuracy of each, and the
run's lag and compute trends.

    uv run --extra live python scripts/21_live_analysis.py --mode live-config
    uv run --extra live python scripts/21_live_analysis.py --mode alpha
    uv run --extra live python scripts/21_live_analysis.py --mode capture \
        --capture runs/live_demo/friends_s06e01a

The _fast and _no-transcript records come from the same command on those captures, whose
predictions are not shipped: make them again first (README, Reproduce).
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from cortexlive import capture, reference, stream  # noqa: E402
from cortexlive import features as F  # noqa: E402
from cortexlive.align import EXCLUDED_START, trim_fmri  # noqa: E402
from cortexlive.data import SUBJECTS, TR_SECONDS, BoldStore, split_by_episode  # noqa: E402
from cortexlive.score import noise_ceiling, pearson_per_parcel  # noqa: E402

gate = importlib.import_module("18_stream_gate")  # module name starts with a digit
tm = importlib.import_module("12_trimodal")

DATA = ROOT / "data" / "algonauts"
OUT = ROOT / "runs" / "live_analysis"
VAL_SEASONS = (6,)


def write(name: str, record: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(record, indent=2))
    print(f"wrote {OUT / name}")


def splits(subject: str):
    store = BoldStore(DATA, subject)
    vg = [g for g in store.groups() if any(f"-s{sn:02d}e" in g for sn in VAL_SEASONS)]
    train, val = split_by_episode(store.runs, val_groups=vg)
    return store, train, val


def design(store: BoldStore, runs, rows: dict[str, np.ndarray]):
    """Windowed design matrix + BOLD over runs, exactly 12_trimodal's indexing."""
    xs, ys = [], []
    for run in runs:
        z = rows.get(run.stimulus)
        if z is None:
            continue
        bold = trim_fmri(store.load(run)).astype(np.float32)
        idx = tm.window_index(bold.shape[0], z.shape[0])
        xs.append(z[idx].reshape(len(idx), -1))
        ys.append(bold)
    return np.concatenate(xs), np.concatenate(ys)


def project_all(head: stream.HeadArtifact, stims, video_mod: F.Modality) -> dict:
    """stimulus -> projected (n_tr, 750) rows, cut to the shortest modality like Corpus."""
    out = {}
    for stim in sorted(stims):
        try:
            out[stim] = gate.cached_rows(head, stim, video_mod)
        except SystemExit:
            continue  # not cached in this config
    return out


# ------------------------------------------------------------------- live-config


def mode_live_config() -> None:
    head = stream.HeadArtifact()
    configs = {"256px (training config)": F.VIDEO, "192px-v2 (live config)": gate.video_modality()}
    record: dict = {"protocol": "12_trimodal: per-subject Pearson over concatenated season-6 "
                    "val runs, mean over 1000 parcels; normalized = mean r/ceiling (ceiling = "
                    "sqrt of test-retest reliability) over parcels with significant "
                    "reliability and ceiling > 0.1 (score.NoiseCeiling.normalize)",
                    "head": "models/final_head.npz, unchanged (no refit)"}
    for label, vmod in configs.items():
        per_subject, normalized, n_eps = {}, {}, {}
        for s in SUBJECTS:
            store, _, val = splits(s)
            rows = project_all(head, {r.stimulus for r in val}, vmod)
            x, y = design(store, val, rows)
            pred = x @ head.arrays[f"{s}_coef"].T + head.arrays[f"{s}_intercept"]
            r = pearson_per_parcel(pred, y)
            per_subject[s] = float(r.mean())
            normalized[s] = float(np.nanmean(noise_ceiling(store).normalize(r)))
            n_eps[s] = len({run.stimulus for run in val} & set(rows))
            print(f"  {label:<24} {s}: r {per_subject[s]:.6f}  ({n_eps[s]} val episodes)")
        record[label] = {
            "mean": float(np.mean(list(per_subject.values()))),
            "normalized_mean": float(np.mean(list(normalized.values()))),
            "per_subject": per_subject,
            "val_episodes": n_eps,
        }
        print(f"  {label:<24} mean r {record[label]['mean']:.6f}  "
              f"normalized {record[label]['normalized_mean']:.4f}")
    a, b = (record[k]["mean"] for k in configs)
    record["relative_change"] = b / a - 1
    print(f"192px vs 256px: {record['relative_change']:+.2%}")
    write("live_config.json", record)


# ------------------------------------------------------------------------ alpha

ALPHA_GRID = (1e3, 1e4, 1e5, 2e5, 3e5, 5e5, 7e5, 1e6, 1.5e6, 2e6, 4e6, 1e7, 3e7, 1e8)


def mode_alpha() -> None:
    """Closed-form ridge over a wide alpha grid; the shipped 1e6 must reproduce meta.json."""
    head = stream.HeadArtifact()
    meta = json.loads(stream.META_PATH.read_text())
    record: dict = {"alphas": list(ALPHA_GRID), "grid_shipped": list(tm.ALPHAS),
                    "per_subject": {}}
    stims = set()
    for s in SUBJECTS:
        _, train, val = splits(s)
        stims |= {r.stimulus for r in train} | {r.stimulus for r in val}
    rows = project_all(head, stims, F.VIDEO)
    print(f"projected {len(rows)} stimuli")
    for s in SUBJECTS:
        store, train, val = splits(s)
        x_tr, y_tr = design(store, train, rows)
        x_va, y_va = design(store, val, rows)
        # Ridge with intercept == ridge on centered data; solve every alpha from one
        # eigendecomposition of the Gram matrix (accumulated in float64, in chunks).
        mx, my = x_tr.mean(0, dtype=np.float64), y_tr.mean(0, dtype=np.float64)
        d = x_tr.shape[1]
        gram = np.zeros((d, d))
        xty = np.zeros((d, y_tr.shape[1]))
        for i in range(0, len(x_tr), 20_000):
            xc = x_tr[i : i + 20_000] - mx
            gram += xc.T @ xc
            xty += xc.T @ (y_tr[i : i + 20_000] - my)
        lam, vecs = np.linalg.eigh(gram)
        proj = vecs.T @ xty
        xv = (x_va - mx).astype(np.float64) @ vecs
        val_r = {}
        for alpha in ALPHA_GRID:
            pred = xv @ (proj / (lam + alpha)[:, None]) + my
            val_r[f"{alpha:g}"] = float(pearson_per_parcel(pred, y_va).mean())
        best = max(val_r, key=val_r.get)
        shipped = meta["per_subject"][s]
        record["per_subject"][s] = {
            "val_r": val_r,
            "best_alpha": float(best),
            "best_val_r": val_r[best],
            "shipped_alpha": shipped["alpha"],
            "shipped_val_r_meta": shipped["val_r"],
            "reproduces_meta": abs(val_r[f"{shipped['alpha']:g}"] - shipped["val_r"]) < 1e-5,
            "monotone_decline_above_shipped": all(
                val_r[f"{a:g}"] >= val_r[f"{b:g}"]
                for a, b in zip(ALPHA_GRID, ALPHA_GRID[1:], strict=False)
                if a >= shipped["alpha"]
            ),
        }
        print(f"  {s}: best alpha {float(best):g} r {val_r[best]:.6f}  "
              f"(1e6: {val_r['1e+06']:.6f}, meta {shipped['val_r']:.6f})")
        del x_tr, y_tr, x_va, y_va, gram
    best_mean = float(np.mean([v["best_val_r"] for v in record["per_subject"].values()]))
    ship_mean = float(np.mean([v["val_r"]["1e+06"] for v in record["per_subject"].values()]))
    record["mean_val_r_shipped"] = ship_mean
    record["mean_val_r_per_subject_best"] = best_mean
    record["gain_if_alpha_retuned"] = best_mean - ship_mean
    record["note"] = ("per-subject best is itself selected on season-6 val, so the gain is "
                      "an upper bound; 256px features, the training config")
    print(f"mean val r: shipped {ship_mean:.6f}, per-subject best {best_mean:.6f} "
          f"(+{best_mean - ship_mean:.4f})")
    write("alpha.json", record)


# ---------------------------------------------------------------------- capture


def offline_rows(head, stim: str, zero_text: bool = False) -> np.ndarray:
    """Cached feature rows in the live configuration (192px video)."""
    z = gate.cached_rows(head, stim, gate.video_modality())
    if zero_text:  # what the live path sees with no transcript: zero raw text rows
        k = head.modalities.index("text")
        dim = head.arrays["text_scaler_mean"].shape[0]
        zero = head.project("text", np.zeros((1, dim), np.float32))[0]
        width = z.shape[1] // len(head.modalities)
        z = z.copy()
        z[:, k * width:(k + 1) * width] = zero
    return z


def trend(x: np.ndarray) -> dict:
    return {"mean": float(x.mean()), "p95": float(np.percentile(x, 95)),
            "max": float(x.max()),
            "quarter_means": [float(c.mean()) for c in np.array_split(x, 4)],
            "linear_slope_per_volume": float(np.polyfit(np.arange(len(x)), x, 1)[0])}


def mode_capture(path: pathlib.Path) -> None:
    from cortexlive.align import HRF_DELAY

    head = stream.HeadArtifact()
    meta, rec = capture.load(path)
    known = meta.get("known")
    if not known:
        raise SystemExit(f"{path}: not a CNeuroMod stimulus, nothing to compare against")
    stim, run = known["stimulus"], meta["run"]
    raw = reference.measured(known)
    bold = {s: trim_fmri(b) for s, b in raw.items()}
    n = min(len(b) for b in bold.values())
    with_text = run["transcript"] != "none"
    off = gate.predict_episode(head, offline_rows(head, stim, zero_text=not with_text), n)
    masks = reference.load()["mask"]

    samples = rec["samples"].astype(int)
    rows = samples - EXCLUDED_START
    keep = (rows >= 0) & (rows < n)
    i = rows[keep]
    # A window still had provisional audio if, when the volume was emitted, the audio
    # it needs (through stimulus TR s - HRF_DELAY) lay past the last finalized 30 s chunk.
    fed_s = np.minimum(rec["wall_s"], meta["input"]["duration_s"])
    finalized_s = np.floor(fed_s / stream.AUDIO_CHUNK_S) * stream.AUDIO_CHUNK_S
    needs_s = (samples - HRF_DELAY + 1) * TR_SECONDS
    provisional = (needs_s > finalized_s)[keep] if run["pace"] == "realtime" else None

    per_subject = {}
    for s in sorted(bold):
        live, o, y = rec[s][keep], off[s][i], bold[s][i]
        pc = np.array([np.corrcoef(live[k], o[k])[0, 1] for k in range(len(i))])
        r_live, r_off = pearson_per_parcel(live, y), pearson_per_parcel(o, y)
        m = masks[s]
        d = {"volumes_scored": int(len(i)),
             "pred_corr_mean": float(pc.mean()), "pred_corr_min": float(pc.min()),
             "r_live": float(r_live.mean()), "r_offline": float(r_off.mean()),
             "r_live_reliable": float(r_live[m].mean()),
             "r_offline_reliable": float(r_off[m].mean())}
        if provisional is not None and provisional.any() and (~provisional).any():
            d["pred_corr_mean_provisional_audio"] = float(pc[provisional].mean())
            d["pred_corr_mean_final_audio"] = float(pc[~provisional].mean())
        per_subject[s] = d
        print(f"  {s}: pred-corr {d['pred_corr_mean']:.4f}  r live {d['r_live']:.4f} "
              f"offline {d['r_offline']:.4f}  reliable {d['r_live_reliable']:.4f} / "
              f"{d['r_offline_reliable']:.4f}")

    def avg(k):
        return float(np.mean([d[k] for d in per_subject.values()]))

    lag, tick = rec["lag_s"], rec["tick_ms"]
    record = {
        "capture": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "stimulus": stim,
        "pace": run["pace"],
        "transcript": run["transcript"],
        "device": run["device"],
        "duration_s": meta["input"]["duration_s"],
        "volumes": run["volumes"],
        "expected_volumes": run["expected_volumes"],
        "bold_volumes": {s: int(len(b)) for s, b in raw.items()},
        "volumes_scored": int(keep.sum()),
        "volumes_with_provisional_audio": int(provisional.sum()) if provisional is not None
        else None,
        "offline_reference": "cached features, same head, 192px video, "
        + ("CNeuroMod transcript" if with_text else "zero text rows (no transcript)"),
        "lag_s": trend(lag) if run["pace"] == "realtime" else None,
        "volumes_later_than_1_tr": int(np.sum(lag > TR_SECONDS)) if run["pace"] == "realtime"
        else None,
        "tick_ms": trend(tick),
        "per_subject": per_subject,
        "mean": {k: avg(k) for k in ("pred_corr_mean", "r_live", "r_offline",
                                     "r_live_reliable", "r_offline_reliable")},
    }
    print(json.dumps({k: v for k, v in record.items() if k != "per_subject"}, indent=1))
    write(f"capture_{path.name}.json", record)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live-config", "alpha", "capture"], required=True)
    ap.add_argument("--capture", type=pathlib.Path, help="capture: a capture directory")
    args = ap.parse_args()
    if args.mode == "live-config":
        mode_live_config()
    elif args.mode == "alpha":
        mode_alpha()
    else:
        if args.capture is None:
            raise SystemExit("--mode capture needs --capture")
        mode_capture(args.capture.resolve())


if __name__ == "__main__":
    main()
