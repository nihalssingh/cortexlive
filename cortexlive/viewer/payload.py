"""What the viewer page draws, built from captures and the reference data."""

from __future__ import annotations

import base64
import functools
from pathlib import Path

import numpy as np

from .. import capture, reference
from ..align import EXCLUDED_END, EXCLUDED_START, trim_fmri
from ..data import TR_SECONDS

ROOT = Path(__file__).resolve().parent.parent.parent
ATLAS = ROOT / "atlas"
SCALE = 3.0  # int8 127 = SCALE standard deviations
TRIM = (EXCLUDED_START, EXCLUDED_END)  # edge volumes left out of training and scoring


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def scored(a: np.ndarray) -> np.ndarray:
    """Rows to take statistics over: finite and past the start trim (those volumes are never
    trained on or scored), and before the end trim too when that leaves enough rows."""
    ok = np.isfinite(a[:, 0])
    ok[: TRIM[0]] = False
    inner = ok.copy()
    inner[len(a) - TRIM[1]:] = False
    return inner if inner.sum() > 2 else ok


def zq(a: np.ndarray, base: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """(n, 1000) with NaN rows -> int8 z-scores per parcel; NaN rows stay 0.

    The statistics come from the scored rows. With fewer than 3 of them they come from
    `base`, each parcel's (mean, sd) over Friends season 6, as on the live page.
    """
    ok, rows = np.isfinite(a[:, 0]), scored(a)
    out = np.zeros(a.shape, np.int8)
    if rows.sum() > 2:
        mean, sd = a[rows].mean(0), a[rows].std(0)
    elif base is not None:
        mean, sd = base
    else:
        return out
    z = (a[ok] - mean) / np.where(sd < 1e-8, 1.0, sd)
    out[ok] = np.clip(np.round(z / SCALE * 127), -127, 127).astype(np.int8)
    return out


def f16(a: np.ndarray) -> str:
    return b64(np.asarray(a, np.float32).astype(np.float16))


@functools.cache
def atlas() -> dict:
    views = np.load(ATLAS / "views.npz", allow_pickle=False)
    parc = np.load(ATLAS / "schaefer1000.npz", allow_pickle=False)
    names = [str(x) for x in views["view_names"]]
    return {
        "res": int(views["res"][0]),
        "view_names": names,
        "parcel": {v: b64(views[f"{v}_parcel"].astype(np.int16)) for v in names},
        "shade": {v: b64((np.clip(views[f"{v}_shade"], 0, 1) * 255).astype(np.uint8))
                  for v in names},
        "network_index": b64(parc["network_index"].astype(np.int16)),
        "network_names": [str(x) for x in parc["network_display"]],
    }


def capture_payload(path: Path) -> dict:
    meta, arrays = capture.load(path)
    subjects = meta["model"]["subjects"]
    samples = arrays["samples"].astype(int)
    n_vol = max(meta["run"]["expected_volumes"] or 0, int(samples.max()) + 1 if len(samples) else 0)
    covered = np.zeros(n_vol, np.uint8)
    covered[samples] = 1
    ref = reference.load()

    known = meta.get("known")
    bold = ctrl = parcels = None
    if known:
        try:
            bold = reference.measured(known)
            trimmed = {s: len(trim_fmri(b)) for s, b in bold.items()}
            ctrl = reference.control_predictions(known, trimmed)
        except Exception:
            bold = ctrl = None
        if (path / "scores.npz").exists():
            with np.load(path / "scores.npz") as z:
                parcels = {k: z[k] for k in z.files}

    streams = {}
    for s in subjects:
        full = np.full((n_vol, 1000), np.nan, np.float32)
        if len(samples):
            full[samples] = arrays[s]
        base = (ref["mean"][s], ref["sd"][s]) if ref and s in ref["sd"] else None
        entry = {"pred": b64(zq(full, base)), "maps": {}}
        rows = scored(full)
        if rows.sum() > 2:
            sd_run, mean_run = full[rows].std(0), full[rows].mean(0)
            if ref and s in ref["sd"]:
                sd_ref = np.where(ref["sd"][s] < 1e-8, np.nan, ref["sd"][s])
                entry["maps"]["strength"] = f16(sd_run / sd_ref)
                entry["maps"]["shift"] = f16((mean_run - ref["mean"][s]) / sd_ref)
        if bold and s in bold:
            m = np.full((n_vol, 1000), np.nan, np.float32)
            k = min(n_vol, len(bold[s]) - EXCLUDED_START)  # the scored, trimmed timeline
            m[EXCLUDED_START:k] = bold[s][EXCLUDED_START:k]
            entry["measured"] = b64(zq(m))
        if ctrl and s in ctrl:
            c = np.full((n_vol, 1000), np.nan, np.float32)
            rows = ctrl[s][: max(0, n_vol - EXCLUDED_START)]
            c[EXCLUDED_START:EXCLUDED_START + len(rows)] = rows
            entry["control"] = b64(zq(c))
        if parcels and f"{s}_r" in parcels:
            entry["maps"]["accuracy"] = f16(parcels[f"{s}_r"])
            if np.isfinite(parcels[f"{s}_control_r"]).any():
                entry["maps"]["gain"] = f16(parcels[f"{s}_r"] - parcels[f"{s}_control_r"])
        streams[s] = entry

    masks = {s: b64((ref["mask"][s] if ref and s in ref["mask"] else np.ones(1000, bool))
                    .astype(np.uint8)) for s in subjects}
    shown = capture.finite(meta)
    # where the video sits on this machine stays out of the page (and so out of exports);
    # the server finds the media itself
    shown["input"] = {k: v for k, v in shown["input"].items() if k != "path"}
    return {
        "id": path.name,
        "meta": shown,
        "n_vol": n_vol,
        "tr": TR_SECONDS,
        "scale": SCALE,
        "trim": list(TRIM),
        "covered": b64(covered),
        "masks": masks,
        "streams": streams,
        "has_media": capture.media_path(path, meta) is not None,
    }


def capture_entry(path: Path, cid: str | None = None) -> dict | None:
    try:
        meta = capture.load_meta(path)
    except Exception:
        return None
    sc = meta.get("scores") or {}
    return {
        "id": cid or path.name,
        "name": meta["input"]["name"],
        "created": meta["created"],
        "duration_s": meta["input"]["duration_s"],
        "volumes": meta["run"]["volumes"],
        "pace": meta["run"]["pace"],
        "stopped": meta["run"].get("stopped", False),
        "known": (meta.get("known") or {}).get("stimulus"),
        "r": (sc.get("mean") or {}).get("r"),
    }


def capture_list(root: Path, extra: dict[str, Path] | None = None) -> list[dict]:
    entries = [capture_entry(p, cid) for cid, p in (extra or {}).items()]
    entries += [capture_entry(p) for p in capture.find(root)]
    return [e for e in entries if e]
