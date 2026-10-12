"""Captures: what one run of the model produced, saved as a directory.

    <capture>/
      capture.json       input, run settings, timing, and scores when the input is a
                         CNeuroMod stimulus with recorded fMRI
      predictions.npz    samples (volume index), lag_s, tick_ms, wall_s, and one
                         (n_volumes, 1000) float32 array per subject
      scores.npz         per-parcel accuracy (known stimuli only)
      run.log            console output of the run
      input.<ext>        the video itself, when it was uploaded through the viewer

Volume `s` is the predicted BOLD at time s * TR of the input.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import re
import subprocess
from pathlib import Path

import numpy as np

from .data import TR_SECONDS
from .run import RunResult

FORMAT = "cortexlive-capture/1"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = ROOT / "captures"


def new_dir(root: Path, input_name: str) -> Path:
    # a capture id must start with a letter or digit and hold no "..", or the viewer refuses it
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(input_name).stem)
    stem = re.sub(r"\.{2,}", ".", stem).strip("-._")[:60] or "input"
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = root / f"{stem}-{stamp}"
    n = 1
    while path.exists():
        n += 1
        path = root / f"{stem}-{stamp}-{n}"
    path.mkdir(parents=True)
    return path


def machine() -> dict:
    """What the run ran on: the timing in a capture means little without it."""
    info = {"system": f"{platform.system()} {platform.release()}", "arch": platform.machine()}
    if platform.system() == "Darwin":
        def sysctl(key: str) -> str:
            out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
            return out.stdout.strip()

        info.update(chip=sysctl("machdep.cpu.brand_string"), model=sysctl("hw.model"),
                    memory_gb=round(int(sysctl("hw.memsize") or 0) / 2**30))
    return info


def _stats(x: np.ndarray) -> dict | None:
    if not len(x):
        return None
    return {"mean": float(x.mean()), "p95": float(np.percentile(x, 95)), "max": float(x.max()),
            "quarter_means": [float(c.mean()) for c in np.array_split(x, 4) if len(c)]}


def finite(obj):
    """JSON has no NaN: non-finite numbers become null."""
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [finite(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def save(path: Path, result: RunResult, *, input_path: Path | None, input_name: str,
         device: str, transcript: str, head) -> dict:
    """Write predictions.npz and capture.json for one run; returns the metadata."""
    vols = result.volumes
    samples = np.array([v.s for v in vols], dtype=np.int32)
    arrays = {
        "samples": samples,
        "lag_s": np.array([v.lag_s for v in vols], dtype=np.float32),
        "tick_ms": np.array([v.tick_ms for v in vols], dtype=np.float32),
        "wall_s": np.array([v.wall_s for v in vols], dtype=np.float32),
    }
    for subj in head.subjects:
        arrays[subj] = (np.stack([v.preds[subj] for v in vols]).astype(np.float32) if vols
                        else np.zeros((0, 1000), np.float32))
    np.savez_compressed(path / "predictions.npz", **arrays)

    media_s = result.media_s
    expected = int((media_s - 1e-9) / TR_SECONDS) + 1 if media_s else len(vols)
    if input_path is not None and input_path.is_relative_to(ROOT):
        input_path = input_path.relative_to(ROOT)  # portable across checkouts
    meta = {
        "format": FORMAT,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "input": {
            "name": input_name,
            "path": str(input_path) if input_path else None,
            "duration_s": media_s,
        },
        "run": {
            "pace": result.pace,
            "device": device,
            "machine": machine(),
            "transcript": transcript,
            "volumes": len(vols),
            "expected_volumes": expected,
            "stopped": result.stopped,
            "wall_s": result.wall_s,
            "realtime_factor": (media_s / result.wall_s) if media_s and result.wall_s else None,
            "first_volume_s": float(vols[0].wall_s) if vols else None,
            "lag_s": _stats(arrays["lag_s"]) if result.pace == "realtime" else None,
            "tick_ms": _stats(arrays["tick_ms"]),
            "notes": result.notes,
        },
        "model": {
            "subjects": list(head.subjects),
            "layers": head.layers,
            "window_tr": head.window,
            "tr_s": TR_SECONDS,
        },
        "known": None,
        "scores": None,
    }
    write_meta(path, meta)
    return meta


def write_meta(path: Path, meta: dict) -> None:
    (path / "capture.json").write_text(json.dumps(finite(meta), indent=2, allow_nan=False))


def load_meta(path: Path) -> dict:
    return json.loads((path / "capture.json").read_text())


def load(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    meta = load_meta(path)
    with np.load(path / "predictions.npz") as z:
        arrays = {k: z[k] for k in z.files}
    return meta, arrays


def media_path(path: Path, meta: dict) -> Path | None:
    """The capture's video: an uploaded copy inside it, else the original path. The viewer
    serves it only if it resolves to a video file."""
    for p in sorted(path.glob("input.*")):
        return p
    src = meta.get("input", {}).get("path")
    if not src:
        return None
    src = Path(src) if Path(src).is_absolute() else ROOT / src
    return src if src.exists() else None


def find(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted((p for p in root.iterdir() if (p / "capture.json").exists()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
