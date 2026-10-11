"""Verify the live pipeline against the training pipeline, then pace it.

**head-parity** -- no models, no media. Feeds cached features through HeadArtifact's
project/predict and scores against the recorded BOLD for one episode: if the window
layout, modality order or PCA plumbing in stream.py drifted from training, this number
collapses. Also scores the episode with 192px video features, the live configuration.

**extract-parity** -- loads the three truncated models, runs the live extractors over
the first N seconds of the episode (unpaced) and correlates every TR row against the
cache: ffmpeg piping, clip boundaries, KV-cache text and TR pooling must reproduce what
the head was trained on. Text should be ~1.0 (causal LM, exact); video ~1.0 against the
192px cache; audio ~1.0 too, since 240 s is eight whole 30 s chunks and none is
provisional (21_live_analysis.py --mode capture measures provisional audio).

**attrib** -- feeds the whole episode unpaced and predicts from the extractors' final
rows, which separates the cost of the live extraction path from the cost of emitting
on time with provisional audio.

**gate** -- paces the episode at 1x wall clock and predicts every volume, reporting
per-volume compute, emission lag, and compute in the first vs second half. Late volumes
are emitted as soon as their features exist, so lag, not a miss count, measures whether
the pipeline keeps up. This loop is older than cortexlive/run.py's, which `cortexlive run`
and the viewer use: it re-forwards the partial audio chunk for every volume, so its lag
(runs/stream_gate/gate.json) is higher than a capture's (runs/live_demo/).

    uv run --extra live python scripts/18_stream_gate.py --mode head-parity
    uv run --extra live python scripts/18_stream_gate.py --mode extract-parity --seconds 240
    uv run --extra live python scripts/18_stream_gate.py --mode extract-parity --seconds 240 \
        --open-ended
    uv run --extra live python scripts/18_stream_gate.py --mode attrib
    uv run --extra live python scripts/18_stream_gate.py --mode gate --record
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from cortexlive import features as F  # noqa: E402
from cortexlive import stream  # noqa: E402
from cortexlive.align import EXCLUDED_START, HRF_DELAY, STIMULUS_WINDOW, trim_fmri  # noqa: E402
from cortexlive.data import SUBJECTS, TR_SECONDS, BoldStore  # noqa: E402
from cortexlive.score import pearson_per_parcel  # noqa: E402
from cortexlive.stimulus import transcript_path  # noqa: E402

DATA = ROOT / "data" / "algonauts"
OUT = ROOT / "runs" / "stream_gate"
EPISODE = "s06e01a"


def movie_path(stim: str) -> pathlib.Path:
    season = int(stim[1:3])
    return DATA / "stimuli/movies/friends" / f"s{season}" / f"friends_{stim}.mkv"


def video_modality(suffix: str = "v2", crop: int = 192) -> F.Modality:
    sub = f"{F.VIDEO.subdir}_res{crop}{suffix}"
    return F.Modality("video", sub, F.VIDEO.cached_hz, F.VIDEO.best_layer)


def cached_rows(head: stream.HeadArtifact, stim: str, video_mod: F.Modality) -> np.ndarray:
    """(n_tr, n_mod * n_components) projected feature rows from the cache."""
    mods = {"text": F.TEXT, "audio": F.AUDIO, "video": video_mod}
    parts, n = {}, None
    for name in head.modalities:
        feats = F.load_canonical(mods[name], stim, head.layers[name])
        if feats is None:
            raise SystemExit(f"{name}: {stim} not cached at {mods[name].path}")
        parts[name] = head.project(name, feats)
        n = len(parts[name]) if n is None else min(n, len(parts[name]))
    return np.concatenate([parts[name][:n] for name in head.modalities], axis=1)


def predict_episode(head: stream.HeadArtifact, z: np.ndarray, n_trimmed: int) -> dict:
    """Vectorized offline predictions for every trimmed BOLD sample."""
    ends = np.arange(n_trimmed) + EXCLUDED_START - HRF_DELAY
    idx = np.clip(ends[:, None] - np.arange(STIMULUS_WINDOW - 1, -1, -1)[None, :], 0, len(z) - 1)
    x = z[idx].reshape(n_trimmed, -1)
    return {
        s: x @ head.arrays[f"{s}_coef"].T + head.arrays[f"{s}_intercept"]
        for s in head.subjects
    }


def episode_bold(stim: str) -> dict[str, np.ndarray]:
    out = {}
    for s in SUBJECTS:
        store = BoldStore(DATA, s)
        runs = [r for r in store.runs if r.stimulus == stim]
        if runs:
            out[s] = trim_fmri(store.load(runs[0]))
    return out


def mode_head_parity(stim: str) -> None:
    head = stream.HeadArtifact()
    bold = episode_bold(stim)
    print(f"episode {stim}, subjects {sorted(bold)}")
    record: dict[str, dict] = {"episode": stim}
    configs = (("256px (training config)", F.VIDEO), ("192px-v2 (live config)", video_modality()))
    for label, vmod in configs:
        z = cached_rows(head, stim, vmod)
        n = min(min(b.shape[0] for b in bold.values()), len(z) - EXCLUDED_START)
        preds = predict_episode(head, z, n)
        rs = {s: float(pearson_per_parcel(preds[s][: bold[s].shape[0]], bold[s][:n]).mean())
              for s in bold}
        mean = float(np.mean(list(rs.values())))
        print(f"  {label:<24} mean r {mean:.4f}   " +
              " ".join(f"{s.split('-')[1]}:{v:.4f}" for s, v in rs.items()))
        record[label] = {"mean": mean, "per_subject": rs}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "head_parity.json").write_text(json.dumps(record, indent=2))


def feed_media(enc: stream.LiveEncoder, stim: str, seconds: float, words: list[list[str]],
               anchor: bool = True) -> None:
    """Push N seconds of the real media file through the live extractors, unpaced."""
    path = movie_path(stim)
    n_frames = int(seconds * stream.VIDEO_FPS)
    # Anchor to the FULL file, not the fed prefix: the cache's pooling grid hangs off
    # the whole episode's length, so a prefix-anchored grid samples different instants
    # and audio parity collapses.
    full = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "s16le",
                           "-ac", "1", "-ar", str(stream.AUDIO_SR), "-"],
                          capture_output=True, check=True).stdout
    total_samples = len(full) // 2
    total_frames = int(round(total_samples / stream.AUDIO_SR * stream.VIDEO_FPS))
    if anchor:
        enc.set_total(total_frames, total_samples)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vf",
           f"scale={stream.VIDEO_CROP}:{stream.VIDEO_CROP}", "-r", str(stream.VIDEO_FPS),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    nbytes = stream.VIDEO_CROP * stream.VIDEO_CROP * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=nbytes * 16)
    got = 0
    try:
        while got < n_frames and (chunk := proc.stdout.read(nbytes)) and len(chunk) == nbytes:
            enc.video.push(np.frombuffer(chunk, np.uint8).reshape(
                stream.VIDEO_CROP, stream.VIDEO_CROP, 3))
            got += 1
    finally:
        proc.kill()

    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-t", str(seconds), "-i", str(path),
                          "-f", "s16le", "-ac", "1", "-ar", str(stream.AUDIO_SR), "-"],
                         capture_output=True, check=True).stdout
    enc.audio.push(np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0)
    enc.audio.flush_partial()

    for t in range(int(seconds / TR_SECONDS)):
        enc.text.push_tr(words[t] if t < len(words) else [])


def mode_extract_parity(stim: str, seconds: float, anchor: bool = True) -> None:
    tw = importlib.import_module("03_extract_text")
    words = tw.tr_words(transcript_path(DATA, "friends", stim))
    head = stream.HeadArtifact()
    enc = stream.LiveEncoder(head)
    enc.text.load()  # Llama otherwise loads on the first words, inside the timing
    t0 = time.time()
    feed_media(enc, stim, seconds, words, anchor=anchor)
    wall = time.time() - t0
    print(f"extracted {seconds:.0f}s of media in {wall:.1f}s "
          f"({seconds / wall:.2f}x realtime, unpaced)")

    n_tr = int(seconds / TR_SECONDS) - 1
    mods = {"text": F.TEXT, "audio": F.AUDIO, "video": video_modality()}
    report = {}
    for name in head.modalities:
        cached = F.load_canonical(mods[name], stim, head.layers[name])
        rows_live, rows_cache = [], []
        for t in range(n_tr):
            live = enc.text.tr_row(t) if name == "text" else \
                (enc.video if name == "video" else enc.audio).pool.tr_row(t)
            if live is not None and t < len(cached):
                rows_live.append(live)
                rows_cache.append(cached[t])
        a, b = np.stack(rows_live), np.stack(rows_cache).astype(np.float32)
        corr = [float(np.corrcoef(a[i], b[i])[0, 1]) for i in range(len(a))
                if a[i].std() > 0 and b[i].std() > 0]
        report[name] = {"n_tr": len(a), "mean_row_corr": float(np.mean(corr)),
                        "min_row_corr": float(np.min(corr))}
        print(f"  {name:<6} {len(a)} TRs   row corr vs cache: mean {np.mean(corr):.4f}  "
              f"min {np.min(corr):.4f}")
    # Prediction-level parity: same TRs, live rows vs cached rows, through the head.
    bold = episode_bold(stim)
    n = min(n_tr - EXCLUDED_START, min(b.shape[0] for b in bold.values()))
    z_live = []
    for t in range(n + EXCLUDED_START):
        parts = []
        for name in head.modalities:
            if name == "text":
                raw = enc.text.tr_row(t)
            else:
                raw = (enc.video if name == "video" else enc.audio).pool.tr_row(t)
            parts.append(head.project(name, raw[None, :])[0])
        z_live.append(np.concatenate(parts))
    z_live = np.stack(z_live)
    z_cache = cached_rows(head, stim, video_modality())
    m = min(len(z_live), len(z_cache))
    z_live, z_cache = z_live[:m], z_cache[:m]
    # which modality's PROJECTED rows diverge? (250 columns each, in head order)
    for k, name in enumerate(head.modalities):
        a = z_live[:, k * 250 : (k + 1) * 250]
        b = z_cache[:, k * 250 : (k + 1) * 250]
        ac = a - a.mean(1, keepdims=True)
        bc = b - b.mean(1, keepdims=True)
        cc = float(np.mean((ac * bc).sum(1) /
                           (np.sqrt((ac * ac).sum(1) * (bc * bc).sum(1)) + 1e-9)))
        print(f"  projected {name:<6} row corr live-vs-cache: {cc:.4f}")
    # mix-and-match: swap one modality at a time to attribute the prediction gap
    mixes = {"live": z_live, "cached": z_cache}
    for k, name in enumerate(head.modalities):
        z_mix = z_cache.copy()
        z_mix[:, k * 250 : (k + 1) * 250] = z_live[:, k * 250 : (k + 1) * 250]
        mixes[f"cached+live-{name}"] = z_mix
    for label, z in mixes.items():
        preds = predict_episode(head, z, n)
        rs = [float(pearson_per_parcel(preds[s], bold[s][:n]).mean()) for s in sorted(bold)]
        print(f"  {label:<22} mean r {float(np.mean(rs)):.4f}")
    p_live = predict_episode(head, z_live, n)
    p_cache = predict_episode(head, z_cache, n)
    for s in sorted(bold):
        y = bold[s][:n]
        rl = float(pearson_per_parcel(p_live[s], y).mean())
        rc = float(pearson_per_parcel(p_cache[s], y).mean())
        pc = float(np.mean([np.corrcoef(p_live[s][i], p_cache[s][i])[0, 1] for i in range(n)]))
        print(f"  {s}: r live-rows {rl:.4f}   r cached-rows {rc:.4f}   pred-corr {pc:.4f}")
        report.setdefault("prediction_parity", {})[s] = {
            "r_live_rows": rl, "r_cached_rows": rc, "pred_corr": pc}
    OUT.mkdir(parents=True, exist_ok=True)
    report["anchored"] = anchor
    report["seconds"] = seconds
    name = "extract_parity.json" if anchor else "extract_parity_openended.json"
    (OUT / name).write_text(json.dumps(report, indent=2))


def mode_attrib(stim: str) -> None:
    """Split live-vs-offline accuracy: is the gap audio provisionality, or the live path?

    The paced run emits predictions whose newest audio TRs come from a provisional
    partial chunk. After the run, every chunk is finalized. Predicting again from the
    extractors' END-state rows isolates that one difference: emitted-live vs end-state
    is the price of freshness; end-state vs cached-offline is whatever the live
    extraction/pooling path itself costs.
    """
    tw = importlib.import_module("03_extract_text")
    words = tw.tr_words(transcript_path(DATA, "friends", stim))
    head = stream.HeadArtifact()
    enc = stream.LiveEncoder(head)
    enc.text.load()  # Llama otherwise loads on the first words, inside the timing
    bold = episode_bold(stim)
    n = min(b.shape[0] for b in bold.values())
    seconds = (n + EXCLUDED_START + 2) * TR_SECONDS
    t0 = time.time()
    feed_media(enc, stim, seconds, words)
    print(f"fed {seconds:.0f}s in {time.time() - t0:.0f}s; predicting from end-state rows")

    rows = []
    for t in range(n + EXCLUDED_START):
        parts = []
        for name in head.modalities:
            if name == "text":
                raw = enc.text.tr_row(t)
            else:
                raw = (enc.video if name == "video" else enc.audio).pool.tr_row(t)
            parts.append(head.project(name, raw[None, :])[0])
        rows.append(np.concatenate(parts))
    z = np.stack(rows)
    preds = predict_episode(head, z, n)
    rs = {s: float(pearson_per_parcel(preds[s][: bold[s].shape[0]], bold[s][:n]).mean())
          for s in bold}
    mean = float(np.mean(list(rs.values())))
    print(f"end-state live rows: mean r {mean:.4f}   "
          + " ".join(f"{s.split('-')[1]}:{v:.4f}" for s, v in rs.items()))
    print("compare with head-parity's cached-offline number (runs/stream_gate/head_parity.json)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attrib.json").write_text(json.dumps({"end_state_mean_r": mean, "per_subject": rs},
                                                indent=2))


def mode_gate(stim: str, limit_s: float, record: bool = False) -> None:
    """Paced 1x run: media is released on the wall clock; every TR must clear in time."""
    tw = importlib.import_module("03_extract_text")
    words = tw.tr_words(transcript_path(DATA, "friends", stim))
    head = stream.HeadArtifact()
    enc = stream.LiveEncoder(head)
    enc.text.load()  # Llama otherwise loads on the first words, inside the timing
    path = movie_path(stim)

    nbytes = stream.VIDEO_CROP * stream.VIDEO_CROP * 3
    vproc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vf",
         f"scale={stream.VIDEO_CROP}:{stream.VIDEO_CROP}", "-r", str(stream.VIDEO_FPS),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, bufsize=nbytes * stream.FRAMES_PER_CLIP)
    wav = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "s16le",
                          "-ac", "1", "-ar", str(stream.AUDIO_SR), "-"],
                         capture_output=True).stdout
    wav = np.frombuffer(wav, np.int16).astype(np.float32) / 32768.0
    # Episode playback: length known up front, so the pooling grids anchor to it and
    # reproduce the training grid exactly (see TrPooler). Open-ended input cannot.
    enc.set_total(int(len(wav) / stream.AUDIO_SR * stream.VIDEO_FPS), len(wav))

    t0 = time.time()
    fed_frames, fed_samples, ticks, lags = 0, 0, [], []
    recorded: dict[int, dict[str, np.ndarray]] = {}
    misses, s_emitted = 0, -1
    # The first video clip only completes 2.13 s in, so predictions for the first
    # couple of TRs are structurally impossible, not deadline misses.
    warmup = 3
    duration = min(limit_s, len(wav) / stream.AUDIO_SR)
    print(f"gate: {stim}, {duration / 60:.1f} min at 1x, device {enc.device}")
    while True:
        now = time.time() - t0
        if now >= duration:
            break
        # release media up to the wall clock
        want_frames = min(int(now * stream.VIDEO_FPS), int(duration * stream.VIDEO_FPS))
        while fed_frames < want_frames:
            chunk = vproc.stdout.read(nbytes)
            if not chunk or len(chunk) < nbytes:
                break
            enc.video.push(np.frombuffer(chunk, np.uint8).reshape(
                stream.VIDEO_CROP, stream.VIDEO_CROP, 3))
            fed_frames += 1
        want_samples = int(now * stream.AUDIO_SR)
        if want_samples > fed_samples:
            enc.audio.push(wav[fed_samples:want_samples])
            fed_samples = want_samples
        while len(enc.text.rows) < int(now / TR_SECONDS):
            t = len(enc.text.rows)
            enc.text.push_tr(words[t] if t < len(words) else [])

        s_now = int(now / TR_SECONDS)
        if s_now > s_emitted:
            # Predict EVERY slot up to s_now, not just s_now: a slot's stimulus window
            # is fixed by its index, not the wall clock, so a late prediction is the
            # same computation delivered late -- and lag records the lateness. Without
            # this, falling behind silently skips slots, and the miss counter cannot
            # see them.
            for s in range(s_emitted + 1, s_now + 1):
                tick_start = time.time()
                enc.audio.flush_partial()
                preds = enc.predict(s)
                dt = time.time() - tick_start
                ticks.append(dt)
                if preds is None and s >= warmup:
                    misses += 1
                if record and preds is not None:
                    recorded[s] = preds
                lag = (time.time() - t0) - s * TR_SECONDS
                lags.append(lag)
                if s % 40 == 0:
                    print(f"  TR {s:>4}  tick {dt * 1000:>6.0f} ms  lag {lag:>6.2f}s  "
                          f"misses {misses}")
            s_emitted = s_now
        else:
            time.sleep(0.02)
    vproc.kill()

    ticks_arr, lags_arr = np.array(ticks), np.array(lags)
    half = len(ticks_arr) // 2
    n_slots = int(duration / TR_SECONDS) + 1
    result = {
        "episode": stim, "duration_s": duration, "n_ticks": len(ticks_arr),
        "n_slots": n_slots,
        "n_predicted": len(recorded) if record else None,
        "misses_after_warmup": misses,
        "lag_s": {"mean": float(lags_arr.mean()), "p95": float(np.percentile(lags_arr, 95)),
                   "max": float(lags_arr.max())},
        "tick_ms": {"mean": float(ticks_arr.mean() * 1000),
                     "p95": float(np.percentile(ticks_arr, 95) * 1000),
                     "max": float(ticks_arr.max() * 1000)},
        "budget_ms": TR_SECONDS * 1000,
        "first_half_mean_ms": float(ticks_arr[:half].mean() * 1000),
        "second_half_mean_ms": float(ticks_arr[half:].mean() * 1000),
        "sustained": bool(ticks_arr.mean() < TR_SECONDS and misses == 0),
        "note": "misses exclude the first 3 warm-up TRs (first 64-frame clip lands at "
        "2.13s). Ticks time flush_partial+predict only; extractor forwards run in the "
        "media-feed path, which is why lag, not tick, is the end-to-end freshness measure.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate.json").write_text(json.dumps(result, indent=2))
    if record and recorded:
        samples = sorted(recorded)
        arrays = {"samples": np.array(samples, dtype=np.int32),
                  "tick_ms": ticks_arr * 1000, "lag_s": lags_arr}
        for subj in head.subjects:
            arrays[subj] = np.stack([recorded[s][subj] for s in samples]).astype(np.float32)
        np.savez_compressed(OUT / f"live_{stim}.npz", **arrays)
        print(f"recorded live predictions -> {OUT / f'live_{stim}.npz'}")
    print(f"\nticks mean {result['tick_ms']['mean']:.0f} ms  p95 {result['tick_ms']['p95']:.0f} ms"
          f"  budget {result['budget_ms']:.0f} ms  misses {misses} (post-warmup)")
    print(f"tick time, first vs second half: {result['first_half_mean_ms']:.0f} -> "
          f"{result['second_half_mean_ms']:.0f} ms")
    print(f"lag: mean {result['lag_s']['mean']:.2f}s  p95 {result['lag_s']['p95']:.2f}s  "
          f"max {result['lag_s']['max']:.2f}s")
    print(f"GATE {'PASSED' if result['sustained'] else 'FAILED'}  -> {OUT / 'gate.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["head-parity", "extract-parity", "gate", "attrib"],
                    required=True)
    ap.add_argument("--episode", default=EPISODE)
    ap.add_argument("--seconds", type=float, default=120.0, help="extract-parity media length")
    ap.add_argument("--limit", type=float, default=1e9, help="gate: cap the paced run (s)")
    ap.add_argument("--record", action="store_true",
                    help="gate: also save the live per-TR predictions to runs/stream_gate/")
    ap.add_argument("--open-ended", action="store_true",
                    help="extract-parity: no length anchor -- prices truly live input")
    args = ap.parse_args()
    if args.mode == "head-parity":
        mode_head_parity(args.episode)
    elif args.mode == "extract-parity":
        mode_extract_parity(args.episode, args.seconds, anchor=not args.open_ended)
    elif args.mode == "attrib":
        mode_attrib(args.episode)
    else:
        mode_gate(args.episode, args.limit, record=args.record)


if __name__ == "__main__":
    main()
