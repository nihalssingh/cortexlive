"""Run the live encoder over a video file, volume by volume.

This is the one run loop behind `cortexlive run` and the viewer's live mode, derived from
the paced mode of scripts/18_stream_gate.py:

  * realtime pace: frames and audio are released on the wall clock (frame i when 1x
    playback reaches it), so the streams stay aligned however long a V-JEPA2 forward
    blocks.
  * a volume is emitted once every stimulus TR in its window has features, and every
    volume is emitted -- late ones catch up -- so lag (emission time minus the volume's
    own time in the input) measures freshness.
  * fast pace: the same path, clocked by the media position instead of the wall clock,
    for batch runs. It waits for finalized audio instead of re-forwarding the partial
    chunk, so its predictions reproduce the offline pipeline; lag means nothing there,
    throughput does.

A volume `s` is the predicted BOLD at time s * TR of the input. Because blood flow lags
neural activity, it is predicted from what was on screen about 3-10 s earlier (a 5-TR
window ending 2 TRs before s * TR).
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import stream
from .align import EXCLUDED_START
from .data import TR_SECONDS

FFMPEG = ("ffmpeg", "-nostdin", "-v", "error")  # -nostdin: a keypress must not reach it
NBYTES = stream.VIDEO_CROP * stream.VIDEO_CROP * 3
SR = stream.AUDIO_SR
FPS = stream.VIDEO_FPS
_TO_RAW = ["-vf", f"scale={stream.VIDEO_CROP}:{stream.VIDEO_CROP}", "-r", str(FPS),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]


class RunError(RuntimeError):
    """An input the model cannot run on, explained."""


@dataclass
class Volume:
    s: int  # volume index: predicted BOLD at s * TR of the input
    lag_s: float  # emission time minus s * TR (realtime pace)
    tick_ms: float  # partial-audio refresh (first volume of a batch) + the head
    wall_s: float  # emission time since the run started
    preds: dict[str, np.ndarray]  # subject -> (n_parcels,)


@dataclass
class RunResult:
    volumes: list[Volume]
    media_s: float  # input length, s
    wall_s: float
    pace: str
    stopped: bool = False
    notes: list[str] = field(default_factory=list)


def probe(path: str | Path) -> dict:
    """Duration and which streams a file has (ffprobe)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration:stream=codec_type", "-of", "json", str(path)],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if out.returncode != 0:
        raise RunError(f"cannot read {path}: {out.stderr.strip()[:300]}")
    info = json.loads(out.stdout or "{}")
    kinds = {s.get("codec_type") for s in info.get("streams", [])}
    return {"duration": float(info.get("format", {}).get("duration") or 0.0),
            "video": "video" in kinds, "audio": "audio" in kinds}


def _frame(chunk: bytes) -> np.ndarray:
    return np.frombuffer(chunk, np.uint8).reshape(stream.VIDEO_CROP, stream.VIDEO_CROP, 3)


class _FileFrames:
    """A file's frames, released up to a clock position.

    At most one clip is read per call, so a machine slower than real time leaves its
    backlog in ffmpeg's pipe (bounded) rather than in memory, and the run loop gets to
    emit volumes and notice a stop between clips.
    """

    def __init__(self, path: str | Path):
        # Its own session: a ctrl-c in the terminal stops the run, not the decoder.
        self.proc = subprocess.Popen([*FFMPEG, "-i", str(path), *_TO_RAW],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     start_new_session=True,
                                     bufsize=NBYTES * stream.FRAMES_PER_CLIP)
        self.read = 0
        self.eof = False

    def due(self, now: float) -> list[np.ndarray]:
        out = []
        while (not self.eof and self.read < int(now * FPS)
               and len(out) < stream.FRAMES_PER_CLIP):
            chunk = self.proc.stdout.read(NBYTES)
            if len(chunk) < NBYTES:
                self.eof = True
                break
            out.append(_frame(chunk))
            self.read += 1
        return out

    def close(self) -> None:
        self.proc.kill()
        self.proc.stdout.close()
        self.proc.wait()


def decode_audio(path: str | Path, info: dict) -> tuple[np.ndarray, list[str]]:
    """The whole audio track at 16 kHz mono; silence where the file has none."""
    n = int(round(info["duration"] * SR))
    if not info["audio"]:
        return np.zeros(n, np.float32), ["no audio track: the audio stream was fed silence"]
    raw = subprocess.run([*FFMPEG, "-i", str(path), "-vn", "-f", "s16le", "-ac", "1",
                          "-ar", str(SR), "-"], capture_output=True,
                         start_new_session=True).stdout
    wav = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if len(wav) < n - SR:  # audio ends well before the video: keep the streams aligned
        note = (f"audio ends at {len(wav) / SR:.1f} s of {info['duration']:.1f} s: "
                "padded with silence")
        return np.concatenate([wav, np.zeros(n - len(wav), np.float32)]), [note]
    return wav, []


def run(enc: stream.LiveEncoder, *, path: str | Path, pace: str = "realtime",
        words: list[list[str]] | None = None,
        on_start: Callable[[float | None], None] | None = None,
        on_volume: Callable[[Volume], None] | None = None,
        stop: threading.Event | None = None,
        log: Callable[[str], None] = print) -> RunResult:
    """Stream one input through `enc`; every emitted volume goes to `on_volume`."""
    if pace not in ("realtime", "fast"):
        raise ValueError(pace)
    enc.reset()
    info = probe(path)
    if not info["video"]:
        raise RunError(f"{path} has no video stream")
    wav, notes = decode_audio(path, info)
    media_s = len(wav) / SR
    if media_s * FPS < stream.FRAMES_PER_CLIP:
        raise RunError(f"{path} is shorter than one 64-frame video clip (2.1 s)")
    if int((media_s - 1e-9) / TR_SECONDS) < EXCLUDED_START:  # no volume past the start trim
        raise RunError(f"{path} is too short ({media_s:.1f} s): predictions start "
                       f"{EXCLUDED_START * TR_SECONDS:.2f} s into a video, because the brain "
                       "responds several seconds after the picture and training leaves out "
                       f"the first {EXCLUDED_START} volumes of every run")
    # Known length: anchor both pooling grids, for parity with training.
    enc.set_total(int(media_s * FPS), len(wav))
    frames = _FileFrames(path)
    for n in notes:
        log(n)

    last_media = int((media_s - 1e-9) / TR_SECONDS)  # the last volume inside the video
    volumes: list[Volume] = []
    t0 = time.time()
    if on_start and not (stop is not None and stop.is_set()):
        on_start(media_s)
    fed = flushed_at = 0
    s_next = 0
    stopped = False
    try:
        while True:
            if stop is not None and stop.is_set():
                stopped = True
                break
            if pace == "realtime":
                clock = time.time() - t0
            else:  # media clock: one more clip per pass
                clock = (frames.read + stream.FRAMES_PER_CLIP) / FPS
            for f in frames.due(clock):
                enc.video.push(f)  # a V-JEPA2 forward runs on every 64th frame
            now = time.time() - t0 if pace == "realtime" else frames.read / FPS
            end = frames.eof
            if end and frames.read < media_s * FPS - stream.FRAMES_PER_CLIP:
                # The picture stopped early (a truncated file): no volume past it.
                video_s = frames.read / FPS
                last_media = min(last_media, int((video_s - 1e-9) / TR_SECONDS))
                notes.append(f"video ends at {video_s:.1f} s of {media_s:.1f} s: "
                             "volumes stop there")
                log(notes[-1])
                media_s = video_s

            want = len(wav) if end else min(int(now * SR), len(wav))
            if want > fed:
                enc.audio.push(wav[fed:want])
                fed = want
            n_text = int(np.ceil(media_s / TR_SECONDS)) if end else int(now / TR_SECONDS)
            while len(enc.text.rows) < n_text:
                t = len(enc.text.rows)
                enc.text.push_tr(words[t] if words and t < len(words) else [])

            # Every volume whose time has come, once its whole window has features. At
            # the end of a file every stream is final, so the rest go out clamped at the
            # edges exactly as align_features clamps the training data.
            last = last_media if end else min(int(now / TR_SECONDS), last_media)
            progressed = False
            if s_next <= last:
                tick = time.time()
                # realtime: refresh the provisional partial audio chunk so the newest
                # volumes exist; fast: wait for finalized chunks (offline parity).
                if fed != flushed_at and (pace == "realtime" or end):
                    enc.audio.flush_partial()
                    flushed_at = fed
                ready = last if end else min(last, enc.ready_through())
                while s_next <= ready:
                    preds = enc.predict(s_next)
                    if preds is None:
                        break  # warm-up: nothing pooled yet
                    wall = time.time() - t0
                    v = Volume(s=s_next, lag_s=wall - s_next * TR_SECONDS,
                               tick_ms=(time.time() - tick) * 1000, wall_s=wall, preds=preds)
                    volumes.append(v)
                    if on_volume:
                        on_volume(v)
                    s_next += 1
                    progressed = True
                    tick = time.time()
            if end and (s_next > last or not progressed):
                break
            if not progressed and pace == "realtime":
                time.sleep(0.01)
    finally:
        frames.close()
    return RunResult(volumes=volumes, media_s=media_s, wall_s=time.time() - t0, pace=pace,
                     stopped=stopped, notes=notes)
