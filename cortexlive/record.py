"""Run one video through the model and save the result as a capture.

The single entry point behind `cortexlive run` and the viewer's live mode, so a capture
is the same whichever way it was made.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import capture, reference, stream
from . import run as runner
from .stimulus import words_per_tr


def choose_transcript(enc, name: str, transcript: str | Path,
                      log: Callable[[str], None] = print
                      ) -> tuple[dict | None, list | None, str, str | None]:
    """(known stimulus, words per TR or None, label, note) for an input called `name`.

    Words need Llama 3.2 1B, which is optional: it loads here, before the run's clock
    starts. Without it an automatic transcript is left out, and the note says so for the
    capture; a transcript asked for by path is an error instead.
    """
    known = reference.known_stimulus(name)
    words, label, note = None, "none", None
    if isinstance(transcript, Path):
        words, label = words_per_tr(transcript), transcript.name
    elif transcript == "auto" and known:
        words = reference.transcript(known)
        label = "CNeuroMod transcript" if words else "none"
    if known:
        log(f"recognised {known['stimulus']} ({known['source']}, {known['split']} split)")
    if words and not enc.text.loaded:
        problem = None if stream.text_model_installed() else "is not installed"
        if problem is None:
            log("loading Llama 3.2 1B for the transcript")
            try:
                enc.text.load()
            except OSError as e:  # e.g. offline with an incomplete download
                problem = f"could not be loaded ({str(e).splitlines()[0] if str(e) else e!r})"
        if problem:
            if isinstance(transcript, Path):
                raise runner.RunError(f"a transcript needs Llama 3.2 1B, which {problem}: "
                                      + stream.TEXT_ACCESS)
            words, label = None, "none"
            note = f"transcript left out: Llama 3.2 1B {problem}"
    if note:
        log(f"transcript: none, because Llama 3.2 1B {problem} (to add it: {stream.TEXT_ACCESS})")
    else:
        log(f"transcript: {label}")
    return known, words, label, note


def record(enc, video: Path, out: Path, *, input_name: str | None = None,
           pace: str = "realtime", transcript: str | Path = "auto",
           on_start: Callable[[float | None], None] | None = None,
           on_volume: Callable[[runner.Volume], None] | None = None,
           stop: threading.Event | None = None,
           log: Callable[[str], None] = print) -> dict:
    """Run `video`, write the capture to `out`, return its metadata.

    transcript: "auto" uses the CNeuroMod transcript when the input is a known stimulus
    (there is no speech recognition, so other inputs run with an empty text stream),
    "none" never uses one, or a path to a words-per-TR .tsv.
    """
    name = input_name or video.name
    known, words, label, note = choose_transcript(enc, name, transcript, log)

    result = runner.run(enc, path=video, pace=pace, words=words, on_start=on_start,
                        on_volume=on_volume, stop=stop, log=log)
    meta = capture.save(out, result, input_path=video.absolute(), input_name=name,
                        device=enc.device, transcript=label, head=enc.head)
    if known:
        meta["known"] = known
        if note:
            meta["run"]["notes"] = [*meta["run"]["notes"], note]
        if result.volumes:
            log("scoring against the recorded fMRI ...")
            _, arrays = capture.load(out)
            scores, parcels = reference.score(arrays, known)
            np.savez_compressed(out / "scores.npz", **{
                f"{s}_{k}": v for s, d in parcels.items() for k, v in d.items()})
            meta["scores"] = scores
        capture.write_meta(out, meta)
    return meta
