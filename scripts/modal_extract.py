"""Extract audio + video features on Modal. Fan-out across containers, resumable.

The work is sharded across many containers that each process a handful of stimuli, write
straight to a persistent Volume, and exit. A container dying costs one stimulus, and
re-running skips everything already done.

    uv run --extra remote modal token new             # once, opens a browser
    uv run --extra remote modal run scripts/modal_extract.py --limit 3   # measure first
    uv run --extra remote modal run scripts/modal_extract.py             # then all 336

Run it through `uv run`, not a standalone `modal`: the @local_entrypoint executes in
whatever Python invoked the CLI, and it imports cortexlive (and so h5py) to enumerate
stimuli. A global `modal` install has neither and fails before reaching the cloud.

Results land in the `cortexlive-cache` Volume; copy them into cache/ with
    uv run --extra remote python scripts/16_pull_cache.py --what video audio

The audio and video features behind the shipped model came from this image. Only torch is
pinned, and the other library versions used were not recorded; the live extractors, on the
versions in uv.lock, reproduce those features (runs/stream_gate/extract_parity.json).
"""

from __future__ import annotations

import pathlib
import sys

import modal

APP_NAME = "cortexlive-extract"
DATASET_URL = "https://github.com/courtois-neuromod/algonauts_2025.competitors.git"

VIDEO_MODEL = "facebook/vjepa2-vitl-fpc64-256"
AUDIO_MODEL = "facebook/w2v-bert-2.0"

FRAMES_PER_CLIP = 64
CROP = 256
VIDEO_FPS = 30
AUDIO_SR = 16_000
TARGET_HZ = 2.0

app = modal.App(APP_NAME)
cache_vol = modal.Volume.from_name("cortexlive-cache", create_if_missing=True)
hf_vol = modal.Volume.from_name("cortexlive-hf", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "git-annex", "wget")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.56",
        "numpy",
        "datalad",
        "huggingface_hub",
    )
    # Bake the dataset metadata into the image so containers do not each re-clone it.
    # This pulls git/annex pointers only -- no movie bytes.
    .run_commands(
        "git config --global user.email cortexlive@example.com",
        "git config --global user.name cortexlive",
        f"datalad install -r -s {DATASET_URL} /data/algonauts || true",
    )
    .env({"HF_HOME": "/hf"})
)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": cache_vol, "/hf": hf_vol},
    timeout=60 * 60,
    retries=modal.Retries(max_retries=2),
)
def extract_batch(items: list[tuple[str, str]], modality: str, crop: int = CROP) -> list[str]:
    """Process (stimulus, source) pairs. Returns the stimuli actually written.

    `crop` exists to price the resolution lever. Resolution is the main speed knob for
    video (attention is O(n^2) in patches) but, unlike layer depth and precision, it
    cannot be swept from the cached features -- it changes what goes *into* the model.
    Spatial mean pooling means the output dimension is 1024 whatever the crop, so a head
    fit on 256 features can be scored directly on 192 ones.
    """
    import subprocess

    import numpy as np
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    data = pathlib.Path("/data/algonauts")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def ffmpeg_video(path, fps=VIDEO_FPS, size=crop):
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"scale={size}:{size}",
            "-r",
            str(fps),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        n = size * size * 3
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=n * 16)
        try:
            while (chunk := proc.stdout.read(n)) and len(chunk) == n:
                yield np.frombuffer(chunk, np.uint8).reshape(size, size, 3)
        finally:
            proc.stdout.close()
            proc.wait()

    def ffmpeg_audio(path, sr=AUDIO_SR):
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-",
        ]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0

    def resample_to(x, n_out):
        if x.shape[0] == n_out:
            return x
        src, dst = np.linspace(0, 1, x.shape[0]), np.linspace(0, 1, n_out)
        flat = x.reshape(x.shape[0], -1)
        res = np.empty((n_out, flat.shape[1]), dtype=np.float32)
        for j in range(flat.shape[1]):
            res[:, j] = np.interp(dst, src, flat[:, j])
        return res.reshape((n_out, *x.shape[1:]))

    def movie_path(stim, source):
        # NOTE: friends movies are prefixed (friends_s01e01a.mkv) but movie10 movies are
        # NOT (life01.mkv), even though movie10 *transcripts* are (movie10_life01.tsv).
        if source == "friends":
            season = int(stim[1:3])
            return data / "stimuli/movies/friends" / f"s{season}" / f"friends_{stim}.mkv"
        movie = "".join(c for c in stim if not c.isdigit())
        return data / "stimuli/movies/movie10" / movie / f"{stim}.mkv"

    models, fes, outdirs = {}, {}, {}
    if modality in ("video", "both"):
        # crop_size MUST be overridden: HF V-JEPA2 freezes its 3D-RoPE grid from
        # config.crop_size at init and never looks at the input tensor, so feeding
        # 192px pixels to a 256-config model scrambles every positional coordinate.
        # (That bug is what the _res192/_res128 caches were extracted with; the fixed
        # dirs carry a v2 suffix so the two generations can never be confused.)
        vm = AutoModel.from_pretrained(VIDEO_MODEL, dtype=torch.float16, crop_size=crop)
        models["video"] = vm.to(device).eval()
        name = VIDEO_MODEL.split("/")[-1]
        # Native resolution keeps the original directory name so the reference cache
        # stays where every earlier script expects it.
        outdirs["video"] = pathlib.Path("/cache/video") / (
            name if crop == CROP else f"{name}_res{crop}v2"
        )
    if modality in ("audio", "both"):
        am = AutoModel.from_pretrained(AUDIO_MODEL, dtype=torch.float16)
        models["audio"] = am.to(device).eval()
        fes["audio"] = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
        outdirs["audio"] = pathlib.Path("/cache/audio") / AUDIO_MODEL.split("/")[-1]
    for d in outdirs.values():
        d.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def do_video(path):
        m = models["video"]
        tub, hidden = m.config.tubelet_size, m.config.hidden_size
        grid = (crop // m.config.patch_size) ** 2
        chunks, buf = [], []
        for frame in ffmpeg_video(path):
            buf.append(frame)
            if len(buf) < FRAMES_PER_CLIP:
                continue
            x = torch.from_numpy(np.stack(buf)).to(device).permute(0, 3, 1, 2)
            x = x.half().div_(255.0).unsqueeze(0)
            buf = []
            # the predictor is skipped: no encoder state depends on it
            out = m(pixel_values_videos=x, output_hidden_states=True, skip_predictor=True)
            per_layer = [
                h.view(1, h.shape[1] // grid, grid, hidden).mean(2).squeeze(0)
                for h in out.hidden_states
            ]
            chunks.append(torch.stack(per_layer, dim=1).to(torch.float16).cpu().numpy())
        if not chunks:
            return np.zeros((0, m.config.num_hidden_layers + 1, hidden), np.float16)
        seq = np.concatenate(chunks)
        n_out = max(1, int(round(seq.shape[0] / (VIDEO_FPS / tub) * TARGET_HZ)))
        return resample_to(seq.astype(np.float32), n_out).astype(np.float16)

    @torch.no_grad()
    def do_audio(path, chunk_s=30.0):
        m, fe = models["audio"], fes["audio"]
        wav = ffmpeg_audio(path)
        step = int(chunk_s * AUDIO_SR)
        chunks = []
        for start in range(0, len(wav), step):
            seg = wav[start : start + step]
            if len(seg) < AUDIO_SR // 10:
                break
            inp = fe(seg, sampling_rate=AUDIO_SR, return_tensors="pt")
            out = m(
                input_features=inp["input_features"].to(device).half(), output_hidden_states=True
            )
            st = torch.stack(out.hidden_states, 2).squeeze(0)
            chunks.append(st.to(torch.float16).cpu().numpy())
        if not chunks:
            return np.zeros((0, m.config.num_hidden_layers + 1, m.config.hidden_size), np.float16)
        seq = np.concatenate(chunks)
        n_out = max(1, int(round(len(wav) / AUDIO_SR * TARGET_HZ)))
        return resample_to(seq.astype(np.float32), n_out).astype(np.float16)

    import time as _time

    written = []
    for stim, source in items:
        t_start = _time.time()
        todo = {
            m: outdirs[m] / f"{stim}.npy"
            for m in models
            if not (outdirs[m] / f"{stim}.npy").exists()
        }
        if not todo:
            continue
        path = movie_path(stim, source)
        rel = str(path.relative_to(data))
        try:
            subprocess.run(["datalad", "get", rel], cwd=data, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"FETCH FAILED {stim}: {e.stderr[:200]}")
            continue
        t_fetch = _time.time()

        # Timed per stage, so the logs show whether a run is download, decode or GPU
        # bound. A faster GPU only helps the last case.
        stage = {}
        for mod, dest in todo.items():
            t0 = _time.time()
            feats = do_video(path) if mod == "video" else do_audio(path)
            np.save(dest, feats)
            stage[mod] = _time.time() - t0
            print(f"{stim} {mod} {feats.shape} {feats.nbytes / 1e6:.0f}MB")

        subprocess.run(["datalad", "drop", rel], cwd=data, capture_output=True)
        written.append(stim)
        cache_vol.commit()

        total = _time.time() - t_start
        parts = " ".join(f"{k}={v:.0f}s" for k, v in stage.items())
        print(f"TIMING {stim}: total={total:.0f}s fetch={t_fetch - t_start:.0f}s {parts}")

    return written


@app.function(image=image, timeout=30 * 60, volumes={"/cache": cache_vol})
def verify(crop: int = CROP) -> dict:
    """Audit the cache server-side. Cheap, and it cannot be fooled by a healthy file count.

    Counting files does not detect the failures that actually happen here:

      * a stimulus whose movie failed to fetch is `continue`d -- no file, no error, and
        nothing in the count to say which one is missing
      * a stimulus whose video failed to decode writes a real .npy containing ZERO rows
      * a truncated decode writes a file that is short but not empty, which looks fine
        until it silently misaligns against the fMRI months later

    So this opens every array and checks rows, dtype, finiteness, and -- the strongest
    check available without the fMRI -- whether video and audio agree on the stimulus's
    duration, since the two are produced by completely independent code paths.

        uv run --extra remote modal run scripts/modal_extract.py::verify
    """
    import numpy as np

    name = VIDEO_MODEL.split("/")[-1]
    dirs = {
        "video": pathlib.Path("/cache/video") / (name if crop == CROP else f"{name}_res{crop}v2"),
        "audio": pathlib.Path("/cache/audio") / AUDIO_MODEL.split("/")[-1],
    }
    cache_vol.reload()

    rows: dict[str, dict[str, int]] = {}
    report: dict = {"counts": {}, "zero_rows": [], "not_finite": [], "bad_shape": []}
    for mod, d in dirs.items():
        files = sorted(d.glob("*.npy")) if d.exists() else []
        report["counts"][mod] = len(files)
        for f in files:
            a = np.load(f, mmap_mode="r")
            if a.ndim != 3 or a.shape[1] != 25 or a.shape[2] != 1024:
                report["bad_shape"].append(f"{mod}/{f.stem} {a.shape}")
            if a.shape[0] == 0:
                report["zero_rows"].append(f"{mod}/{f.stem}")
                continue
            rows.setdefault(f.stem, {})[mod] = int(a.shape[0])
            # Sampling beats loading 24 GB; NaN from a bad forward pass is never isolated
            # to a few frames, so a stride of 64 finds it.
            probe = np.asarray(a[::64, ::4, ::16], dtype=np.float32)
            if not np.isfinite(probe).all():
                report["not_finite"].append(f"{mod}/{f.stem}")

    # Video is EXPECTED to be a little shorter than audio. Frames are buffered into clips
    # of 64 and the final partial clip is dropped, so video loses up to 63/30 = 2.10 s
    # (4.2 samples at 2 Hz) off the tail; audio uses the whole waveform. Verified against
    # the data: observed losses run 0.33-1.80 s, all inside that bound. It is harmless
    # because `trim_fmri` already discards the last 5 TRs = 7.45 s of every run, so the
    # dropped tail sits inside a region nothing reads. Only a LARGER gap is a real fault.
    tolerance = 5
    mismatch, expected_tail = [], 0
    for stem, got in rows.items():
        if len(got) != 2:
            continue
        delta = got["audio"] - got["video"]
        if delta == 0:
            continue
        if 0 < delta <= tolerance:
            expected_tail += 1
        else:
            mismatch.append(f"{stem} video={got['video']} audio={got['audio']} delta={delta}")
    report["duration_mismatch"] = mismatch
    report["expected_tail_trim"] = expected_tail
    report["total_hours"] = round(sum(v.get("video", 0) for v in rows.values()) / 2.0 / 3600, 2)
    report["clean"] = not (
        report["zero_rows"] or report["not_finite"] or report["bad_shape"] or mismatch
    )

    print(
        f"VERIFY counts={report['counts']} hours={report['total_hours']} "
        f"expected_tail_trim={report['expected_tail_trim']}"
    )
    for key in ("zero_rows", "not_finite", "bad_shape", "duration_mismatch"):
        if report[key]:
            print(f"VERIFY {key}: {report[key][:20]}")
    print(f"VERIFY clean={report['clean']}")
    return report


@app.function(image=image, timeout=24 * 60 * 60, volumes={"/cache": cache_vol})
def drive(items: list, modality: str, crop: int, batch: int) -> int:
    """Fan out from INSIDE Modal, so the laptop is not part of the control path.

    `--detach` alone is not enough. It keeps the app alive when the client dies, but
    `.starmap()` still feeds its inputs from the local process, so a closed laptop stops
    new work being submitted. Running the loop in a container removes the laptop entirely:
    it can sleep, move network, or shut down and the run continues.

    Filtering happens here rather than in the worker because this container has the
    volume mounted and the laptop does not. A worker skips finished stimuli too, but only
    after starting and loading both backbones -- so on a restart, batches that
    are entirely done would burn GPU time to discover they have nothing to do. Dropping
    them here makes resuming nearly free, which matters because resuming is the whole
    point of running detached.

    CPU-only; it just waits on the GPU workers.
    """
    name = VIDEO_MODEL.split("/")[-1]
    dirs = {
        "video": pathlib.Path("/cache/video") / (name if crop == CROP else f"{name}_res{crop}v2"),
        "audio": pathlib.Path("/cache/audio") / AUDIO_MODEL.split("/")[-1],
    }
    wanted = [m for m in ("video", "audio") if modality in (m, "both")]
    cache_vol.reload()
    have = {
        m: {p.stem for p in dirs[m].glob("*.npy")} if dirs[m].exists() else set() for m in wanted
    }

    todo = [(s, src) for s, src in items if any(s not in have[m] for m in wanted)]
    print(
        f"DRIVER {len(items)} requested, {len(items) - len(todo)} already cached, "
        f"{len(todo)} to do",
        flush=True,
    )
    if not todo:
        print("DRIVER COMPLETE: nothing to do", flush=True)
        return 0

    batches = [todo[i : i + batch] for i in range(0, len(todo), batch)]
    done = 0
    for result in extract_batch.starmap([(b, modality, crop) for b in batches]):
        done += len(result)
        print(f"DRIVER [{done}/{len(todo)}] latest: {result}", flush=True)
    print(f"DRIVER COMPLETE: {done}/{len(todo)}", flush=True)
    return done


@app.local_entrypoint()
def main(
    modality: str = "both",
    limit: int = 0,
    batch: int = 6,
    crop: int = CROP,
    seasons: str = "",
    spawn: bool = False,
):
    """Enumerate stimuli locally (we already have the fMRI), then fan out.

    `--seasons 6` restricts to the held-out episodes, about ten hours of video instead of
    sixty-four. That is all the resolution experiment and the 192 px reference data need:
    the head stays frozen from the 256 px fit and only has to be scored.

    `--spawn` hands the fan-out to a driver container and returns immediately, so the run
    does not depend on this machine. Pair it with `modal run --detach`:

        uv run --extra remote modal run --detach scripts/modal_extract.py --spawn

    Check on it with `modal app logs <id>` or by counting files in the volume; re-running
    is always safe
    because every stimulus is skipped if its .npy already exists.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from cortexlive.data import SUBJECTS, BoldStore

    data = pathlib.Path(__file__).resolve().parent.parent / "data" / "algonauts"
    stimuli: dict[str, str] = {}
    for subject in SUBJECTS:
        for run in BoldStore(data, subject).runs:
            stimuli.setdefault(run.stimulus, run.source)

    items = sorted(stimuli.items())
    if seasons:
        want = {f"s{int(s):02d}e" for s in seasons.split(",")}
        items = [(s, src) for s, src in items if any(s.startswith(w) for w in want)]
    if limit:
        items = items[:limit]
    if not items:
        raise SystemExit("no stimuli selected")
    batches = [items[i : i + batch] for i in range(0, len(items), batch)]
    print(
        f"{len(items)} stimuli -> {len(batches)} containers of {batch}, "
        f"modality={modality}, crop={crop}"
    )

    if spawn:
        call = drive.spawn(items, modality, crop, batch)
        print(f"\ndriver spawned: {call.object_id}")
        print("the run no longer depends on this machine. Watch it with:")
        print(f"  uv run --extra remote modal app logs {app.app_id}")
        print("  uv run --extra remote modal volume ls cortexlive-cache /video")
        return

    done = 0
    for result in extract_batch.starmap([(b, modality, crop) for b in batches]):
        done += len(result)
        print(f"  [{done}/{len(items)}] latest: {result}")
    print(f"\ncomplete: {done} stimuli written to the cortexlive-cache volume")
