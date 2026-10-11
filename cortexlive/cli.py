"""cortexlive -- predict human cortical responses to any video, in real time.

    cortexlive view [CAPTURE]       open the viewer: drop a video to run it live, or replay captures
    cortexlive run VIDEO            run the model over a video (1x real time), save a capture
    cortexlive export CAPTURE       write a capture as one HTML file that opens without a server
    cortexlive doctor               check this machine can run the model
    cortexlive setup                download the backbones / install a model bundle
    cortexlive bundle               pack the trained model and reference data for distribution
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

from . import capture

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
BACKBONES = ("facebook/vjepa2-vitl-fpc64-256", "facebook/w2v-bert-2.0", "meta-llama/Llama-3.2-1B")
TEXT_BACKBONE = BACKBONES[2]  # optional: gated by Meta, and only transcripts use it
BUNDLE = ("models/final_head.npz", "runs/final_head/meta.json", "models/reference.npz",
          "models/control_head.npz")  # what setup installs from a bundle, relative to the checkout
BUNDLE_NOTICES = ("NOTICE", "licenses/MODEL_LICENSE.md",
                  "licenses/LLAMA_3.2_COMMUNITY_LICENSE.txt")  # travel inside every bundle
# The Llama 3.2 license asks that a model trained on Llama outputs carry "Llama" at the start
# of its name; the head is fit on Llama 3.2 features.
MODEL_NAME = "Llama-cortexlive"


def _captures_root(args) -> Path:
    root = args.captures or os.environ.get("CORTEXLIVE_CAPTURES")
    return Path(root).expanduser() if root else capture.DEFAULT_ROOT


def _fmt_time(s: float) -> str:
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


# --------------------------------------------------------------------------- run


def _r(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def cmd_run(args) -> int:
    from .align import EXCLUDED_START
    from .data import TR_SECONDS
    from .record import record
    from .run import RunError
    from .stream import (
        HEAD_PATH,
        META_PATH,
        TEXT_ACCESS,
        LiveEncoder,
        ModelBusy,
        text_model_installed,
    )

    video = Path(args.video).expanduser()
    if not video.is_file():
        print(f"no such file: {video}", file=sys.stderr)
        return 2
    t = args.transcript
    transcript = Path(t).expanduser() if t not in ("auto", "none") else t
    if isinstance(transcript, Path) and not transcript.is_file():
        print(f"no such transcript: {transcript}", file=sys.stderr)
        return 2
    if isinstance(transcript, Path) and not text_model_installed():
        print(f"a transcript needs Llama 3.2 1B, which is not installed: {TEXT_ACCESS}",
              file=sys.stderr)
        return 2
    if not (HEAD_PATH.exists() and META_PATH.exists()):
        print("the trained head is not installed: cortexlive setup --models <bundle> "
              "(cortexlive doctor lists what is missing)", file=sys.stderr)
        return 2
    out = (Path(args.out).expanduser() if args.out
           else capture.new_dir(_captures_root(args), video.name))
    out.mkdir(parents=True, exist_ok=True)
    log_file = (out / "run.log").open("w")
    tty = sys.stdout.isatty()
    stop = threading.Event()

    def log(msg: str) -> None:
        log_file.write(msg + "\n")
        log_file.flush()
        print(("\r\033[K" if tty else "") + msg)

    def on_volume(v) -> None:
        if tty:
            lag = f"  lag {v.lag_s:5.2f}s" if args.pace == "realtime" else ""
            print(f"\r\033[K  volume {v.s + 1:>5}  t={_fmt_time(v.s * 1.49)}{lag}", end="",
                  flush=True)
        elif v.s % 50 == 0:
            lag = f" lag {v.lag_s:.2f}s" if args.pace == "realtime" else ""
            log_file.write(f"volume {v.s}{lag}\n")

    def on_sigint(signum, frame) -> None:  # first ctrl-c: stop and keep what was computed
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set()
        print(("\r\033[K" if tty else "") + "stopping (ctrl-c again to abort)", flush=True)

    previous = signal.signal(signal.SIGINT, on_sigint)
    log(f"cortexlive run {video.name} -> {out}")
    meta, code = None, 0
    try:
        log("loading models ...")
        enc = LiveEncoder()
        if not stop.is_set():
            meta = record(enc, video, out, pace=args.pace, transcript=transcript,
                          on_volume=on_volume, stop=stop, log=log)
    except KeyboardInterrupt:
        log("interrupted")
        code = 130
    except (RunError, ModelBusy, OSError) as e:
        hint = ("" if isinstance(e, (RunError, ModelBusy))
                else " (cortexlive doctor lists what is missing)")
        log(f"error: {e}{hint}")
        code = 1
    finally:
        signal.signal(signal.SIGINT, previous)
        if tty:
            print()
    if meta is None or meta["run"]["volumes"] <= EXCLUDED_START:  # nothing to show
        if meta is not None:
            log(f"stopped before the first prediction ({EXCLUDED_START * TR_SECONDS:.2f} s "
                "into the video)" + ("" if args.out else ": nothing kept"))
        log_file.close()
        if not args.out:
            shutil.rmtree(out, ignore_errors=True)
        return code or 130
    r = meta["run"]
    line = f"{r['volumes']}/{r['expected_volumes']} volumes in {r['wall_s']:.1f}s"
    if r["lag_s"]:
        line += f", lag {r['lag_s']['mean']:.2f}s mean / {r['lag_s']['p95']:.2f}s p95"
    elif r["realtime_factor"]:
        line += f" ({r['realtime_factor']:.2f}x real time)"
    log(line + (" (stopped)" if r["stopped"] else ""))
    sc = meta.get("scores")
    if sc and sc.get("mean"):
        m = sc["mean"]
        log(f"vs recorded fMRI ({sc['stimulus']}): r {_r(m['r'])} (control {_r(m['control_r'])}),"
            f" reliable parcels r {_r(m['r_reliable'])}")
    log(f"view: cortexlive view {out}")
    log_file.close()
    return 130 if r["stopped"] else 0


# -------------------------------------------------------------------------- view


def _unopenable(path: Path) -> bool:
    """Say why a folder can't be opened as a capture; False if it can."""
    if not (path / "capture.json").exists():
        print(f"not a capture: {path}", file=sys.stderr)
    elif not (path / "predictions.npz").exists():
        print(f"{path} keeps only its summary (capture.json), not its predictions, so it "
              "can't be opened", file=sys.stderr)
    else:
        return False
    return True


def cmd_view(args) -> int:
    from .viewer.server import serve

    target = Path(args.capture).expanduser() if args.capture else None
    if target is not None and _unopenable(target):
        return 2
    try:
        serve(_captures_root(args), port=args.port, open_capture=target,
              browser=not args.no_browser)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        print(f"port {args.port} is already in use, probably by another cortexlive viewer: "
              f"use that one (http://localhost:{args.port}/), quit it with ctrl-c in its "
              f"terminal, or pick another port with --port {args.port + 1}", file=sys.stderr)
        return 2
    return 0


# ------------------------------------------------------------------------ export


def cmd_export(args) -> int:
    from .viewer import html, payload

    src = Path(args.capture).expanduser()
    if _unopenable(src):
        return 2
    embed = {"atlas": payload.atlas(), "capture": payload.capture_payload(src)}
    out = Path(args.output).expanduser() if args.output else src / f"{src.name}.html"
    out.write_text(html(embed))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


# ------------------------------------------------------------------------ doctor


def _check(ok: bool, label: str, fix: str = "", optional: bool = False) -> bool:
    mark = "ok" if ok else "--" if optional else "MISSING"
    hint = "" if ok or not fix else f"\n           {fix}"
    print(f"  {mark:<7}  {label}{hint}")
    return ok


def cmd_doctor(args) -> int:
    print("cortexlive doctor")
    ok = True
    ok &= _check(sys.version_info >= (3, 11), f"python {sys.version.split()[0]}", "needs 3.11+")
    for tool in ("ffmpeg", "ffprobe"):
        ok &= _check(shutil.which(tool) is not None, tool, "brew install ffmpeg")
    try:
        import torch

        mps = torch.backends.mps.is_available()
        where = "mps" if mps else "cpu only: slower than real time"
        _check(True, f"torch {torch.__version__} ({where})")
    except ImportError:
        ok &= _check(False, "torch", "uv sync --extra live")
    try:
        import transformers

        _check(True, f"transformers {transformers.__version__}")
    except ImportError:
        ok &= _check(False, "transformers", "uv sync --extra live")
    try:
        from huggingface_hub import try_to_load_from_cache

        for repo in BACKBONES:
            have = isinstance(try_to_load_from_cache(repo, "config.json"), str)
            if repo == TEXT_BACKBONE:
                from .stream import TEXT_ACCESS, text_model_installed

                _check(text_model_installed(), f"backbone {repo}",
                       "optional: reads the transcripts of Friends and Movie10 files; "
                       f"for access, {TEXT_ACCESS}", optional=True)
            else:
                ok &= _check(have, f"backbone {repo}", "cortexlive setup")
    except ImportError:
        ok &= _check(False, "huggingface-hub", "uv sync --extra live")
    ok &= _check((MODELS / "final_head.npz").exists() and
                 (ROOT / "runs" / "final_head" / "meta.json").exists(),
                 "trained head (models/final_head.npz)", "cortexlive setup --models <bundle>")
    _check((MODELS / "reference.npz").exists(), "reference data (models/reference.npz)",
           "optional: reliability masks and whole-run maps", optional=True)
    _check((MODELS / "control_head.npz").exists(), "control baseline (models/control_head.npz)",
           "optional: the speech-feature control for CNeuroMod inputs", optional=True)
    _check((ROOT / "data" / "algonauts").exists(), "CNeuroMod data (data/algonauts)",
           "optional: scores Friends / Movie10 inputs against recorded fMRI", optional=True)
    print("ready" if ok else "not ready -- see above")
    return 0 if ok else 1


# ------------------------------------------------------------------------- setup


def _install_bundle(src: str) -> int:
    if src.startswith("http://"):
        print("use an https URL for the model bundle", file=sys.stderr)
        return 2
    downloaded = None
    try:
        if src.startswith("https://"):
            import urllib.request
            from urllib.parse import urlparse

            MODELS.mkdir(exist_ok=True)
            downloaded = MODELS / (Path(urlparse(src).path).name or "bundle.tar.gz")
            print(f"downloading {src}")
            urllib.request.urlretrieve(src, downloaded)
        path = downloaded or Path(src).expanduser()
        if not path.is_file():
            print(f"no such bundle: {path}", file=sys.stderr)
            return 2
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        installed = []
        with tarfile.open(path) as tar:
            for name in BUNDLE:  # only these files, whatever else the archive holds
                try:
                    member = tar.getmember(name)
                except KeyError:
                    continue
                if member.isfile():
                    dest = ROOT / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(tar.extractfile(member).read())
                    installed.append(name)
    except (OSError, tarfile.TarError) as e:  # a failed download, or not an archive
        print(f"{src} is not a cortexlive model bundle, or could not be read: {e}",
              file=sys.stderr)
        return 1
    finally:
        if downloaded is not None:
            downloaded.unlink(missing_ok=True)
    if not {"models/final_head.npz", "runs/final_head/meta.json"} <= set(installed):
        print(f"{path.name} is not a cortexlive model bundle", file=sys.stderr)
        return 1
    print(f"installed {', '.join(installed)}\nbundle sha256 {digest}\n"
          f"{MODEL_NAME}: CC BY-NC 4.0, built with Llama (licenses/MODEL_LICENSE.md)")
    return 0


def cmd_setup(args) -> int:
    if args.models:
        code = _install_bundle(args.models)
        if code:
            return code
    if not args.skip_backbones:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("huggingface-hub missing: uv sync --extra live", file=sys.stderr)
            return 1
        for repo in BACKBONES:
            print(f"fetching {repo} ...")
            try:
                snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.model",
                                                        "tokenizer*"])
            except Exception as e:
                # the Hub's own reason, e.g. "awaiting a review", rather than the status line
                msg = getattr(e, "server_message", None) or next(
                    (ln for ln in str(e).splitlines() if ln.strip()), type(e).__name__)
                if repo == TEXT_BACKBONE:
                    from huggingface_hub.errors import RepositoryNotFoundError  # gated: a subclass

                    from .stream import TEXT_ACCESS

                    hint = (f"For access, {TEXT_ACCESS}." if isinstance(e, RepositoryNotFoundError)
                            else "Run `cortexlive setup` again to finish it.")
                    print(f"  skipped: {msg}\n  Llama 3.2 1B is optional: it reads the "
                          f"transcripts of Friends and Movie10 files. {hint}")
                    continue
                print(f"  failed: {msg}")
                return 1
    return cmd_doctor(args)


# ------------------------------------------------------------------------ bundle


def cmd_bundle(args) -> int:
    from . import reference
    from .stream import HeadArtifact

    if args.build_reference:
        print("building models/reference.npz and models/control_head.npz ...", flush=True)
        reference.build(HeadArtifact(), log=lambda m: print(m, flush=True))
    files = [ROOT / name for name in BUNDLE]
    missing = [str(f.relative_to(ROOT)) for f in files if not f.exists()]
    if missing:
        print(f"missing: {', '.join(missing)} (fit the head with scripts/17_fit_final_head.py, "
              "then run with --build-reference, which needs the dataset and the feature cache: "
              "README, Reproduce)", file=sys.stderr)
        return 1
    notices = [ROOT / name for name in BUNDLE_NOTICES]
    missing = [str(f.relative_to(ROOT)) for f in notices if not f.exists()]
    if missing:
        print(f"missing license notices: {', '.join(missing)}", file=sys.stderr)
        return 1
    files += notices
    try:
        version = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip() or "dev"
    except OSError:
        version = "dev"
    out = Path(args.output).expanduser() if args.output else \
        ROOT / "dist" / f"{MODEL_NAME}-{version}.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    def anonymous(info: tarfile.TarInfo) -> tarfile.TarInfo:  # no local user in the archive
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    with tarfile.open(out, "w:gz") as tar:
        for f in files:
            if f.exists():
                tar.add(f, arcname=str(f.relative_to(ROOT)), filter=anonymous)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)\nsha256 {digest}")
    return 0


# -------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cortexlive", description=__doc__.split("\n")[0])
    ap.add_argument("--captures", help="captures directory (default: captures/ in this checkout)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("view", help="open the viewer: drop a video to run it live",
                       description="Open the viewer in the browser: drop a video onto the page "
                       "to run it live, or replay captures.")
    p.add_argument("capture", nargs="?", help="a capture directory to open")
    p.add_argument("--port", type=int, default=8765, help="(default: 8765)")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser")
    p.set_defaults(func=cmd_view)

    p = sub.add_parser("run", help="run the model over a video and save a capture",
                       description="Run the model over a video and save a capture.")
    p.add_argument("video", help="any video file ffmpeg can read")
    p.add_argument("--out", help="capture directory (default: captures/<name>-<time>)")
    p.add_argument("--fast", dest="pace", action="store_const", const="fast",
                   default="realtime", help="run as fast as possible instead of at 1x")
    p.add_argument("--transcript", default="auto",
                   help="auto (the CNeuroMod transcript of a recognised stimulus), none, or "
                   "a CNeuroMod-format .tsv")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("export", help="write a capture as one HTML file",
                       description="Write a capture as one HTML file that opens in a browser "
                       "without a server. The video is not included.")
    p.add_argument("capture", help="a capture directory")
    p.add_argument("-o", "--output", help="output file (default: <capture>/<name>.html)")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("doctor", help="check this machine can run the model",
                       description="Check this machine can run the model.")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("setup", help="download the backbones; install a model bundle",
                       description="Download the three backbones and install a model bundle.")
    p.add_argument("--models", help="model bundle (.tar.gz path or URL) from `cortexlive bundle`")
    p.add_argument("--skip-backbones", action="store_true",
                   help="only install the model bundle")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("bundle", help="pack the trained model and reference data",
                       description="Pack the trained model, its reference data and its license "
                       "notices as a model bundle.")
    p.add_argument("-o", "--output",
                   help=f"output file (default: dist/{MODEL_NAME}-<version>.tar.gz)")
    p.add_argument("--build-reference", action="store_true",
                   help="rebuild models/reference.npz and the control baseline first")
    p.set_defaults(func=cmd_bundle)

    args = ap.parse_args(argv)
    t0 = time.time()
    code = args.func(args)
    if args.cmd in ("run", "bundle") and code == 0:
        print(f"done in {time.time() - t0:.0f}s")
    return code


if __name__ == "__main__":
    sys.exit(main())
