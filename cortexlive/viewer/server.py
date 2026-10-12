"""Local HTTP server for the viewer: captures, media, and live runs.

    GET  /                               the viewer
    GET  /api/atlas                      cortical surface views + networks
    GET  /api/captures                   saved captures, newest first
    GET  /api/captures/<id>              one capture, ready to draw
    GET  /api/captures/<id>/media        its video (byte ranges, for seeking)
    POST /api/live                       body = a video file; runs it live as a new capture
    POST /api/captures/<id>/rerun        runs a capture's video live again
    POST /api/live/stop                  stops the running session
    GET  /api/live/<session>/events      server-sent events for a session

It listens on 127.0.0.1 only, and answers only requests addressed to that host and,
for POSTs, sent from its own page: a web page elsewhere cannot read captures through a
rebound DNS name or start and stop runs.
"""

from __future__ import annotations

import json
import mimetypes
import re
import shutil
import signal
import threading
import time
import traceback
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import numpy as np

from .. import capture, reference
from ..data import TR_SECONDS
from ..run import RunError
from ..stream import ModelBusy
from . import html, payload

VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".mpg", ".mpeg", ".ts",
              ".mts", ".m2ts", ".wmv", ".flv", ".ogv", ".3gp"}
DISK_RESERVE = 2 << 30  # an upload must leave this much disk free


class Busy(RuntimeError):
    """The previous live run has not stopped yet."""


class Session:
    def __init__(self, sid: int, out: Path, video: Path, name: str):
        self.id, self.out, self.video, self.name = sid, out, video, name
        self.events: list[str] = []
        self.cond = threading.Condition()
        self.done = False
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def push(self, event: str, data: dict) -> None:
        with self.cond:
            n = len(self.events)
            self.events.append(f"id: {n}\nevent: {event}\ndata: {json.dumps(data)}\n\n")
            self.cond.notify_all()

    def finish(self) -> None:
        with self.cond:
            self.done = True
            self.cond.notify_all()


class Live:
    """One live session at a time; the models load on first use and stay loaded."""

    def __init__(self, root: Path):
        self.root = root
        self.enc = None
        self.lock = threading.Lock()  # guards session bookkeeping
        self.running = threading.Lock()  # held for a whole run: one run uses the encoder
        self.sessions: dict[int, Session] = {}
        self.current: Session | None = None
        self.next_id = 1

    def start(self, video: Path, name: str, out: Path) -> Session:
        with self.lock:
            cur = self.current
            if cur and cur.thread and cur.thread.is_alive():
                cur.stop.set()
                cur.thread.join(timeout=30)
                if cur.thread.is_alive():
                    raise Busy("the previous run is still stopping; try again in a moment")
            for s in self.sessions.values():  # finished sessions: their pages have moved on
                if s.done:
                    s.events = []
            sess = Session(self.next_id, out, video, name)
            self.next_id += 1
            self.sessions[sess.id] = sess
            self.current = sess
            sess.thread = threading.Thread(target=self._run, args=(sess,), daemon=True)
            sess.thread.start()
            return sess

    def stop(self, wait: float = 0.0) -> None:
        """Stop the running session; with `wait`, give it that long to save its capture."""
        cur = self.current
        if cur:
            cur.stop.set()
            if wait and cur.thread and cur.thread.is_alive():
                cur.thread.join(timeout=wait)

    def busy(self) -> bool:
        cur = self.current
        return bool(cur and cur.thread and cur.thread.is_alive())

    def discard_unsaved(self) -> None:
        """Remove a run that will not finish saving, so no half-made capture is left."""
        cur = self.current
        if cur and not (cur.out / "capture.json").exists():
            shutil.rmtree(cur.out, ignore_errors=True)

    def _run(self, sess: Session) -> None:
        with self.running:
            self._run_locked(sess)

    def _run_locked(self, sess: Session) -> None:
        from ..record import record
        from ..stream import LiveEncoder

        log_file = (sess.out / "run.log").open("a")

        def log(msg: str) -> None:
            log_file.write(msg + "\n")
            log_file.flush()
            sess.push("status", {"text": msg})

        captured = failed = False
        try:
            if self.enc is None:
                log("loading the models; the video starts from the beginning when they are ready")
                self.enc = LiveEncoder()
            enc = self.enc
            if sess.stop.is_set():
                return
            ref = reference.load()

            def on_start(duration):
                subs = [s for s in enc.head.subjects if ref and s in ref["mask"]]
                masks = {s: payload.b64(ref["mask"][s].astype(np.uint8)) for s in subs}
                # each parcel's mean and SD over Friends season 6: the page colours the first
                # volumes against these until the run has statistics of its own
                stats = {s: {"mean": payload.f16(ref["mean"][s]), "sd": payload.f16(ref["sd"][s])}
                         for s in subs}
                sess.push("start", {"duration": duration, "capture": sess.out.name,
                                    "subjects": list(enc.head.subjects), "masks": masks,
                                    "stats": stats, "tr": TR_SECONDS, "trim": list(payload.TRIM)})

            def on_volume(v):
                sess.push("volume", {
                    "s": v.s, "lag": round(v.lag_s, 3), "tick": round(v.tick_ms, 1),
                    "preds": {k: payload.b64(p.astype(np.float16)) for k, p in v.preds.items()},
                })

            meta = record(enc, sess.video, sess.out, input_name=sess.name, on_start=on_start,
                          on_volume=on_volume, stop=sess.stop, log=log)
            # volumes come in order from 0: keep the run once it reached the first one shown
            captured = meta["run"]["volumes"] > payload.TRIM[0]
            enc.reset()  # release the run's feature buffers
            enc.release_memory()
        except Exception as e:  # surfaced to the page
            failed = True
            expected = isinstance(e, (RunError, ModelBusy))
            if not expected:
                traceback.print_exc()
            msg = str(e) if expected else f"{type(e).__name__}: {e}"
            sess.push("failed", {"message": msg.replace(str(sess.video), sess.name)})
        finally:
            log_file.close()
            if not captured:
                shutil.rmtree(sess.out, ignore_errors=True)  # nothing worth keeping
            if not failed:
                sess.push("end", {"capture": sess.out.name if captured else None,
                                  "stopped": sess.stop.is_set()})
            sess.finish()


def make_handler(root: Path, live: Live, extra: dict[str, Path], port: int):
    page_cache: dict[str, bytes] = {}
    payload_cache: OrderedDict[str, tuple[tuple, bytes]] = OrderedDict()
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    origins = {f"http://{h}" for h in hosts}

    def capture_dir(cid: str) -> Path | None:
        if cid in extra:
            return extra[cid]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", cid) or ".." in cid:
            return None
        p = root / cid
        return p if (p / "capture.json").exists() else None

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            # no other page may frame the viewer (clickjacking)
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _allowed(self, post: bool = False) -> bool:
            if self.headers.get("Host", "") not in hosts:
                return False
            origin = self.headers.get("Origin")
            return not post or origin is None or origin in origins

        def do_GET(self):
            if not self._allowed():
                self.close_connection = True
                return self._json({"error": "forbidden"}, 403)
            path = unquote(urlparse(self.path).path)
            if path in ("/", "/index.html"):
                if "page" not in page_cache:
                    page_cache["page"] = html().encode()
                return self._send(200, page_cache["page"], "text/html; charset=utf-8")
            if path == "/api/atlas":
                if "atlas" not in page_cache:
                    page_cache["atlas"] = json.dumps(payload.atlas()).encode()
                return self._send(200, page_cache["atlas"], "application/json")
            if path == "/api/captures":
                return self._json(payload.capture_list(root, extra))
            m = re.fullmatch(r"/api/captures/([^/]+)(/media)?", path)
            if m:
                d = capture_dir(m.group(1))
                if d is None:
                    return self._json({"error": "no such capture"}, 404)
                meta = capture.load_meta(d)
                media = capture.media_path(d, meta)
                if m.group(2):
                    if media is None:
                        return self._json({"error": "video not found"}, 404)
                    return self._file(media)
                key = ((d / "capture.json").stat().st_mtime, media is not None)
                hit = payload_cache.get(str(d))
                if hit is None or hit[0] != key:
                    # the id the page uses is the one this server knows it by (ext-1-... for a
                    # capture opened from outside the captures folder), not the folder's name
                    hit = (key, json.dumps({**payload.capture_payload(d), "id": m.group(1)})
                           .encode())
                    payload_cache[str(d)] = hit
                    while len(payload_cache) > 4:
                        payload_cache.popitem(last=False)
                payload_cache.move_to_end(str(d))
                return self._send(200, hit[1], "application/json")
            m = re.fullmatch(r"/api/live/(\d+)/events", path)
            if m:
                sess = live.sessions.get(int(m.group(1)))
                if sess is None:
                    return self._json({"error": "no such session"}, 404)
                return self._events(sess)
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            # Every POST ends its connection: a refused POST's unread body must never be read
            # as a further request, which would carry no Origin and pass the check below.
            self.close_connection = True
            if not self._allowed(post=True):
                return self._json({"error": "forbidden"}, 403)
            path = unquote(urlparse(self.path).path)
            if path == "/api/live":
                return self._upload()
            m = re.fullmatch(r"/api/captures/([^/]+)/rerun", path)
            if m:
                d = capture_dir(m.group(1))
                if d is None:
                    return self._json({"error": "no such capture"}, 404)
                meta = capture.load_meta(d)
                media = capture.media_path(d, meta)
                if media is None:
                    return self._json({"error": "the video for this capture is gone"}, 404)
                out = capture.new_dir(root, meta["input"]["name"])
                return self._start(media, meta["input"]["name"], out)
            if path == "/api/live/stop":
                live.stop()
                return self._json({"ok": True})
            return self._json({"error": "not found"}, 404)

        def _upload(self) -> None:
            name = Path(unquote(self.headers.get("X-Filename", "video.mp4"))).name
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length <= 0:
                self.close_connection = True
                return self._json({"error": "empty upload"}, 400)
            root.mkdir(parents=True, exist_ok=True)
            if length > shutil.disk_usage(root).free - DISK_RESERVE:
                self.close_connection = True
                return self._json({"error": "not enough disk space for this video"}, 413)
            ext = Path(name).suffix.lower()
            out = capture.new_dir(root, name)
            video = out / f"input{ext if ext in VIDEO_EXTS else '.video'}"
            left = length
            try:
                with video.open("wb") as f:
                    while left > 0:
                        chunk = self.rfile.read(min(1 << 20, left))
                        if not chunk:
                            break
                        f.write(chunk)
                        left -= len(chunk)
            except OSError:  # the connection was reset or the disk filled: keep nothing
                shutil.rmtree(out, ignore_errors=True)
                return
            if left > 0:  # the page went away mid-upload
                shutil.rmtree(out, ignore_errors=True)
                self.close_connection = True
                return self._json({"error": "upload incomplete"}, 400)
            return self._start(video, name, out)

        def _start(self, video: Path, name: str, out: Path) -> None:
            try:
                sess = live.start(video, name, out)
            except Busy as e:
                shutil.rmtree(out, ignore_errors=True)
                return self._json({"error": str(e)}, 409)
            return self._json({"session": sess.id, "capture": out.name})

        def _events(self, sess: Session) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            last = self.headers.get("Last-Event-ID", "")
            i = int(last) + 1 if last.isdigit() else 0  # a reconnect resumes, never replays
            try:
                while True:
                    with sess.cond:
                        while i >= len(sess.events) and not sess.done:
                            sess.cond.wait(timeout=10.0)
                        batch = sess.events[i:]
                        done = sess.done
                    for ev in batch:
                        self.wfile.write(ev.encode())
                    i += len(batch)
                    self.wfile.flush()
                    if done and i >= len(sess.events):
                        return
            except (BrokenPipeError, ConnectionResetError):
                return

        def _file(self, path: Path) -> None:
            # Only a video is served. A capture names its video (capture.media_path), and a
            # capture made elsewhere could name any file, or link to one as input.mp4: the
            # type comes from the file the path resolves to.
            real = path.resolve()
            own = real.name == "input.video"  # how _upload saves a video of an unlisted type
            ctype = ("application/octet-stream" if own
                     else "video/x-matroska" if real.suffix.lower() == ".mkv"
                     else mimetypes.guess_type(real.name.lower())[0] or "")
            if not (own or ctype.startswith("video/")) or not real.is_file():
                return self._json({"error": "video not found"}, 404)
            size = path.stat().st_size
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng or "")
            partial = bool(m and (m.group(1) or m.group(2)))
            if partial:
                if m.group(1):
                    start = int(m.group(1))
                    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            self.send_response(206 if partial else 200)
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Type", ctype)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            try:
                with path.open("rb") as f:
                    f.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = f.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def serve(root: Path, port: int = 8765, open_capture: Path | None = None,
          browser: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    extra: dict[str, Path] = {}
    target = ""
    if open_capture is not None:
        open_capture = open_capture.resolve()
        if open_capture.parent == root.resolve():
            target = open_capture.name
        else:
            cid = f"ext-1-{open_capture.name}"
            extra[cid] = open_capture
            target = cid
    live = Live(root)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(root, live, extra, port))
    url = f"http://localhost:{port}/" + (f"#capture={quote(target)}" if target else "")
    print(f"cortexlive viewer: {url}\n  drop a video onto the page to run it live; "
          "ctrl-c here to quit", flush=True)
    if browser:
        import webbrowser

        threading.Thread(target=lambda: (time.sleep(0.4), webbrowser.open(url)),
                         daemon=True).start()
    def hang_up(*_):  # the terminal window was closed: stop as ctrl-c does
        signal.signal(signal.SIGHUP, signal.SIG_IGN)  # a closing window sends more than one
        raise KeyboardInterrupt

    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:  # respect nohup
        signal.signal(signal.SIGHUP, hang_up)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if live.busy():
            try:
                print("\nstopping the live run and saving it (ctrl-c again to discard it) ...",
                      flush=True)
            except OSError:  # no terminal left to print to
                pass
        try:
            live.stop(wait=30)
        except KeyboardInterrupt:  # interrupted again: give up on the run
            live.discard_unsaved()
        server.server_close()
