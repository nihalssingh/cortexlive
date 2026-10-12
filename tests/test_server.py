"""The viewer server's guards, against a real local server (no model is loaded)."""

from __future__ import annotations

import http.client
import json
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from cortexlive import capture
from cortexlive.run import RunResult, Volume
from cortexlive.viewer import html
from cortexlive.viewer.server import Live, make_handler


class Head:
    subjects = ["sub-01"]
    layers = {"text": 8, "audio": 15, "video": 19}
    window = 5


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "captures"
    out = root / "clip-1"
    out.mkdir(parents=True)
    vols = [Volume(s=i, lag_s=0.5, tick_ms=100.0, wall_s=1.49 * i,
                   preds={"sub-01": np.full(1000, i, np.float32)}) for i in range(8)]
    capture.save(out, RunResult(volumes=vols, media_s=12.0, wall_s=13.0, pace="realtime"),
                 input_path=None, input_name="clip.mp4", device="cpu", transcript="none",
                 head=Head)
    (out / "input.mp4").write_bytes(bytes(range(256)) * 4)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = srv.server_address[1]
    srv.RequestHandlerClass = make_handler(root, Live(root), {}, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield root, port
    srv.shutdown()
    srv.server_close()


def request(port, method, path, headers=None, body=None, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.putrequest(method, path, skip_host=True)
    conn.putheader("Host", host or f"127.0.0.1:{port}")
    for k, v in (headers or {}).items():
        conn.putheader(k, v)
    conn.endheaders(body)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp, data


def test_page_and_capture_list(server):
    root, port = server
    resp, body = request(port, "GET", "/")
    assert resp.status == 200 and body.startswith(b"<!doctype html>")
    assert resp.getheader("X-Frame-Options") == "DENY"  # no other page may frame it
    assert resp.getheader("Content-Security-Policy") == "frame-ancestors 'none'"
    resp, body = request(port, "GET", "/api/captures")
    assert [c["id"] for c in json.loads(body)] == ["clip-1"]
    resp, body = request(port, "GET", "/api/captures/clip-1")
    assert resp.status == 200 and json.loads(body)["n_vol"] == 9


def test_foreign_host_and_origin_are_refused(server):
    root, port = server
    assert request(port, "GET", "/api/captures", host=f"evil.example:{port}")[0].status == 403
    resp, _ = request(port, "POST", "/api/live/stop", {"Origin": "http://evil.example"})
    assert resp.status == 403
    resp, _ = request(port, "POST", "/api/live/stop", {"Origin": f"http://localhost:{port}"})
    assert resp.status == 200


def test_capture_ids_cannot_leave_the_root(server):
    root, port = server
    assert request(port, "GET", "/api/captures/..")[0].status == 404
    assert request(port, "GET", "/api/captures/%2E%2E/media")[0].status == 404


def test_media_ranges(server):
    root, port = server
    resp, body = request(port, "GET", "/api/captures/clip-1/media", {"Range": "bytes=0-99"})
    assert resp.status == 206 and len(body) == 100 and body[:3] == bytes([0, 1, 2])
    assert resp.getheader("Content-Range") == "bytes 0-99/1024"
    resp, _ = request(port, "GET", "/api/captures/clip-1/media", {"Range": "bytes=5000-"})
    assert resp.status == 416


def test_bad_uploads_leave_nothing_behind(server):
    root, port = server
    resp, _ = request(port, "POST", "/api/live", {"Content-Length": "abc"})
    assert resp.status == 400
    conn = socket.create_connection(("127.0.0.1", port), timeout=10)
    conn.sendall(f"POST /api/live HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nX-Filename: a.mp4\r\n"
                 "Content-Length: 100\r\n\r\n".encode() + b"x" * 10)
    conn.shutdown(socket.SHUT_WR)  # the page went away mid-upload
    reply = conn.recv(4096).decode()
    conn.close()
    assert reply.startswith("HTTP/1.1 400")
    assert sorted(p.name for p in root.iterdir()) == ["clip-1"]


def test_an_upload_cut_off_by_a_reset_leaves_nothing_behind(server):
    root, port = server
    conn = socket.create_connection(("127.0.0.1", port), timeout=10)
    conn.sendall(f"POST /api/live HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nX-Filename: a.mp4\r\n"
                 "Content-Length: 10000000\r\n\r\n".encode() + b"x" * 1_000_000)
    time.sleep(0.3)  # the server is writing the upload
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    conn.close()  # a reset, as when the tab closes mid-upload
    deadline = time.time() + 5
    while time.time() < deadline and len(list(root.iterdir())) > 1:
        time.sleep(0.05)
    assert sorted(p.name for p in root.iterdir()) == ["clip-1"]


def _received(conn: socket.socket) -> bytes:
    """Everything the server sends until it closes the connection, or 3 s of silence."""
    data = b""
    conn.settimeout(3)
    try:
        while chunk := conn.recv(65536):
            data += chunk
    except (TimeoutError, ConnectionResetError):
        pass
    return data


def test_a_refused_post_body_is_never_read_as_a_request(server):
    root, port = server
    # A page elsewhere can send any bytes as a POST body, here a whole request without an
    # Origin, which would pass the check if the server read it as the next request.
    smuggled = f"GET /api/captures HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode()
    with socket.create_connection(("127.0.0.1", port), timeout=10) as conn:
        conn.sendall(f"POST /api/live/stop HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                     f"Origin: http://evil.example\r\nContent-Length: {len(smuggled)}\r\n\r\n"
                     .encode() + smuggled)
        reply = _received(conn)
    assert reply.startswith(b"HTTP/1.1 403") and reply.count(b"HTTP/1.1 ") == 1


def test_page_escapes_what_a_capture_says():
    page = html()  # a capture made elsewhere could hold markup in any of these fields
    for raw in ('data-s="${s}"', '">${s.replace(', "${r.volumes}", "${r.expected_volumes}",
                "${v.parcels}", "${S.subj.replace("):
        assert raw not in page


def _capture(out, input_path=None):
    out.mkdir(parents=True, exist_ok=True)
    vols = [Volume(s=i, lag_s=0.5, tick_ms=1.0, wall_s=1.49 * i,
                   preds={"sub-01": np.zeros(1000, np.float32)}) for i in range(8)]
    capture.save(out, RunResult(volumes=vols, media_s=12.0, wall_s=13.0, pace="fast"),
                 input_path=input_path, input_name="clip.mp4", device="cpu", transcript="none",
                 head=Head)


def test_media_is_only_ever_a_video(server, tmp_path):
    root, port = server
    secret = tmp_path / "secret.txt"
    secret.write_text("not a video")
    _capture(root / "by-path", input_path=secret)  # its capture.json names another file
    for name in ("input.mp4", "input.video"):  # or it links one in as its upload
        _capture(root / f"by-link{name}")
        (root / f"by-link{name}" / name).symlink_to(secret)
    for cid in ("by-path", "by-linkinput.mp4", "by-linkinput.video"):
        assert request(port, "GET", f"/api/captures/{cid}/media")[0].status == 404
    assert request(port, "GET", "/api/captures/clip-1/media")[0].status == 200
    _capture(root / "other-type")  # an upload of a type outside VIDEO_EXTS is kept as input.video
    (root / "other-type" / "input.video").write_bytes(b"\0" * 64)
    assert request(port, "GET", "/api/captures/other-type/media")[0].status == 200


def test_a_capture_opened_from_elsewhere_keeps_its_id(tmp_path):
    root, outside = tmp_path / "captures", tmp_path / "elsewhere" / "demo"
    root.mkdir()
    _capture(outside)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = srv.server_address[1]
    srv.RequestHandlerClass = make_handler(root, Live(root), {"ext-1-demo": outside}, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:  # the page reopens, reruns and links to a capture by the id it is given here
        resp, body = request(port, "GET", "/api/captures/ext-1-demo")
        assert resp.status == 200 and json.loads(body)["id"] == "ext-1-demo"
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.mark.parametrize("name", ["_clip.mp4", ".hidden.mov", "a..b.mp4", "Wait....mp4"])
def test_every_new_capture_can_be_opened(server, name):
    root, port = server
    d = capture.new_dir(root, name)
    _capture(d)
    assert request(port, "GET", f"/api/captures/{d.name}")[0].status == 200
