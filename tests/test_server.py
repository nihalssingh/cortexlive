"""The viewer server's guards, against a real local server (no model is loaded)."""

from __future__ import annotations

import http.client
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from cortexlive import capture
from cortexlive.run import RunResult, Volume
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
