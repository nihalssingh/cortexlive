"""The cortexlive viewer: a local web app for live runs and saved captures."""

from __future__ import annotations

import json
from pathlib import Path

PAGE = Path(__file__).resolve().parent / "index.html"
_HEAD = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
         '<meta name="viewport" content="width=device-width, initial-scale=1">\n')


def html(embed: dict | None = None) -> str:
    """The viewer page; `embed` makes it a static page that needs no server.

    index.html holds the page's styles, markup and script; this adds the document
    skeleton around them.
    """
    page = PAGE.read_text()
    if embed is not None:
        data = json.dumps(embed, separators=(",", ":")).replace("<", "\\u003c")
        page = page.replace("<!--EMBED-->", f"<script>window.CORTEXLIVE_EMBED = {data};</script>")
    body = page.index('<div class="app">')
    return _HEAD + page[:body] + "</head>\n<body>\n" + page[body:] + "</body>\n</html>\n"
