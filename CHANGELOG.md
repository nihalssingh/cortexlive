# Changelog

All notable changes to cortexlive. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.1] - 2026-10-12

Security fixes for the viewer, faster live runs and a checked model install. The model is
unchanged: `cortexlive setup` installs `Llama-cortexlive-v0.1.0.tar.gz` from the
[v0.1.0 release](https://github.com/nihalssingh/cortexlive/releases/tag/v0.1.0), as before. To
update a clone, run `git pull`, then `uv sync --extra live`; the installed model keeps working.

### Security

- The viewer closes the connection after every POST, so the body of a refused POST can no
  longer be read as a second request that skips the Origin check.
- The viewer page escapes the fields it shows from `capture.json`, the media route serves
  only video files, and no other site can frame the viewer.

### Changed

- Live runs are faster: V-JEPA 2 skips its predictor (`skip_predictor=True`), which was
  computed but never used, and the features are unchanged. On Friends s06e01a the mean lag
  falls from 0.90 s to 0.48 s, and `--fast` runs at 2.28x real time instead of 1.45x
  (`runs/live_demo/`). The demo captures and the README's figures and images are
  re-measured.
- `cortexlive setup` checks the SHA-256 of a published model file before installing it, and
  a cut-short, corrupt or incomplete file installs nothing.
- The `live` extra needs transformers 5.5 or later. Versions up to 5.3 return a post-norm
  last V-JEPA 2 hidden state, which the truncated encoder would read instead of the features
  the model was fit on.

### Fixed

- A video whose name starts with `.` or `_`, or contains `..`, gets a capture the viewer can
  open.
- An upload cut off part-way leaves no partial capture behind.
- A capture opened from outside `captures/` keeps its id in the page, and capture links are
  URL-encoded.

### Added

- README: how video frames are prepared for V-JEPA 2, and that a recognised file is aligned
  with the recording from its first frame.

## [0.1.0] - 2026-10-11

First public release, with the trained model `Llama-cortexlive-v0.1.0.tar.gz` on the
[v0.1.0 release](https://github.com/nihalssingh/cortexlive/releases/tag/v0.1.0).

[0.1.1]: https://github.com/nihalssingh/cortexlive/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/nihalssingh/cortexlive/releases/tag/v0.1.0
