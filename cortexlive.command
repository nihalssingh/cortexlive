#!/bin/bash
# Double-click in Finder to open the cortexlive viewer: it starts the viewer on this Mac and
# opens its page in your browser. Drop a video onto the page to run it live.
# Close this window, or press ctrl-c in it, to stop.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/cortexlive ]; then
  echo "cortexlive is not installed in this folder yet: follow Install in README.md."
  exit 1
fi

mine() {  # a cortexlive viewer started by this user is listening on port $1
  lsof -nP -a -u "$(id -u)" -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1 &&
    curl -s -m 1 "http://localhost:$1/" | grep -q 'class="brand">cortexlive'
}
free() {  # nothing is listening on port $1
  .venv/bin/python -c 'import socket, sys
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", int(sys.argv[1])))' "$1" 2>/dev/null
}

ports=$(seq 8765 8775)
for port in $ports; do
  if mine "$port"; then
    echo "The viewer is already running: opening http://localhost:$port/"
    open "http://localhost:$port/"
    exit 0
  fi
done
for port in $ports; do
  if free "$port"; then
    exec .venv/bin/cortexlive view --port "$port"
  fi
done
echo "Ports 8765 to 8775 are all in use: close something and try again."
exit 1
