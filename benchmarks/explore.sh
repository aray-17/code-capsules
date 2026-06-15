#!/usr/bin/env bash
# One command to build the claim-reproduction data and open the evidence explorer.
#
#   bash benchmarks/explore.sh            # regenerate data, serve, open the browser
#   bash benchmarks/explore.sh --serve    # skip regeneration; just serve what is there
#   PORT=9000 bash benchmarks/explore.sh  # custom port (default 8000)
#   OPEN=0   bash benchmarks/explore.sh   # do not auto-open a browser
#   PY=python3.12 bash benchmarks/explore.sh   # pin the interpreter
#
# Serves the REPO ROOT over HTTP (the page fetches ../../benchmarks/...) and opens
# http://localhost:PORT/benchmarks/explorer/ . Press Ctrl-C to stop the server.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-python3}"
PORT="${PORT:-8000}"
OPEN="${OPEN:-1}"
URL="http://localhost:${PORT}/benchmarks/explorer/"

if [[ "${1:-}" != "--serve" ]]; then
  echo "==> generating claim results + eval-run catalog (offline; no Docker / API)..."
  "$PY" benchmarks/verify_criteria.py \
    || echo "    WARNING: a claim did not reproduce (see output above); serving anyway."
  "$PY" benchmarks/build_evidence_index.py
  echo
fi

echo "==> serving the repo root at:"
echo "      ${URL}"
echo "    (open over http, not file://; press Ctrl-C to stop)"

if [[ "$OPEN" == "1" ]]; then
  # Cache-bust the opened URL so a browser that cached an older build re-fetches
  # the document (the page's own data fetches already pass {cache:"no-store"}).
  OPEN_URL="${URL}?v=$(date +%s)"
  ( sleep 1
    if command -v open >/dev/null 2>&1; then open "$OPEN_URL"
    elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$OPEN_URL"
    fi ) >/dev/null 2>&1 &
fi

# Serve with Cache-Control: no-store so a browser reload always shows the latest
# explorer (Python's plain http.server lets browsers heuristically cache it).
exec "$PY" - "$PORT" <<'PYSERVE'
import sys, http.server, socketserver
PORT = int(sys.argv[1])
class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
    httpd.serve_forever()
PYSERVE
