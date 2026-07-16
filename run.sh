#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# ─────────────────────────────────────────────────────────────
# Build (if needed) and launch the mock server as a standalone
# PyInstaller binary. No container runtime required.
# ─────────────────────────────────────────────────────────────

log()  { printf '\033[1;34m[run.sh]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[run.sh] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

VENV=.venv
BIN=dist/mock-server

command -v python3 >/dev/null 2>&1 || die "python3 not found. Install Python 3.13, then re-run."

if [ ! -d "$VENV" ]; then
    log "Creating venv ($VENV)..."
    python3 -m venv "$VENV"
fi

# Rebuild if there's no binary yet, or any source file is newer than it.
if [ ! -x "$BIN" ] || [ -n "$(find app requirements.txt -newer "$BIN" 2>/dev/null)" ]; then
    log "Installing build dependencies..."
    "$VENV/bin/pip" install --quiet --disable-pip-version-check \
        -r requirements.txt -r requirements-dev.txt

    log "Building binary ($BIN)..."
    "$VENV/bin/pyinstaller" --noconfirm --clean --onefile --name mock-server \
        --collect-all cv2 --collect-all onnxruntime --collect-all numpy \
        --add-data "app/static:app/static" \
        app/__main__.py
else
    log "Binary is up to date, skipping rebuild."
fi

mkdir -p data

log "Starting mock-server on http://localhost:8000 ..."
exec "./$BIN"
