#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# ─────────────────────────────────────────────────────────────
# Launch the mock server under Podman, bootstrapping the toolchain 
# if it's missing. 
# Installs require sudo and network access.
# ─────────────────────────────────────────────────────────────

log()  { printf '\033[1;34m[run.sh]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[run.sh]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[run.sh] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# Echo the host's package-manager install command, or empty if unknown.
pkg_install_cmd() {
    if   command -v dnf     >/dev/null 2>&1; then echo "sudo dnf install -y"
    elif command -v yum     >/dev/null 2>&1; then echo "sudo yum install -y"
    elif command -v apt-get >/dev/null 2>&1; then echo "sudo apt-get update && sudo apt-get install -y"
    elif command -v zypper  >/dev/null 2>&1; then echo "sudo zypper install -y"
    elif command -v pacman  >/dev/null 2>&1; then echo "sudo pacman -S --noconfirm"
    elif command -v brew    >/dev/null 2>&1; then echo "brew install"
    fi
}

install_pkg() {
    local pkg="$1" cmd
    cmd="$(pkg_install_cmd)"
    [ -n "$cmd" ] || return 1
    log "Installing '$pkg'  ->  $cmd $pkg"
    eval "$cmd $pkg"
}

# 1) Podman itself
if command -v podman >/dev/null 2>&1; then
    log "podman present: $(podman --version)"
else
    warn "podman not found. Installing it (this needs sudo)."
    install_pkg podman || die "No supported package manager found. Install 'podman' manually, then re-run."
    command -v podman >/dev/null 2>&1 || die "podman install did not take effect. Open a new shell and re-run."
    log "podman installed: $(podman --version)"
fi

mkdir -p data

IMAGE=mock-server
NAME=mock-server

log "Building image ($IMAGE)..."
podman build -f Containerfile -t "$IMAGE" .

# Remove any previous container with the same name so re-runs don't collide
if podman container exists "$NAME" 2>/dev/null; then
    log "Removing existing container '$NAME' ..."
    podman rm -f "$NAME" >/dev/null
fi

log "Starting container ($NAME) on http://localhost:8000 ..."
exec podman run \
    --name "$NAME" \
    -p 8000:8000 \
    -v "$(pwd)/config.yaml:/app/config.yaml:ro,Z" \
    -v "$(pwd)/models:/app/models:ro,Z" \
    -v "$(pwd)/data:/app/data:Z" \
    -e CONFIG_PATH=/app/config.yaml \
    --restart unless-stopped \
    "$IMAGE"
