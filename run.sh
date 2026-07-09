#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# ─────────────────────────────────────────────────────────────
# Launch the mock server under Podman (team standard — not Docker),
# bootstrapping the toolchain if it's missing. Safe to re-run: if
# podman + a compose provider are already present this just builds
# and starts the stack. Installs require sudo and network access.
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

# Return the working compose provider ("podman compose" or "podman-compose"), or non-zero.
detect_compose() {
    if   podman compose version   >/dev/null 2>&1; then echo "podman compose"
    elif command -v podman-compose >/dev/null 2>&1; then echo "podman-compose"
    else return 1
    fi
}

# 1) Podman itself
if command -v podman >/dev/null 2>&1; then
    log "podman present: $(podman --version)"
else
    warn "podman not found — installing it (this needs sudo)."
    install_pkg podman || die "No supported package manager found. Install 'podman' manually, then re-run."
    command -v podman >/dev/null 2>&1 || die "podman install did not take effect. Open a new shell and re-run."
    log "podman installed: $(podman --version)"
fi

# 2) A compose provider for podman (podman-compose). Install it if absent.
if COMPOSE="$(detect_compose)"; then
    log "compose provider: $COMPOSE"
else
    warn "no podman compose provider found — installing podman-compose."
    if ! install_pkg podman-compose; then
        # Distro doesn't package it (or no pkg manager): fall back to pip.
        warn "package manager couldn't provide podman-compose; trying pip."
        command -v pip3 >/dev/null 2>&1 || install_pkg python3-pip || true
        command -v pip3 >/dev/null 2>&1 || die "Need pip3 or a packaged podman-compose. Install 'podman-compose' manually and re-run."
        pip3 install --user podman-compose || die "pip could not install podman-compose. Install it manually and re-run."
    fi
    COMPOSE="$(detect_compose)" || die "podman-compose still not available (a new shell may be needed for PATH changes). Install it manually and re-run."
    log "compose provider: $COMPOSE"
fi

mkdir -p data

log "Building and starting the mock server (http://localhost:8000) ..."
exec $COMPOSE up --build
