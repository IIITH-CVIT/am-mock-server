#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

DB_PATH="data/db.sqlite"

if ! command -v sqlite3 >/dev/null 2>&1; then
    echo "sqlite3 not found, installing..."
    sudo apt-get update
    sudo apt-get install -y sqlite3
fi

if [ ! -f "$DB_PATH" ]; then
    echo "No db found at $DB_PATH yet — start the server first (./run.sh)." >&2
    exit 1
fi

sqlite3 "$DB_PATH"
