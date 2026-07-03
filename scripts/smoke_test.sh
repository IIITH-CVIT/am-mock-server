#!/usr/bin/env bash
set -euo pipefail 
cd "$(dirname "${BASH_SOURCE[0]}")/.."

podman compose up --build -d 
trap 'podman compose logs; podman compose down' EXIT

for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null; then
        echo "OK: server healthy after ${i}s"
        exit 0
    fi 
    sleep 1
done

echo "FAILED: server never became healthy" >&2
exit 1