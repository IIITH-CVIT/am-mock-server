"""Standalone verification tool: compute a face vector from an image and
call POST /api/v1/identify/ with it, exactly like a real edge device would.

Usage (run inside the container, e.g. via identify_image.sh):
    python -m app.cli_identify < photo.jpg
    python -m app.cli_identify photo.jpg
"""

import json
import sys
import urllib.parse
import urllib.request

from app.core.face_engine import face_engine

IDENTIFY_URL = "http://localhost:8000/api/v1/identify/"


def main() -> None:
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            image_bytes = f.read()
    else:
        image_bytes = sys.stdin.buffer.read()

    vector = face_engine.embed(image_bytes)

    data = urllib.parse.urlencode(
        {
            "type": "face",
            "face_vector": json.dumps(vector),
            "vector_type": "registration",
        }
    ).encode()

    request = urllib.request.Request(IDENTIFY_URL, data=data, method="POST")
    with urllib.request.urlopen(request) as response:
        print(json.dumps(json.loads(response.read()), indent=2))


if __name__ == "__main__":
    main()
