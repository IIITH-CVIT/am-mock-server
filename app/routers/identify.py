import json
from typing import Literal, Optional

import numpy as np
from fastapi import APIRouter, Form, HTTPException

from app.core.config import settings
from app.core.database import get_conn, get_registration, list_vectors_by_kind
from app.core.face_engine import face_engine
from app.schemas.identify import IdentifyResponse

MAX_VECTOR_LENGTH = 4096

router = APIRouter(prefix="/api/v1/identify", tags=["identify"])

def _parse_vector(raw: str, max_length: int = MAX_VECTOR_LENGTH) -> list[float]:
    raw = raw.strip()

    try:
        if raw.startswith("["):
            vec = [float(x) for x in json.loads(raw)]
        else:
            vec = [float(x) for x in raw.split(",") if x.strip()]
    except(json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code = 400, detail = f"invalid vector format: {exc}")
    
    if not vec:
        raise HTTPException(status_code = 400, detail = "vector must not be empty")
    if len(vec) > max_length:
        raise HTTPException(
            status_code = 400,
            detail = f"Vector length {len(vec)} exceeds max allowed {max_length}"
        )
    return vec 

def _normalized_l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return float("inf")
    return float(np.linalg.norm(a / a_norm - b / b_norm))

def _find_best_match(kind: str, query: list[float], vector_type: Optional[str]):
    """Returns (registration, distance) for the closest stored vector; smallest normalized
    L2 distance wins."""
    query_arr = np.array(query, dtype=np.float32)
    with get_conn() as conn:
        candidates = list_vectors_by_kind(conn, kind=kind, dim=len(query), source=vector_type)
        best_row = None
        best_distance = float("inf")
        for row in candidates:
            stored = np.array(json.loads(row["vector"]), dtype=np.float32)
            distance = _normalized_l2_distance(query_arr, stored)
            if distance < best_distance:
                best_distance = distance
                best_row = row
        if best_row is None:
            return None, None
        registration = get_registration(conn, best_row["registration_id"])
        return registration, best_distance

@router.post("/", response_model=IdentifyResponse, summary="Identify")
def identify(
    type: Literal["face", "fingerprint", "id"] = Form(..., description="Identification method"),
    n: int = Form(settings.identify.default_n, description="Number of nearest neighbors"),
    id: Optional[str] = Form(None),
    face_vector: Optional[str] = Form(None),
    fingerprint_vector: Optional[str] = Form(None),
    vector_type: Optional[Literal["registration", "fru", "sau"]] = Form(None),
) -> IdentifyResponse:
    """Unified identification endpoint.

    - type=id: direct registration UUID lookup
    - type=face: face vector search — dimension must match the configured
      `models.embedder_model` (128-dim for sface, 512-dim for auraface)
    - type=fingerprint: fingerprint vector search (no fingerprint capture route exists in this mock,
      so this will only ever return "no match")
    """
    with get_conn() as conn:
        if type == "id":
            if not id:
                return IdentifyResponse(message="id is required for type=id")
            row = get_registration(conn, id)
            if row is None:
                return IdentifyResponse(message="registration not found")
            return IdentifyResponse(
                registration_id=row["id"],
                name=row["full_name"],
                date_of_visit=row["date_of_visit"],
                timeslot=row["timeslot"],
                ticket_category=row["ticket_category"],
                match_type="id",
                distance=0.0,
                confidence=1.0,
                message="match found",
            )

    if type == "face":
        if not face_vector:
            return IdentifyResponse(message="face_vector is required for type=face")
        query = _parse_vector(face_vector)
        expected_dim = face_engine.embedding_dim
        if len(query) != expected_dim:
            raise HTTPException(
                status_code = 400,
                detail = (
                    f"face_vector has {len(query)} dimensions, expected {expected_dim} "
                    f"({face_engine.model_name})"
                ),
            )
        threshold = settings.identify.face_recognition_threshold
        registration, distance = _find_best_match("face", query, vector_type)
    else:
        if not fingerprint_vector:
            return IdentifyResponse(message="fingerprint_vector is required for type=fingerprint")
        query = _parse_vector(fingerprint_vector)
        registration, distance = _find_best_match("fingerprint", query, vector_type)
        threshold = settings.identify.fingerprint_recognition_threshold

    if registration is None or distance is None or distance > threshold:
        return IdentifyResponse(message="no match found")

    return IdentifyResponse(
        registration_id=registration["id"],
        name=registration["full_name"],
        date_of_visit=registration["date_of_visit"],
        timeslot=registration["timeslot"],
        ticket_category=registration["ticket_category"],
        match_type=type,
        distance=distance,
        confidence=max(0.0, 1.0 - distance / 2.0),
        message="match found",
    )
