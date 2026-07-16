import uuid
from datetime import date

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.core.database import (
    get_conn,
    get_registration,
    insert_registration,
    insert_vector,
    list_registrations,
)
from app.core.face_engine import NoFaceDetectedError, face_engine
from app.schemas.registration import RegistrationOut, RegistrationResult, VectorOut

router = APIRouter(prefix="/api/v1/registrations", tags=["registrations"])

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024

DEFAULT_TIMESLOT = "10:00"
DEFAULT_TICKET_CATEGORY = "general"

@router.post("/register", response_model=RegistrationResult, summary="Register")
def register(
    full_name: str = Form(..., min_length = 5, max_length = 200),
    image: UploadFile = File(...),
) -> RegistrationResult:
    """Seed a person's registration: name + one face image.

    Detects the face and computes one embedding from the photo — via whichever
    backend `models.embedder_model` selects (sface or auraface, see
    app/core/face_engine.py). Only the name, visit details, and the vector are
    persisted; the image bytes are discarded. Visit details (date, timeslot,
    ticket category) are not client input — the date is stamped as today, and
    timeslot/ticket category are fixed defaults.
    """

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code = 415,
            detail = f"Expected an image file, got content_type = {image.content_type!r}",
        )

    date_of_visit = date.today().isoformat()
    timeslot = DEFAULT_TIMESLOT
    ticket_category = DEFAULT_TICKET_CATEGORY

    image_bytes = image.file.read(MAX_IMAGE_SIZE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code = 413,
            detail = f"Image exceeds max size of {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)}MB",
        )

    try:
        vector = face_engine.embed(image_bytes)
    except NoFaceDetectedError:
        raise HTTPException(status_code=422, detail="no face detected in uploaded image")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    registration_id = str(uuid.uuid4())
    with get_conn() as conn:
        insert_registration(conn, registration_id, full_name, date_of_visit, timeslot, ticket_category)
        insert_vector(
            conn,
            vector_id=str(uuid.uuid4()),
            registration_id=registration_id,
            kind="face",
            model=face_engine.model_name,
            vector=vector,
        )

    return RegistrationResult(
        registration_id=registration_id,
        status="completed",
        message=f"registration created with a {len(vector)}-dim {face_engine.model_name} face embedding",
    )


@router.get("/", response_model=list[RegistrationOut], summary="List Registrations")
def list_all(
    skip: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500)
) -> list[RegistrationOut]:
    with get_conn() as conn:
        rows = list_registrations(conn, skip=skip, limit=limit)
        out = []
        for row in rows:
            vectors = conn.execute(
                "SELECT kind, model, dim FROM vectors WHERE registration_id = ?",
                (row["id"],),
            ).fetchall()
            out.append(
                RegistrationOut(
                    registration_id=row["id"],
                    full_name=row["full_name"],
                    date_of_visit=row["date_of_visit"],
                    timeslot=row["timeslot"],
                    ticket_category=row["ticket_category"],
                    created_at=row["created_at"],
                    vectors=[VectorOut(kind=v["kind"], model=v["model"], dim=v["dim"]) for v in vectors],
                )
            )
        return out


@router.get("/{registration_id}", response_model=RegistrationOut, summary="Get Registration")
def get_one(registration_id: str) -> RegistrationOut:
    with get_conn() as conn:
        row = get_registration(conn, registration_id)
        if row is None:
            raise HTTPException(status_code=404, detail="registration not found")
        vectors = conn.execute(
            "SELECT kind, model, dim FROM vectors WHERE registration_id = ?",
            (registration_id,),
        ).fetchall()
        return RegistrationOut(
            registration_id=row["id"],
            full_name=row["full_name"],
            date_of_visit=row["date_of_visit"],
            timeslot=row["timeslot"],
            ticket_category=row["ticket_category"],
            created_at=row["created_at"],
            vectors=[VectorOut(kind=v["kind"], model=v["model"], dim=v["dim"]) for v in vectors],
        )
