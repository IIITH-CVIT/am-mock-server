import uuid
from datetime import date, time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

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


@router.post("/register", response_model=RegistrationResult, summary="Register")
async def register(
    full_name: str = Form(...),
    date_of_visit: str = Form(..., description="ISO date, e.g. 2026-07-01"),
    timeslot: str = Form(..., description="ISO time, e.g. 10:00 or 10:00:00"),
    ticket_category: str = Form(...),
    image: UploadFile = File(...),
) -> RegistrationResult:
    """Seed a person's registration: name + visit details + one face image.

    Detects the face and computes an embedding; only the name, visit details,
    and vector are persisted (the image bytes are discarded after embedding).
    """
    try:
        date.fromisoformat(date_of_visit)
        time.fromisoformat(timeslot)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid date/time format: {exc}")

    image_bytes = await image.read()
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
        message="registration created",
    )


@router.get("/", response_model=list[RegistrationOut], summary="List Registrations")
def list_all(skip: int = 0, limit: int = 100) -> list[RegistrationOut]:
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
