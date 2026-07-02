from typing import Literal, Optional

from pydantic import BaseModel


class IdentifyResponse(BaseModel):
    """Mirrors components.schemas.IdentifyResponse in openapi.json exactly."""

    registration_id: Optional[str] = None
    name: Optional[str] = None
    preferred_name: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    date_of_visit: Optional[str] = None
    timeslot: Optional[str] = None
    ticket_category: Optional[str] = None
    distance: Optional[float] = None
    confidence: Optional[float] = None
    match_type: Optional[Literal["face", "fingerprint", "id"]] = None
    message: Optional[str] = None
