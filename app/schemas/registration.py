from typing import Optional

from pydantic import BaseModel


class RegistrationResult(BaseModel):
    registration_id: str
    status: str
    message: str


class VectorOut(BaseModel):
    kind: str
    model: Optional[str]
    dim: int


class RegistrationOut(BaseModel):
    registration_id: str
    full_name: str
    date_of_visit: Optional[str]
    timeslot: Optional[str]
    ticket_category: Optional[str]
    created_at: str
    vectors: list[VectorOut]
