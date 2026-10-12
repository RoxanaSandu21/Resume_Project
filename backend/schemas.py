from pydantic import BaseModel
from typing import Optional, Dict, Any

class JobCreate(BaseModel):
    title: str
    description: str

class JobResponse(BaseModel):
    id: int
    title: str
    description: str

    class Config:
        from_attributes = True

class EvaluationResponse(BaseModel):
    candidate_id: int
    full_name: Optional[str]
    email: Optional[str]
    match_score: float
    bias_report: Dict[Any, Any]

    class Config:
        from_attributes = True