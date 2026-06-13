from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class SeniorityLevel(str, Enum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"


class JobRequirements(BaseModel):
    title: str
    seniority: SeniorityLevel
    required_skills: List[str]
    nice_to_have_skills: List[str]
    domain: str
    years_of_experience: Optional[int] = None
    key_responsibilities: List[str]
    soft_skills: List[str]
    summary: str
