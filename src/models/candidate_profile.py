from pydantic import BaseModel, Field
from typing import List, Optional


class CandidateProfile(BaseModel):
    name: str
    current_role: str
    current_company: Optional[str] = None
    years_of_experience: Optional[int] = None
    skills: List[str] = Field(default_factory=list)
    notable_projects: List[str] = Field(default_factory=list)
    education: Optional[str] = None
    source_url: Optional[str] = None
    raw_profile_text: str = ""


class ScoredCandidate(BaseModel):
    profile: CandidateProfile
    overall_score: float = Field(ge=0.0, le=10.0)
    skills_match_score: float = Field(ge=0.0, le=10.0)
    experience_score: float = Field(ge=0.0, le=10.0)
    domain_fit_score: float = Field(ge=0.0, le=10.0)
    strengths: List[str]
    gaps: List[str]
    recommendation: str
    rank: Optional[int] = None
