from typing import TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from src.models.job_requirements import JobRequirements
from src.models.candidate_profile import CandidateProfile, ScoredCandidate


class HRGraphState(TypedDict):

    # Initial input
    job_description: str

    # Output of the jd_analyzer node
    requirements: Optional[JobRequirements]

    # Output of the candidate_searcher node
    candidates: Optional[list[CandidateProfile]]

    # Output of the human_approval node (gate)
    approved_candidates: Optional[list[CandidateProfile]]
    human_approved: Optional[bool]         # True = proceed, False = stop

    # Output of the profile_scorer node
    scored_candidates: Optional[list[ScoredCandidate]]

    # Output of the report_generator node
    report_path: Optional[str]
    report_content: Optional[str]

    # Log/debug messages accumulated along the graph
    # Annotated[..., add_messages] = append instead of overwrite
    messages: Annotated[list[BaseMessage], add_messages]

    # Error handling
    error: Optional[str]
