from typing import TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from src.models.job_requirements import JobRequirements
from src.models.candidate_profile import CandidateProfile, ScoredCandidate


class HRGraphState(TypedDict):

    # Input iniziale
    job_description: str

    # Output del nodo jd_analyzer
    requirements: Optional[JobRequirements]

    # Output del nodo candidate_searcher
    candidates: Optional[list[CandidateProfile]]

    # Output del nodo human_approval (gate)
    approved_candidates: Optional[list[CandidateProfile]]
    human_approved: Optional[bool]         # True = procedi, False = stop

    # Output del nodo profile_scorer
    scored_candidates: Optional[list[ScoredCandidate]]

    # Output del nodo report_generator
    report_path: Optional[str]
    report_content: Optional[str]

    # Messaggi di log/debug accumulati lungo il grafo
    # Annotated[..., add_messages] = append instead of overwrite
    messages: Annotated[list[BaseMessage], add_messages]

    # Error handling
    error: Optional[str]
