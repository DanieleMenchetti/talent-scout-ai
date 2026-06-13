from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from src.models.state import HRGraphState
from src.models.candidate_profile import CandidateProfile, ScoredCandidate
from src.models.job_requirements import JobRequirements


# Pydantic schema used with with_structured_output for scoring
class CandidateScore(BaseModel):
    overall_score: float = Field(ge=0.0, le=10.0)
    skills_match_score: float = Field(ge=0.0, le=10.0)
    experience_score: float = Field(ge=0.0, le=10.0)
    domain_fit_score: float = Field(ge=0.0, le=10.0)
    strengths: list[str]
    gaps: list[str]
    recommendation: str


def profile_scorer_node(state: HRGraphState) -> dict:
    print("\n" + "─" * 60)
    print("  📊 Node: Profile Scorer")
    print("─" * 60)

    candidates: list[CandidateProfile] = state["approved_candidates"]
    requirements: JobRequirements = state["requirements"]

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )
    scoring_llm = llm.with_structured_output(CandidateScore)

    scored: list[ScoredCandidate] = []

    for i, candidate in enumerate(candidates, 1):
        print(f"  → Evaluating {i}/{len(candidates)}: {candidate.name}")

        prompt = f"""You are a senior technical recruiter. Please evaluate this candidate objectively.

        ROLE REQUIREMENTS:
        - Title: {requirements.title} ({requirements.seniority.value})
        - Required Skills: {', '.join(requirements.required_skills)}
        - Optional Skills: {', '.join(requirements.nice_to_have_skills)}
        - Domain: {requirements.domain}
        - Required Years of Experience: {requirements.years_of_experience or 'unspecified'}
        - Responsibilities: {', '.join(requirements.key_responsibilities[:3])}

        CANDIDATE PROFILE:
        - Name: {candidate.name}
        - Current role: {candidate.current_role} @ {candidate.current_company or '?'}
        - Skills: {', '.join(candidate.skills)}
        - Years of experience: {candidate.years_of_experience or '?'}
        - Projects: {', '.join(candidate.notable_projects[:2]) if candidate.notable_projects else 'N/D'}
        - Education: {candidate.education or 'N/D'}
        - Profile: {candidate.raw_profile_text[:300]}

        Scoring rubric (0-10):
        - skills_match_score: How many required skills do you have? 10 = all + nice-to-have, 5 = half, 0 = none
        - experience_score: Correct experience? 10 = perfect, 5 = 2 years difference, 1 = completely wrong
        - domain_fit_score: Background in the right domain? 10 = exact, 5 = adjacent, 1 = completely different
        - overall_score: Weighted average (skills 50%, experience 30%, domain 20%) + your rating

        Be critical and objective. Not all candidates deserve high scores.
        """

        try:
            score: CandidateScore = scoring_llm.invoke(prompt)
            scored.append(ScoredCandidate(
                profile=candidate,
                overall_score=score.overall_score,
                skills_match_score=score.skills_match_score,
                experience_score=score.experience_score,
                domain_fit_score=score.domain_fit_score,
                strengths=score.strengths,
                gaps=score.gaps,
                recommendation=score.recommendation,
            ))
        except Exception as e:
            print(f"     ⚠️  Errore scoring {candidate.name}: {e}")
            scored.append(ScoredCandidate(
                profile=candidate,
                overall_score=0.0,
                skills_match_score=0.0,
                experience_score=0.0,
                domain_fit_score=0.0,
                strengths=[],
                gaps=["Errore durante la valutazione"],
                recommendation="Valutazione fallita.",
            ))

    # Sort by score and assign rank
    scored.sort(key=lambda x: x.overall_score, reverse=True)
    for rank, sc in enumerate(scored, 1):
        sc.rank = rank

    print("\n  Final ranking:")
    for sc in scored:
        bar = "█" * int(sc.overall_score) + "░" * (10 - int(sc.overall_score))
        print(f"    #{sc.rank} {sc.profile.name:<25} [{bar}] {sc.overall_score:.1f}/10")

    return {
        "scored_candidates": scored,
        "messages": [AIMessage(content=f"{len(scored)} candidates evaluated. Top: {scored[0].profile.name} ({scored[0].overall_score:.1f}/10)")],
    }
