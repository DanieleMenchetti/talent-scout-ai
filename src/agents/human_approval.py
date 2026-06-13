from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, AIMessage

from src.models.state import HRGraphState
from src.models.candidate_profile import CandidateProfile


def human_approval_node(state: HRGraphState) -> dict:
    print("\n" + "═" * 60)
    print("  Node: Human Approval Gate")
    print("═" * 60)

    candidates: list[CandidateProfile] = state["candidates"]

    if not candidates:
        return {
            "human_approved": False,
            "approved_candidates": [],
            "messages": [AIMessage(content="No candidate to approve.")],
        }

    # Build the summary message for the user
    summary_lines = ["Candidates found — approve before proceeding with scoring:\n"]
    for i, c in enumerate(candidates, 1):
        skills_preview = ", ".join(c.skills[:3]) if c.skills else "N/D"
        summary_lines.append(
            f"[{i}] {c.name}\n"
            f"    Role: {c.current_role} @ {c.current_company or '?'}\n"
            f"    Skills: {skills_preview}\n"
            f"    Exp: {c.years_of_experience or '?'} years"
        )

    summary_lines.append(
        "\nOptions:\n"
        "  'all'     → approve all\n"
        "  '1,2,3'   → approve specifics (comma-separated numbers)\n"
        "  'reject'  → reject everyone and terminate"
    )

    summary = "\n".join(summary_lines)
    user_input: str = interrupt(summary)

    print(f"\nInput received: '{user_input}'")
    approved = _parse_approval(user_input.strip().lower(), candidates)

    if not approved:
        print("All candidates rejected.")
        return {
            "human_approved": False,
            "approved_candidates": [],
            "messages": [HumanMessage(content=f"Approval input: {user_input}"),
                         AIMessage(content="Pipeline broken: Candidates rejected by user.")],
        }

    names = ", ".join(c.name for c in approved)
    print(f" Approved: {names}")

    return {
        "human_approved": True,
        "approved_candidates": approved,
        "messages": [
            HumanMessage(content=f"Approval input: {user_input}"),
            AIMessage(content=f"Approved {len(approved)} candidates: {names}"),
        ],
    }


def _parse_approval(
    choice: str, candidates: list[CandidateProfile]
) -> list[CandidateProfile]:
    if choice in ("all", "a"):
        return candidates

    if choice in ("reject", "r", "no", "n"):
        return []

    try:
        indices = [int(x.strip()) - 1 for x in choice.split(",")]
        selected = [candidates[i] for i in indices if 0 <= i < len(candidates)]
        return selected
    except (ValueError, IndexError):
        print("Input not recognized, I approve all as a fallback.")
        return candidates


def should_continue_after_approval(state: HRGraphState) -> str:
    if state.get("human_approved"):
        return "profile_scorer"
    else:
        return "__end__"
