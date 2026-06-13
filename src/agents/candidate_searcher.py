import json
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from src.models.state import HRGraphState
from src.models.candidate_profile import CandidateProfile
from src.models.job_requirements import JobRequirements
from src.tools.definitions import SEARCHER_TOOLS


class CandidateListOutput(BaseModel):
    candidates: list[CandidateProfile]


def candidate_searcher_node(state: HRGraphState) -> dict:
    print("\n" + "─" * 60)
    print("  Node: Candidate Searcher (ReAct Agent)")
    print("─" * 60)

    requirements: JobRequirements = state["requirements"]

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    structured_llm = llm.with_structured_output(CandidateListOutput)

    # The ReAct agent still uses the raw LLM for the Reason→Act→Observe loop
    agent = create_react_agent(llm, tools=SEARCHER_TOOLS)

    skills_str = ", ".join(requirements.required_skills[:5])
    task = f"""You're an experienced technical recruiter. Find 3-5 real candidates for:
    Role: {requirements.title}
    Seniority: {requirements.seniority.value}
    Domain: {requirements.domain}
    Skills: {skills_str}
    Experience: {requirements.years_of_experience or 'unspecified'}

    Strategy:
    1. Search LinkedIn: "site:linkedin.com/in {requirements.title.lower()} {skills_str}"
    2. Search GitHub: "{requirements.required_skills[0] if requirements.required_skills else ''} developer github"
    3. Test variations to find different candidates

    Collect for each: name, role, company, skills, experience, URL.
    """

    # Step 1: the ReAct agent browses the web freely
    search_result = agent.invoke({"messages": [HumanMessage(content=task)]})
    raw_findings = search_result["messages"][-1].content

    # Step 2: with_structured_output converts raw text → Pydantic
    # No more regex, json.loads, _extract_text_content or parsing try/except
    parsed: CandidateListOutput = structured_llm.invoke(
        f"Structure these search results into candidates:\n\n{raw_findings}"
    )

    candidates = parsed.candidates

    print(f"  → Found {len(candidates)} candidates")
    for c in candidates:
        print(f"     • {c.name} — {c.current_role} @ {c.current_company or '?'}")

    return {
        "candidates": candidates,
        "messages": [AIMessage(content=f"Found {len(candidates)} candidates for {requirements.title}")],
    }