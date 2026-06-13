import json
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from src.models.state import HRGraphState
from src.models.candidate_profile import ScoredCandidate
from src.models.job_requirements import JobRequirements
from src.tools.definitions import REPORTER_TOOLS


def report_generator_node(state: HRGraphState) -> dict:
    print("\n" + "─" * 60)
    print("  📝 Node: Report Generator")
    print("─" * 60)

    scored: list[ScoredCandidate] = state["scored_candidates"]
    requirements: JobRequirements = state["requirements"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"talent_report_{requirements.title.replace(' ', '_').lower()}_{timestamp}.md"

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
    )

    # Agent with save_report tool to write the file
    agent = create_react_agent(llm, tools=REPORTER_TOOLS)

    # Serialize data for the prompt
    candidates_data = [
        {
            "rank": sc.rank,
            "name": sc.profile.name,
            "current_role": sc.profile.current_role,
            "current_company": sc.profile.current_company,
            "years_of_experience": sc.profile.years_of_experience,
            "skills": sc.profile.skills,
            "education": sc.profile.education,
            "source_url": sc.profile.source_url,
            "overall_score": sc.overall_score,
            "skills_match_score": sc.skills_match_score,
            "experience_score": sc.experience_score,
            "domain_fit_score": sc.domain_fit_score,
            "strengths": sc.strengths,
            "gaps": sc.gaps,
            "recommendation": sc.recommendation,
        }
        for sc in scored
    ]

    task = f"""You're a senior talent acquisition consultant. Write a professional report in Markdown.

    FILENAME TO USE: {filename}

    ROLE REQUIREMENTS:
    {json.dumps({
        "title": requirements.title,
        "seniority": requirements.seniority.value,
        "domain": requirements.domain,
        "required_skills": requirements.required_skills,
        "summary": requirements.summary,
    }, indent=2, ensure_ascii=False)}

    CANDIDATES EVALUATED (in order of rank):
    {json.dumps(candidates_data, indent=2, ensure_ascii=False)}

    Report Structure:
    1. # Executive Summary — search overview and key recommendation
    2. ## Role Requirements — key skills and required seniority
    3. ## Candidate Rankings — table with name, total score, and strengths
    4. ## Detailed Profiles — one section per candidate with score breakdown, strengths, gaps, and recommendation
    5. ## Next Steps — suggested interview process and key questions

    Style: Executive, direct, every sentence should add value. Use tables for comparisons. Use ✅ ❌ ⚠️ sparingly.

    After writing the report, use the save_report tool to save it as '{filename}'.
    """

    print("report pre-invoke")
    result = agent.invoke({"messages": [HumanMessage(content=task)]})
    print("report post-invoke")

    # Look for the saved path in tool result messages
    report_path = filename  # fallback
    report_content = ""

    for msg in reversed(result["messages"]):
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if '"path"' in content:
            try:
                data = json.loads(content)
                if "path" in data:
                    report_path = data["path"]
            except Exception:
                pass
        if len(content) > 500 and "#" in content:
            report_content = content

    print(f"  → Report saved: {report_path}")

    return {
        "report_path": report_path,
        "report_content": report_content,
        "messages": [AIMessage(content=f"Report generated and saved: {report_path}")],
    }
