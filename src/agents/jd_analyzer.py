from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

from src.models.state import HRGraphState
from src.models.job_requirements import JobRequirements


def jd_analyzer_node(state: HRGraphState) -> dict:
    print("\n" + "─" * 60)
    print("  Node: JD Analyzer")
    print("─" * 60)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )
    structured_llm = llm.with_structured_output(JobRequirements)

    prompt = f"""You're an experienced HR analyst. Analyze this job description and extract structured requirements.

    Normalize skill names (e.g., "k8s" → "Kubernetes," "Postgres" → "PostgreSQL").
    Accurately distinguish mandatory requirements from "nice to haves."
    Infer seniority level from context if it's not explicit.

    JOB DESCRIPTION:
    {state['job_description']}
    """

    requirements: JobRequirements = structured_llm.invoke(prompt)

    print(f"  → Role: {requirements.title} ({requirements.seniority.value})")
    print(f"  → Required skills: {', '.join(requirements.required_skills[:4])}")
    print(f"  → Domain: {requirements.domain}")

    return {
        "requirements": requirements,
        "messages": [AIMessage(content=f"Job description analized: {requirements.title}, {requirements.seniority.value}, domain: {requirements.domain}")],
    }
