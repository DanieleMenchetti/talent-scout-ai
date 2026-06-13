import argparse
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langgraph.types import Command
from langchain_core.messages import HumanMessage

from src.graph import build_hr_graph

from dotenv import load_dotenv

load_dotenv()


SAMPLE_JD = """
Senior Backend Engineer — FinTech Payments Platform

About Us:
We are a fast-growing fintech startup building B2B payments
infrastructure. Our platform processes over $2 billion
in annual transaction volume in 15 countries.

The Role:
We are seeking a Senior Backend Engineer for our Payments Core team.
You will design and build systems that move money reliably
and securely at scale.

Responsibilities:
- Design and implement high-throughput, low-latency payment processing services
- Build and maintain REST and gRPC APIs used by hundreds of partners
- Ensure PCI-DSS compliance in our transaction processing stack
- Lead technical design reviews and mentor junior engineers

Requirements:
- 5+ years of backend experience
- Strong proficiency in Python or Go
- Deep experience with PostgreSQL and Redis
- Experience with event-driven architectures (Kafka or similar)
- Understanding of financial systems and payment protocols
- Experience with Docker and Kubernetes in production

Nice to Have:
- Experience with PCI-DSS or SOC2 compliance
- Knowledge of SWIFT, ISO 20022, or card network protocols
- Experience with AWS
- Previous startup experience
"""


def main():
    parser = argparse.ArgumentParser(description="HR Multi-Agent System (LangGraph + Gemini)")
    parser.add_argument("--jd", type=str, default=None, help="Path to the job description .txt file")
    args = parser.parse_args()

    # Verify Gemini API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("\n❌ Error: GOOGLE_API_KEY not set.")
        print("   Export with: export GOOGLE_API_KEY=your_key")
        print("   Get a free key on: https://aistudio.google.com/apikey")
        sys.exit(1)

    # Create the results folder if it doesn't exist
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Load job description
    if args.jd:
        with open(args.jd, "r", encoding="utf-8") as f:
            job_description = f.read()
        print(f"📄 Job description loaded from: {args.jd}")
    else:
        job_description = SAMPLE_JD
        print("📄 I use the example job description")

    # Build the graph
    graph, _ = build_hr_graph()

    # Each graph run has a unique thread_id — required by the checkpointer
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n🧵 Thread ID: {thread_id}")
    print("=" * 60)

    # ── First invocation: start the graph ────────────────────────────────────
    # The graph runs until human_approval, where interrupt() will pause it.
    print("\n🚀 Starting pipeline...\n")

    result = graph.invoke(
        {
            "job_description": job_description,
            "messages": [HumanMessage(content="Start the candidate search")],
        },
        config=config,
    )

    # ── Check if the graph is paused on interrupt ────────────────────────────
    graph_state = graph.get_state(config)

    if graph_state.next:
        # The graph is waiting for human input
        # Retrieve the interrupt message from pending tasks
        pending_tasks = graph_state.tasks
        interrupt_message = ""
        for task in pending_tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupt_message = task.interrupts[0].value
                break

        if interrupt_message:
            print("\n" + "═" * 60)
            print("  ⏸️  PAUSED GRAPH — Input required")
            print("═" * 60)
            print(interrupt_message)

        user_input = input("\n  Your choice: ").strip()
        print("\n▶️  Pipeline recovery...\n")

        final_result = graph.invoke(
            Command(resume=user_input),
            config=config,
        )
    else:
        # The graph has terminated (candidates rejected before interrupt, or other reason)
        final_result = result

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  📊 EXECUTION SUMMARY")
    print("═" * 60)

    report_path = final_result.get("report_path")
    scored = final_result.get("scored_candidates", [])
    human_approved = final_result.get("human_approved")

    if human_approved is False:
        print("  ❌ Pipeline closed: candidates not approved.")
    elif report_path:
        print(f"  ✅ Completed successfully!")
        print(f"  📄 Report: {report_path}")
        if scored:
            print(f"  🏆 Top candidate: {scored[0].profile.name} ({scored[0].overall_score:.1f}/10)")
    else:
        print("  ⚠️  Pipeline completed but report not found.")

    print(f"\n  Thread ID (for debug): {thread_id}")
    print()


if __name__ == "__main__":
    main()
