from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.models.state import HRGraphState
from src.agents.jd_analyzer import jd_analyzer_node
from src.agents.candidate_searcher import candidate_searcher_node
from src.agents.human_approval import human_approval_node, should_continue_after_approval
from src.agents.profile_scorer import profile_scorer_node
from src.agents.report_generator import report_generator_node


def build_hr_graph():
    # ── 1. Crea il grafo tipizzato ────────────────────────────────────────────
    builder = StateGraph(HRGraphState)

    # ── 2. Registra i nodi ────────────────────────────────────────────────────
    builder.add_node("jd_analyzer",        jd_analyzer_node)
    builder.add_node("candidate_searcher", candidate_searcher_node)
    builder.add_node("human_approval",     human_approval_node)
    builder.add_node("profile_scorer",     profile_scorer_node)
    builder.add_node("report_generator",   report_generator_node)

    # ── 3. Edge statici (sequenza lineare fino all'approval gate) ─────────────
    builder.add_edge(START,                "jd_analyzer")
    builder.add_edge("jd_analyzer",        "candidate_searcher")
    builder.add_edge("candidate_searcher", "human_approval")

    # ── 4. Edge condizionale dopo human_approval ──────────────────────────────
    # should_continue_after_approval legge state["human_approved"] e decide:
    #   True  → "profile_scorer"
    #   False → END
    builder.add_conditional_edges(
        "human_approval",
        should_continue_after_approval,
        {
            "profile_scorer": "profile_scorer",
            "__end__": END,
        },
    )

    # ── 5. Edge statici nella seconda metà del pipeline ───────────────────────
    builder.add_edge("profile_scorer",  "report_generator")
    builder.add_edge("report_generator", END)

    # ── 6. Checkpointer per la persistenza dello stato ────────────────────────
    # NECESSARIO per interrupt() / human-in-the-loop.
    # MemorySaver → in-memory (sviluppo)
    # SqliteSaver / RedisSaver → produzione
    checkpointer = MemorySaver()

    # ── 7. Compila il grafo ───────────────────────────────────────────────────
    graph = builder.compile(checkpointer=checkpointer)

    return graph, checkpointer
