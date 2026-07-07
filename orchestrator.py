"""
Supervisor-orchestrated multi-agent system for internship outreach.

Pattern: Supervisor. Research -> Writer -> Critic is a fixed pipeline (each
step always leads to the next), and a dedicated supervisor node makes the
one decision that actually branches: given the critic's verdict, should we
retry the writer, or move on to human approval? This is the direct
extension of the conditional-edge routing from Week 3 (route_after_review),
pulled out into its own explicit decision node using LangGraph's Command
object instead of being baked into one function.

An earlier version of this graph routed every single node back through the
supervisor and had the supervisor re-check `critic_feedback` to decide
whether to re-run the critic. That was buggy: after a retry, the writer's
new draft never got re-evaluated because `critic_feedback` was still set
from the previous round, so the supervisor kept sending it back to the
writer forever (caught this during testing — see the recursion error it
produced). Fixed edges for the pipeline steps eliminate that whole class
of bug.

Flow:
    research -> writer -> critic -> supervisor -> [retry: writer]
                                               -> [proceed: human_approval]
    human_approval -> [approved: send -> end]
                    -> [rejected: end]

Failure handling built in (per this week's brief, at least one is required
— this has three):
    1. Retries: critic rejection sends the draft back to the writer with
       feedback, up to MAX_DRAFT_ATTEMPTS.
    2. Fallback agent: research falls back from site-scraping to Wikipedia
       to manually-supplied notes if the primary source fails.
    3. Human-in-the-loop: nothing gets sent without an explicit approval
       step, and if the writer/critic loop can't converge, it's forced to
       human review rather than sent silently.
"""

import asyncio
import os
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langchain_mcp_adapters.client import MultiServerMCPClient

from agents import (
    AgentState,
    research_node,
    writer_node,
    critic_node,
    MAX_DRAFT_ATTEMPTS,
)


# ---------------------------------------------------------------------------
# Human approval checkpoint
# ---------------------------------------------------------------------------
def human_approval_node(state: AgentState) -> AgentState:
    print("\n" + "=" * 70)
    print("HUMAN APPROVAL CHECKPOINT")
    print("=" * 70)
    print(f"To: {state['recipient_email']}")
    print(f"Subject: {state['subject']}\n")
    print(state["body"])
    print("=" * 70)

    if not state["approved"]:
        print(
            f"\nNote: the critic did NOT approve this draft after "
            f"{state['draft_attempts']} attempts. Feedback: {state['critic_feedback']}"
        )

    answer = input("\nSend this email? [y/n]: ").strip().lower()
    return {**state, "human_approved": answer == "y"}


def route_after_human(state: AgentState) -> Literal["send", "end"]:
    return "send" if state["human_approved"] else "end"


# ---------------------------------------------------------------------------
# Send node (Gmail MCP — same approach as Week 3)
# ---------------------------------------------------------------------------
async def _send_email_via_mcp(state: AgentState) -> AgentState:
    print(f"[send] Sending email to {state['recipient_email']} via Gmail MCP...")
    server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gmail_mcp_server.py")

    client = MultiServerMCPClient(
        {
            "gmail": {
                "command": "python",
                "args": [server_path],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    send_tool = next((t for t in tools if t.name == "send_email"), None)
    if send_tool is None:
        available = [t.name for t in tools]
        raise RuntimeError(f"Could not find send_email tool. Available: {available}")

    result = await send_tool.ainvoke(
        {"to": state["recipient_email"], "subject": state["subject"], "body": state["body"]}
    )
    print(f"[send] MCP tool result: {result}")
    return {**state, "sent": True}


def send_node(state: AgentState) -> AgentState:
    return asyncio.run(_send_email_via_mcp(state))


# ---------------------------------------------------------------------------
# Supervisor: the one real branching decision in this graph
# ---------------------------------------------------------------------------
def supervisor_node(state: AgentState) -> Command[Literal["writer", "human_approval"]]:
    if not state["approved"] and state["draft_attempts"] < MAX_DRAFT_ATTEMPTS:
        print(f"[supervisor] Draft rejected (attempt {state['draft_attempts']}), sending back to writer.")
        return Command(goto="writer")

    if not state["approved"]:
        print("[supervisor] Max attempts reached without approval — escalating to human review.")
    return Command(goto="human_approval")


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("research", research_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("send", send_node)

    graph.set_entry_point("research")
    graph.add_edge("research", "writer")
    graph.add_edge("writer", "critic")
    graph.add_edge("critic", "supervisor")
    # supervisor's own return value (a Command) decides the next hop
    graph.add_conditional_edges("human_approval", route_after_human, {"send": "send", "end": END})
    graph.add_edge("send", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    print("=== Internship Outreach Multi-Agent System ===\n")
    company_name = input("Company name: ").strip()
    company_url = input("Company careers/about page URL (optional, press enter to skip): ").strip()
    manual_notes = ""
    if not company_url:
        manual_notes = input("No URL given — paste a short job description or note instead (optional): ").strip()
    recipient_name = input("Recipient name: ").strip()
    recipient_email = input("Recipient email: ").strip()

    sender_profile = """
Sanskar Srivastava — B.Tech student building toward AI/ML internships.
Recent project work:
- Built a LangChain ReAct agent and a cyclic LangGraph workflow with
  self-correction loops (Agentic AI Learners' Space).
- Built a self-hosted Gmail MCP server (Python MCP SDK) to send agent-drafted
  emails, after the suggested npm package was unavailable.
- Built a RAG-powered chatbot using FAISS and HuggingFace embeddings (ML
  Learners' Space).
- Comfortable with Python, SQL, and browser automation via Playwright.
"""

    initial_state: AgentState = {
        "company_name": company_name,
        "company_url": company_url,
        "manual_job_notes": manual_notes,
        "recipient_email": recipient_email,
        "recipient_name": recipient_name,
        "sender_profile": sender_profile,
        "research_summary": "",
        "research_source": "",
        "subject": "",
        "body": "",
        "critic_feedback": None,
        "approved": False,
        "draft_attempts": 0,
        "human_approved": None,
        "sent": False,
    }

    final_state = app.invoke(initial_state)

    # Close the browser explicitly here, before Python starts tearing down —
    # relying on atexit to do this during interpreter shutdown can race with
    # the Playwright driver process on Windows and throw a harmless but
    # alarming-looking EPIPE error after the script has already finished.
    from browser_tools import close_browser
    close_browser()

    print("\n" + "=" * 70)
    print("FINAL STATE")
    print("=" * 70)
    for key, value in final_state.items():
        print(f"{key}: {value}")