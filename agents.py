"""
The three specialist agents for the internship outreach system.

Design note: I went with a single shared global state (TypedDict) across all
agents rather than isolated per-agent scratchpads. Per this week's theory,
shared state is the right call for a small pipeline-shaped system like this
one (a handful of agents, no deep hierarchy) — isolated state only starts
paying for itself once you've got many agents stepping on each other.
"""

import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from browser_tools import fetch_page_text
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.5,
)

MIN_BODY_WORDS = 60
MAX_DRAFT_ATTEMPTS = 3

GENERIC_PHRASES = [
    "i am writing to you today",
    "i hope this email finds you well and i am reaching out",
    "to whom it may concern",
    "i am a highly motivated individual",
]

# Reusing the same retry-then-simplify Wikipedia wrapper from Week 3 —
# it's the fallback path when we can't scrape the company site directly.
_wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1200)
)


class AgentState(TypedDict):
    # inputs
    company_name: str
    company_url: str
    manual_job_notes: str  # optional human-supplied context, used if scraping fails
    recipient_email: str
    recipient_name: str
    sender_profile: str

    # research agent output
    research_summary: str
    research_source: str  # "scraped" | "wikipedia_fallback" | "manual" | "none"

    # writer agent output
    subject: str
    body: str

    # critic agent output
    critic_feedback: str
    approved: bool
    draft_attempts: int

    # human checkpoint + send
    human_approved: bool
    sent: bool


# ---------------------------------------------------------------------------
# Research agent
# ---------------------------------------------------------------------------
def research_node(state: AgentState) -> AgentState:
    print(f"\n[research] Looking into {state['company_name']}...")

    summary = ""
    source = "none"

    if state.get("company_url"):
        try:
            # Real headless browser instead of a plain HTTP GET — some
            # careers pages render with JS and would come back empty
            # otherwise. Reuses the thread-pinned Playwright setup from the
            # SOC-2026 browser agent (see browser_tools.py).
            page_text = fetch_page_text(state["company_url"])
            summary = " ".join(page_text.split())[:1500]
            source = "scraped"
            print(f"[research] Fetched {len(summary)} chars from {state['company_url']}")
        except Exception as e:
            print(f"[research] Browser fetch failed ({e}), falling back to Wikipedia...")

    if not summary:
        # Fallback agent: if direct scraping isn't possible (site blocks bots,
        # URL not given, etc.), fall back to a general-purpose lookup rather
        # than failing silently — same principle as the Week 3 search retry.
        try:
            wiki_result = _wiki.run(state["company_name"])
            if wiki_result and "No good Wikipedia" not in wiki_result:
                summary = wiki_result[:1200]
                source = "wikipedia_fallback"
                print("[research] Used Wikipedia fallback.")
        except Exception as e:
            print(f"[research] Wikipedia fallback also failed: {e}")

    if not summary and state.get("manual_job_notes"):
        summary = state["manual_job_notes"]
        source = "manual"
        print("[research] Using manually supplied job notes instead.")

    if not summary:
        summary = f"No specific research found on {state['company_name']}."
        source = "none"

    return {**state, "research_summary": summary, "research_source": source}


# ---------------------------------------------------------------------------
# Writer agent
# ---------------------------------------------------------------------------
def writer_node(state: AgentState) -> AgentState:
    attempt = state.get("draft_attempts", 0)
    print(f"\n[writer] Attempt {attempt + 1}: drafting email to {state['recipient_name']}...")

    feedback_note = (
        f"\nThe previous draft was rejected for this reason, fix it: {state['critic_feedback']}"
        if state.get("critic_feedback")
        else ""
    )

    prompt = f"""Write a short, personalised, professional cold email for an internship inquiry.

Recipient name: {state['recipient_name']}
Company: {state['company_name']}
Research notes about the company (use specifics from this, don't invent facts beyond it):
{state['research_summary']}

Sender profile (sign off with the sender's name given here):
{state['sender_profile']}
{feedback_note}

Requirements:
- Reference at least one concrete detail from the research notes.
- Reference at least one concrete project from the sender profile.
- At least {MIN_BODY_WORDS} words in the body.
- Polite, professional, not generic boilerplate.
- Respond in EXACTLY this format, nothing else:
SUBJECT: <subject line>
BODY:
<email body>
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content

    if "SUBJECT:" in text and "BODY:" in text:
        subject = text.split("SUBJECT:")[1].split("BODY:")[0].strip()
        body = text.split("BODY:")[1].strip()
    else:
        subject = f"Regarding an internship opportunity at {state['company_name']}"
        body = text.strip()

    return {**state, "subject": subject, "body": body}


# ---------------------------------------------------------------------------
# Critic agent
# ---------------------------------------------------------------------------
def critic_node(state: AgentState) -> AgentState:
    body = state["body"]
    word_count = len(body.split())
    too_short = word_count < MIN_BODY_WORDS
    too_generic = any(p in body.lower() for p in GENERIC_PHRASES)

    # Ground the draft against the actual research, not just the raw context
    # string — this is the main behavioural difference from Week 3's version.
    research_keywords = [
        w.strip(".,()").lower()
        for w in state["research_summary"].split()
        if len(w) > 5
    ]
    mentions_research = any(kw in body.lower() for kw in research_keywords[:30])

    approved = (not too_short) and (not too_generic) and mentions_research

    feedback_parts = []
    if too_short:
        feedback_parts.append(f"Body has only {word_count} words, needs at least {MIN_BODY_WORDS}.")
    if too_generic:
        feedback_parts.append("Body uses generic boilerplate phrasing, make it more specific.")
    if not mentions_research:
        feedback_parts.append("Body doesn't clearly reference the research findings, tie them in explicitly.")

    feedback = " ".join(feedback_parts) if feedback_parts else "Looks good."
    print(f"[critic] word_count={word_count}, approved={approved}")

    return {
        **state,
        "approved": approved,
        "critic_feedback": feedback,
        "draft_attempts": state.get("draft_attempts", 0) + 1,
    }
