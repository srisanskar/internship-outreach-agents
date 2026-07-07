# Internship Outreach Multi-Agent System

Capstone project for Agentic AI Learners' Space 2026 - Week 4.

## Problem statement

I'm currently applying to AI/ML internships, and honestly the annoying part
isn't finding openings, it's writing an email for each one that doesn't
sound copy-pasted. To do it properly you have to actually look up what the
company does, tie in something specific from your own projects, check the
draft doesn't sound generic, and only then send it. That's basically three
different jobs, so instead of writing one script that does all of it badly,
I split it across agents that each do one part well.

This felt like a genuine multi-agent problem rather than "one agent with
extra steps" - the research step, the writing step, and the quality-check
step all need pretty different logic and none of them can really do the
others' job.

## Agents (2+ required - I used 3)

1. **Research agent** - given a company name and (optionally) a careers/about
   page URL, opens the page with a real headless browser and pulls the
   visible text. If there's no URL, or the page fails to load, it falls back
   to a Wikipedia search for general info about the company, and if even
   that comes up empty, it'll just use a manually typed note instead.

2. **Writer agent** - takes whatever the research agent found plus a short
   blurb about my own project background, and drafts a subject + body for
   the email. If a previous draft got rejected it also gets that feedback so
   it can fix the actual problem instead of just trying again blind.

3. **Critic agent** - checks the draft: is it long enough, does it avoid
   generic boilerplate phrases, does it actually mention something from the
   research (not just restate the input). If it fails any of these it sends
   feedback back.

## Orchestration pattern used: Supervisor

```
research → writer → critic → supervisor ─┬─(reject, attempts < 3)→ writer (retry)
                                          └─(approved OR out of attempts)→ human_approval → send
```

I went with Supervisor because there's one point in the flow where a real
decision has to get made - after the critic checks a draft, something has to
decide whether to loop back or move forward - and everywhere else is just a
fixed sequence. The supervisor node makes that call using LangGraph's
`Command` object, which routes to whichever node it returns instead of
needing separate conditional-edge functions everywhere.

I thought about Pipeline since most of the flow really is sequential, but
Pipeline alone doesn't have a clean way to express "go back a step if this
fails," which is the whole point of having a critic in the first place. So
Supervisor fit better even though only one node in the whole graph is doing
the branching.

**Shared vs isolated state:** I used one shared state object across all
three agents instead of giving each one its own scratchpad. With only 3
agents running in a fixed order there wasn't really a risk of one agent's
output getting clobbered by another, so isolated state would've just been
extra complexity for no real benefit here.

## Failure handling (brief asks for at least 1, this has 3)

- **Retry with feedback** - critic rejects → writer gets specific feedback
  and tries again, max 3 times so it can't loop forever.
- **Fallback chain in research** - real browser scrape → Wikipedia → manual
  notes, in that order, so a bad URL doesn't just kill the whole run.
- **Human approval gate** - nothing gets sent without a y/n prompt, and if
  the writer/critic loop never converges after 3 tries it gets forced to
  this same checkpoint instead of sending something bad or just failing
  silently.

## What I reused from earlier weeks (and what's new)

- The Gmail MCP server (`gmail_mcp_server.py`) is basically unchanged from
  Week 3 - self-hosted with the `mcp` Python SDK since the npm package the
  assignment originally suggested was still 404ing.
- The critic agent's approach (word count / generic-phrase check / keyword
  grounding) is the same idea as Week 3's cold-email `review_node`, just
  checking against research findings instead of a hand-typed context string.
- The Wikipedia fallback in research reuses the retry-then-simplify wrapper
  from the Week 3 ReAct agent almost exactly.
- New for this week: the actual research step uses a real headless browser
  (`browser_tools.py`) instead of Week 3's plain approach, because some
  careers pages render with JS and a normal request would just come back
  empty. The thread-pinning setup in there is carried over from a different
  course project of mine (SOC-2026 browser agent) where I'd already run
  into and fixed the "can't switch to a different thread" Playwright error.
- Also new: the supervisor node itself, using `Command`-based routing
  instead of the plain conditional-edge function I used in Week 3.

One thing I skipped on purpose: LangGraph has a native `interrupt()` +
checkpointer mechanism for human-in-the-loop that lets you pause a graph and
resume it later, even across separate runs. I used a plain blocking
`input()` instead, which works fine for a single run but can't be paused and
picked up again later. Given the one-week timeline I wanted something I
could fully test end-to-end rather than debug a checkpointer setup under
deadline pressure. Swapping in `interrupt()` would be the natural next step
if I extend this.

## Files

- `agents.py` - state definition + research/writer/critic node functions
- `browser_tools.py` - headless browser fetch for research (Playwright)
- `orchestrator.py` - supervisor + human checkpoint + send node + graph
  wiring + CLI entry point
- `gmail_mcp_server.py` - Gmail MCP server, from Week 3

## Setup

```bash
pip install langchain-groq langgraph langchain-mcp-adapters langchain-community mcp playwright python-dotenv wikipedia
playwright install chromium
```

Make a `.env` file in the same folder with your own keys (not included in
this repo, obviously):

```
GROQ_API_KEY=your_key_here
GMAIL_ADDRESS=your_gmail@gmail.com
GMAIL_APP_PASSWORD=your_gmail_app_password
```

## Running it

```bash
python orchestrator.py
```

It'll ask for the company name, a careers/about page URL (optional - you can
paste a job description instead if you skip it), and the recipient's
name/email. Then it researches, drafts, critiques, loops if needed, asks you
to approve, and sends.

## Testing

I tested the graph's actual routing logic - does it retry correctly, does it
stop after 3 attempts, does it escalate to human review properly — using
stubbed research/writer/send functions, since I don't always have Groq/Gmail
access in every environment I write code in. Caught and fixed a real bug
this way (an earlier version of the supervisor routing could loop forever
because it wasn't re-checking the critic's verdict after a retry — fixed by
making research → writer → critic a fixed sequence and only branching at the
one point that actually needs a decision). The real LLM calls and actual
email sending were tested locally the normal way, same as previous weeks.
