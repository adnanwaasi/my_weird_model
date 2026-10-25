"""The research agent as a LangGraph graph.

    START -> plan -> sweep -> researcher <-> tools
                                  |
                                  v
                                write -> END

- plan:       researcher LLM writes search queries (structured output)
- sweep:      runs the standard tool calls + planned searches in parallel (no LLM)
- researcher: researcher LLM decides which follow-up tools to call
- tools:      runs those calls, records each result as numbered evidence [E#]
- write:      writer LLM turns the evidence into the cited report

To add a step (e.g. a fact-checker), write a node function and wire it in build_graph().
"""

import json
import operator
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

import prompts
from get_llm import get_llm
from tools import TOOLS, call_tool

TOOL_RESULT_CHARS_FOR_RESEARCHER = 4000  # what the small model sees per tool result
DIGEST_CHARS = 600  # per-evidence preview given to the researcher
WRITER_CHARS = 12000  # per-evidence excerpt given to the writer


def log(msg: str):
    print(f"\033[2m{msg}\033[0m", flush=True)


# ---------------------------------------------------------------- state


class Evidence(TypedDict):
    id: str
    tool: str
    args: dict
    result: str


class ResearchState(TypedDict):
    question: str
    ticker: str | None
    queries: list[str]
    evidence: Annotated[list[Evidence], operator.add]  # nodes return new items; LangGraph appends
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    report: str


def is_failed(ev: Evidence) -> bool:
    return ev["result"].startswith('{"error"')


def label(ev: Evidence) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in ev["args"].items())
    return f"[{ev['id']}] {ev['tool']}({args})"


def _call_key(tool: str, args: dict) -> str:
    return tool + json.dumps(args, sort_keys=True)


def _text(content) -> str:
    """Message content can be a string or a list of blocks (e.g. thinking + text)."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


class SearchPlan(BaseModel):
    queries: list[str] = Field(description="5 to 7 diverse web search queries")


# ---------------------------------------------------------------- graph


def build_graph(researcher_spec: str = "researcher", writer_spec: str = "writer", max_steps: int = 8):
    researcher = get_llm(researcher_spec)
    researcher_with_tools = researcher.bind_tools(TOOLS)

    def plan(state: ResearchState):
        q, t = state["question"], state["ticker"]
        prompt = prompts.PLANNER.format(today=date.today(), question=q, ticker_note=f" (ticker {t})" if t else "")
        try:
            queries = researcher.with_structured_output(SearchPlan).invoke(prompt).queries[:7]
        except Exception as e:  # small models sometimes emit broken JSON
            log(f"planner failed ({e}); using default queries")
            queries = []
        subject = t or q
        queries = queries or [f"{subject} latest news", f"{subject} competitors market share",
                              f"{subject} risks", f"{subject} analyst outlook {date.today().year}"]
        log("plan:\n  " + "\n  ".join(queries))
        return {"queries": queries}

    def sweep(state: ResearchState):
        calls = [("web_search", {"query": q}) for q in state["queries"]]
        if t := state["ticker"]:
            calls = [
                ("stock_quote", {"ticker": t}),
                ("price_history", {"ticker": t, "period": "1y"}),
                ("financials", {"ticker": t}),
                ("sec_filings", {"ticker": t}),
                ("news_search", {"query": f"{t} stock"}),
                ("lstm_forecast", {"ticker": t}),
            ] + calls
        log(f"sweep: {len(calls)} calls in parallel")
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda c: call_tool(*c), calls))
        start = len(state["evidence"])
        evidence = [Evidence(id=f"E{start + i + 1}", tool=tool, args=args, result=res)
                    for i, ((tool, args), res) in enumerate(zip(calls, results))]
        for ev in evidence:
            log(f"  {label(ev)}{'  (failed)' if is_failed(ev) else ''}")

        digest = "\n".join(f"{label(e)}: {e['result'][:DIGEST_CHARS]}" for e in evidence if not is_failed(e))
        messages = [
            SystemMessage(prompts.RESEARCHER.format(today=date.today())),
            HumanMessage(f"Research question: {state['question']}\n\nEvidence so far:\n{digest}"),
        ]
        return {"evidence": evidence, "messages": messages}

    def research(state: ResearchState):
        return {"messages": [researcher_with_tools.invoke(state["messages"])], "steps": state["steps"] + 1}

    def tools(state: ResearchState):
        seen = {_call_key(e["tool"], e["args"]) for e in state["evidence"]}
        n = len(state["evidence"])
        new_evidence, replies = [], []
        for tc in state["messages"][-1].tool_calls:
            if _call_key(tc["name"], tc["args"]) in seen:  # small models love repeating calls
                content = "Already called with these arguments; use the earlier result."
            else:
                ev = Evidence(id=f"E{n + len(new_evidence) + 1}", tool=tc["name"], args=tc["args"],
                              result=call_tool(tc["name"], tc["args"]))
                new_evidence.append(ev)
                seen.add(_call_key(ev["tool"], ev["args"]))
                log(f"  step {state['steps']}: {label(ev)}{'  (failed)' if is_failed(ev) else ''}")
                content = f"{ev['id']}: {ev['result'][:TOOL_RESULT_CHARS_FOR_RESEARCHER]}"
            replies.append(ToolMessage(content=content, tool_call_id=tc["id"], name=tc["name"]))
        return {"evidence": new_evidence, "messages": replies}

    def route_after_research(state: ResearchState):
        if state["messages"][-1].tool_calls and state["steps"] < max_steps:
            return "tools"
        log(f"follow-up finished after {state['steps']} rounds")
        return "write"

    def write(state: ResearchState):
        usable = [e for e in state["evidence"] if not is_failed(e)]
        log(f"collected {len(usable)} usable pieces of evidence ({len(state['evidence']) - len(usable)} failed)")
        ticker_note = f" (ticker: {state['ticker']})" if state["ticker"] else ""

        def prompt(max_chars):
            block = "\n\n".join(f'<evidence id="{e["id"]}" call="{label(e)}">\n{e["result"][:max_chars]}\n</evidence>'
                                for e in usable)
            return [
                SystemMessage(prompts.WRITER),
                HumanMessage(f"Today is {date.today()}.\nResearch question: {state['question']}{ticker_note}\n\n"
                             f"{block}\n\nWrite the report now."),
            ]

        def stream(llm, messages):
            parts = []
            for chunk in llm.stream(messages):
                piece = _text(chunk.content)
                print(piece, end="", flush=True)
                parts.append(piece)
            print()
            return "".join(parts)

        try:
            log(f"writing report with {writer_spec}...")
            report = stream(get_llm(writer_spec), prompt(WRITER_CHARS))
        except Exception as e:
            # e.g. no ANTHROPIC_API_KEY: fall back to the local model with shorter excerpts
            log(f"writer unavailable ({type(e).__name__}: {e}); writing with {researcher_spec}")
            report = stream(get_llm(researcher_spec, num_ctx=32768), prompt(1200))
        return {"report": report}

    g = StateGraph(ResearchState)
    g.add_node("plan", plan)
    g.add_node("sweep", sweep)
    g.add_node("researcher", research)
    g.add_node("tools", tools)
    g.add_node("write", write)
    g.add_edge(START, "plan")
    g.add_edge("plan", "sweep")
    g.add_edge("sweep", "researcher")
    g.add_conditional_edges("researcher", route_after_research, {"tools": "tools", "write": "write"})
    g.add_edge("tools", "researcher")
    g.add_edge("write", END)
    return g.compile()


def initial_state(question: str, ticker: str | None) -> ResearchState:
    return ResearchState(question=question, ticker=ticker.upper() if ticker else None,
                         queries=[], evidence=[], messages=[], steps=0, report="")
