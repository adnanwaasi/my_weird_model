"""The research loop: plan -> parallel sweep -> agentic follow-up -> evidence for the writer.

The local (fine-tuned) model does the cheap, high-volume work: planning search queries
and deciding which leads to chase with tools. Every tool result is stored as numbered
evidence ([E1], [E2], ...) so the final report can cite exactly where each claim came from.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date

from ollama import Client

from tools import TOOL_SCHEMAS, call_tool

LOCAL_CTX = 16384
TOOL_RESULT_CHARS_FOR_LOCAL = 4000  # what the small model sees per tool result
DIGEST_CHARS = 600  # per-evidence preview in the follow-up prompt


@dataclass
class Evidence:
    id: str
    tool: str
    args: dict
    result: str

    @property
    def failed(self) -> bool:
        return self.result.startswith('{"error"')

    def label(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"[{self.id}] {self.tool}({args})"


@dataclass
class ResearchState:
    question: str
    ticker: str | None
    evidence: list[Evidence] = field(default_factory=list)
    seen: set = field(default_factory=set)

    def record(self, tool: str, args: dict, result: str) -> Evidence:
        ev = Evidence(f"E{len(self.evidence) + 1}", tool, args, result)
        self.evidence.append(ev)
        return ev

    def run(self, tool: str, args: dict) -> Evidence | None:
        key = (tool, json.dumps(args, sort_keys=True))
        if key in self.seen:  # small models love repeating the same call
            return None
        self.seen.add(key)
        return self.record(tool, args, call_tool(tool, args))


def log(msg: str):
    print(f"\033[2m{msg}\033[0m", flush=True)


class Researcher:
    def __init__(self, model: str, host: str | None = None, max_steps: int = 8):
        self.model = model
        self.client = Client(host=host)
        self.max_steps = max_steps

    def _chat(self, messages, **kw):
        return self.client.chat(model=self.model, messages=messages, options={"num_ctx": LOCAL_CTX}, **kw)

    # ------------------------------------------------------------ 1. plan

    def plan_queries(self, question: str, ticker: str | None) -> list[str]:
        prompt = (
            f"Today is {date.today()}. You are planning market research for: {question!r}"
            + (f" (ticker {ticker})" if ticker else "")
            + ".\nWrite 5 to 7 diverse web search queries covering: latest news and catalysts, "
            "competitive landscape, industry/market size trends, risks and controversies, "
            "analyst/expert opinion, and anything specific to the question.\n"
            'Reply with JSON only: {"queries": ["...", "..."]}'
        )
        try:
            resp = self._chat([{"role": "user", "content": prompt}], format="json")
            queries = [q for q in json.loads(resp.message.content).get("queries", []) if isinstance(q, str)]
        except Exception as e:  # malformed JSON from a small model shouldn't stop the run
            log(f"planner failed ({e}); using default queries")
            queries = []
        subject = ticker or question
        return queries[:7] or [f"{subject} latest news", f"{subject} competitors market share",
                               f"{subject} risks", f"{subject} analyst outlook {date.today().year}"]

    # ------------------------------------------------------------ 2. sweep

    def sweep(self, state: ResearchState, queries: list[str]):
        calls = [("web_search", {"query": q}) for q in queries]
        if state.ticker:
            t = state.ticker
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
        for (tool, args), result in zip(calls, results):
            state.seen.add((tool, json.dumps(args, sort_keys=True)))
            ev = state.record(tool, args, result)
            log(f"  {ev.label()}{'  (failed)' if ev.failed else ''}")

    # ------------------------------------------------------------ 3. follow up

    def follow_up(self, state: ResearchState):
        digest = "\n".join(f"{e.label()}: {e.result[:DIGEST_CHARS]}" for e in state.evidence if not e.failed)
        system = (
            f"Today is {date.today()}. You are a meticulous equity and market research analyst. "
            "You already have the evidence below. Your job now is to close the gaps: open the most "
            "informative URLs with fetch_page (primary sources, filings, earnings coverage, reputable "
            "outlets), and run new searches for anything still unknown. Prefer recent, specific, "
            "numeric facts. Call one or more tools per turn. When the evidence is enough for a "
            "thorough report, reply with a short list of the key open questions and stop calling tools."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Research question: {state.question}\n\nEvidence so far:\n{digest}"},
        ]
        for step in range(1, self.max_steps + 1):
            resp = self._chat(messages, tools=TOOL_SCHEMAS)
            msg = resp.message
            messages.append(msg)
            if not msg.tool_calls:
                log(f"follow-up finished after {step - 1} tool rounds")
                return
            for tc in msg.tool_calls:
                name, args = tc.function.name, dict(tc.function.arguments or {})
                ev = state.run(name, args)
                if ev is None:
                    content = "Already called with these arguments; use the earlier result."
                else:
                    log(f"  step {step}: {ev.label()}{'  (failed)' if ev.failed else ''}")
                    content = f"{ev.id}: {ev.result[:TOOL_RESULT_CHARS_FOR_LOCAL]}"
                messages.append({"role": "tool", "content": content, "tool_name": name})
        log(f"follow-up hit the {self.max_steps}-step limit")

    # ------------------------------------------------------------ all

    def research(self, question: str, ticker: str | None) -> ResearchState:
        state = ResearchState(question, ticker.upper() if ticker else None)
        queries = self.plan_queries(question, state.ticker)
        log("plan:\n  " + "\n  ".join(queries))
        self.sweep(state, queries)
        self.follow_up(state)
        ok = sum(not e.failed for e in state.evidence)
        log(f"collected {ok} usable pieces of evidence ({len(state.evidence) - ok} failed)")
        return state
