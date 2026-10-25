"""Turn collected evidence into the final report.

ClaudeWriter is the default "finisher" in the hybrid setup; LocalWriter is the fallback
when no Anthropic credentials are configured (or you want everything offline).
"""

from datetime import date

from agent import ResearchState, log

REPORT_INSTRUCTIONS = """You are a senior buy-side research analyst writing for a demanding reader.
Write a market research report in Markdown that answers the research question using ONLY the
evidence provided. Cite evidence inline with its id, e.g. "revenue grew 94% YoY [E3]". Never
invent numbers; if something important is missing or sources conflict, say so explicitly.

Structure:
1. **Bottom line**: 3-5 bullet answer to the question, with the single most important number.
2. **Snapshot**: a table of key metrics (price, market cap, valuation multiples, growth, margins,
   balance sheet) when a company is involved.
3. **What's happening now**: recent developments and catalysts, dated.
4. **Business and competitive position**: moat, market share, competitors.
5. **Financial trajectory**: trends across recent periods, not just the latest figure.
6. **Bull case / Bear case**: concrete and falsifiable, each point tied to evidence.
7. **Key risks**.
8. **Valuation context**: versus history, peers and analyst targets where evidence allows.
9. **Quant signal**: the LSTM forecast if present, framed as a weak statistical signal.
10. **What to watch**: upcoming dates and metrics that would change the thesis.
11. **Evidence gaps**.
12. **Sources**: list each cited evidence id with its URL(s).

Be dense and specific; no filler. End with: "Not investment advice."
"""


def _evidence_block(state: ResearchState, max_chars: int) -> str:
    parts = [f"<evidence id=\"{e.id}\" call=\"{e.label()}\">\n{e.result[:max_chars]}\n</evidence>"
             for e in state.evidence if not e.failed]
    return "\n\n".join(parts)


def _user_prompt(state: ResearchState, max_chars: int) -> str:
    subject = f" (ticker: {state.ticker})" if state.ticker else ""
    return (
        f"Today is {date.today()}.\nResearch question: {state.question}{subject}\n\n"
        f"{_evidence_block(state, max_chars)}\n\nWrite the report now."
    )


class ClaudeWriter:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high"):
        import anthropic

        self.client = anthropic.Anthropic()  # ANTHROPIC_API_KEY or an `ant auth login` profile
        self.model = model
        self.effort = effort

    def write(self, state: ResearchState) -> str:
        log(f"writing report with {self.model}...")
        # Server-side fallback: if a safety classifier declines, the API retries on its
        # recommended fallback model inside the same call.
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=64000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": self.effort},
            system=REPORT_INSTRUCTIONS,
            messages=[{"role": "user", "content": _user_prompt(state, max_chars=12000)}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            message = stream.get_final_message()
        print()
        if message.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined to write the report: {message.stop_details}")
        if message.stop_reason == "max_tokens":
            log("warning: report hit max_tokens and may be truncated")
        return "".join(b.text for b in message.content if b.type == "text")


class LocalWriter:
    def __init__(self, researcher):
        self.researcher = researcher

    def write(self, state: ResearchState) -> str:
        log(f"writing report with local model {self.researcher.model}...")
        # Small context window: give the local model shorter excerpts.
        stream = self.researcher.client.chat(
            model=self.researcher.model,
            messages=[
                {"role": "system", "content": REPORT_INSTRUCTIONS},
                {"role": "user", "content": _user_prompt(state, max_chars=1200)},
            ],
            options={"num_ctx": 32768},
            stream=True,
        )
        chunks = []
        for chunk in stream:
            piece = chunk.message.content or ""
            print(piece, end="", flush=True)
            chunks.append(piece)
        print()
        return "".join(chunks)
