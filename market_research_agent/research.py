"""Market research agent CLI.

Examples:
    python research.py "Is NVIDIA still a buy after the latest earnings?" --ticker NVDA
    python research.py "How big is the Indian EV two-wheeler market and who is winning?"
    python research.py "Compare AMD vs Intel in datacenter CPUs" --writer local
"""

import argparse
import re
from datetime import date
from pathlib import Path

from agent import Researcher, log
from writer import ClaudeWriter, LocalWriter

REPORTS = Path(__file__).parent / "reports"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question")
    p.add_argument("--ticker", help="focus company ticker; enables quote, financials, filings and LSTM tools up front")
    p.add_argument("--model", default="market-researcher", help="Ollama model for planning and tool calls")
    p.add_argument("--ollama-host", default=None, help="defaults to OLLAMA_HOST or http://localhost:11434")
    p.add_argument("--writer", choices=["auto", "claude", "local"], default="auto",
                   help="auto = Claude if credentials are configured, otherwise the local model")
    p.add_argument("--claude-model", default="claude-opus-5-5")
    p.add_argument("--max-steps", type=int, default=8, help="max tool-calling rounds after the initial sweep")
    args = p.parse_args()

    researcher = Researcher(args.model, host=args.ollama_host, max_steps=args.max_steps)
    state = researcher.research(args.question, args.ticker)

    report = None
    if args.writer in ("auto", "claude"):
        try:
            report = ClaudeWriter(args.claude_model).write(state)
        except Exception as e:
            if args.writer == "claude":
                raise
            log(f"Claude writer unavailable ({type(e).__name__}: {e}); falling back to local model")
    if report is None:
        report = LocalWriter(researcher).write(state)

    REPORTS.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", (args.ticker or args.question).lower()).strip("-")[:50]
    path = REPORTS / f"{date.today()}-{slug}.md"
    appendix = "\n\n---\n## Evidence log\n\n" + "\n".join(f"- {e.label()}" for e in state.evidence if not e.failed)
    path.write_text(report + appendix)
    log(f"\nsaved {path}")


if __name__ == "__main__":
    main()
