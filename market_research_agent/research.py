"""Market research agent CLI.

Examples:
    python research.py "Is NVIDIA still a buy after the latest earnings?" --ticker NVDA
    python research.py "Who is winning the Indian EV two-wheeler market?" --researcher ollama:qwen2.5:3b
    python research.py "AMD vs Intel in datacenter CPUs" --writer ollama:market-researcher
"""

import argparse
import re
from datetime import date
from pathlib import Path

from get_llm import spec_for
from graph import build_graph, initial_state, is_failed, label, log

REPORTS = Path(__file__).parent / "reports"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question")
    p.add_argument("--ticker", help="focus company ticker; enables quote, financials, filings and LSTM tools up front")
    p.add_argument("--researcher", default=spec_for("researcher"),
                   help="LLM spec for planning and tool calls (default: %(default)s)")
    p.add_argument("--writer", default=spec_for("writer"),
                   help="LLM spec for the final report (default: %(default)s); falls back to the researcher")
    p.add_argument("--max-steps", type=int, default=8, help="max tool-calling rounds after the initial sweep")
    args = p.parse_args()

    graph = build_graph(args.researcher, args.writer, args.max_steps)
    final = graph.invoke(
        initial_state(args.question, args.ticker),
        config={"recursion_limit": 2 * args.max_steps + 10},
    )

    REPORTS.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", (args.ticker or args.question).lower()).strip("-")[:50]
    path = REPORTS / f"{date.today()}-{slug}.md"
    appendix = "\n\n---\n## Evidence log\n\n" + "\n".join(
        f"- {label(e)}" for e in final["evidence"] if not is_failed(e)
    )
    path.write_text(final["report"] + appendix)
    log(f"\nsaved {path}")


if __name__ == "__main__":
    main()
