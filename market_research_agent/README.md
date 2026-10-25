# Market research agent

A Hermes-style tool-using research agent built with **LangChain + LangGraph** on free data sources.

```
START -> plan -> sweep -> researcher <-> tools
                              |
                              v
                            write -> END
```

| Node | LLM | What happens |
|---|---|---|
| `plan` | researcher | Writes 5-7 diverse search queries (structured output) |
| `sweep` | none | Runs these tool calls in parallel: quote, price history, financials, SEC filings, news, LSTM forecast, plus the planned searches |
| `researcher` ⇄ `tools` | researcher | Agentic loop that opens the best URLs and fills gaps (`--max-steps`) |
| `write` | writer | Writes a structured report; every claim cites an evidence id `[E#]` |

Each tool result is stored as numbered evidence. The report must cite it and must list the gaps it couldn't fill.

## Files

| File | What it is |
|---|---|
| `get_llm.py` | **The only place models are created.** `get_llm("writer")` or `get_llm("ollama:qwen2.5:3b")` |
| `graph.py` | The LangGraph state, nodes and wiring |
| `tools.py` | LangChain `@tool`s: DuckDuckGo web/news, page fetch, yfinance, SEC EDGAR, LSTM forecast |
| `prompts.py` | All prompts |
| `research.py` | CLI |

## Choosing models: `get_llm.py`

Models are written as `provider:model` specs:

| Role | Default | Override |
|---|---|---|
| researcher | `ollama:market-researcher` (your Unsloth fine-tune) | `RESEARCHER_LLM=...` or `--researcher ...` |
| writer | `anthropic:claude-opus-5-5` | `WRITER_LLM=...` or `--writer ...` |

If the writer fails (for example, no API key), the researcher model writes the report instead.

**Adding a provider** (OpenAI, Groq, Gemini, ...): write a small builder function in `get_llm.py`
that returns a LangChain chat model, then add it to `PROVIDERS`. **Adding a role** (e.g. a fact-checker):
add it to `ROLE_DEFAULTS`, then call `get_llm("fact_checker")` in a new graph node.

## Setup

```bash
pip install -r requirements.txt
pip install -r ../stock_market_prediction/requirements.txt   # optional, for lstm_forecast

ollama pull qwen2.5:3b                    # stand-in until your fine-tune is ready
export ANTHROPIC_API_KEY=...              # optional; enables the Claude writer
export SEC_USER_AGENT="YourName your@email.com"   # SEC asks bots to identify themselves
```

## Use

```bash
python research.py "Is NVIDIA still a buy after the latest earnings?" --ticker NVDA
python research.py "Who is winning the Indian EV two-wheeler market?" --researcher ollama:qwen2.5:3b
python research.py "AMD vs Intel in datacenter CPUs" --writer ollama:market-researcher   # fully local
python get_llm.py ollama:qwen2.5:3b       # quick check that a model spec works
```

Reports are saved to `reports/YYYY-MM-DD-<slug>.md` with an evidence log appended. Not investment advice.
