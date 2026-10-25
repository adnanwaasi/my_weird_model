# Market research agent

A Hermes-style tool-using research agent built on free data sources. It uses a hybrid of two models:

| Stage | Model | What happens |
|---|---|---|
| Plan | local fine-tune (Ollama) | Writes 5-7 diverse search queries |
| Sweep | (no LLM) | Runs these tool calls in parallel: quote, price history, financials, SEC filings, news, LSTM forecast, plus the planned searches |
| Follow-up | local fine-tune | Agentic loop that opens the best URLs and fills gaps (`--max-steps`) |
| Write | Claude (`claude-opus-5-5`), with local fallback | Writes a structured report where every claim cites an evidence id `[E#]` |

Each tool result is stored as numbered evidence. The report must cite it and must list the gaps it couldn't fill.

**Tools (all free, no API keys):** DuckDuckGo web and news search, page fetching via trafilatura, yfinance
(quote, history, financials), SEC EDGAR filings, and the LSTM forecaster in `../stock_market_prediction`.

## Setup

```bash
pip install -r requirements.txt
pip install -r ../stock_market_prediction/requirements.txt   # optional, for lstm_forecast

# Local model: your Unsloth fine-tune (see ../unsloth_finetuning), or any Qwen2.5 model for now:
ollama pull qwen2.5:3b

# Claude writer (optional; without it the local model writes the report):
export ANTHROPIC_API_KEY=...        # or: ant auth login
export SEC_USER_AGENT="YourName your@email.com"   # SEC asks bots to identify themselves
```

## Use

```bash
python research.py "Is NVIDIA still a buy after the latest earnings?" --ticker NVDA
python research.py "Who is winning the Indian EV two-wheeler market?" --model qwen2.5:3b
python research.py "AMD vs Intel in datacenter CPUs" --writer local
```

Reports are saved to `reports/YYYY-MM-DD-<slug>.md` with an evidence log appended. Not investment advice.
