# my_weird_model

A collection of ML projects I'm building. Ping me for Colab help on Discord: **adnan_waasi**

## Projects

### Market research agent: `market_research_agent/`
A Hermes-style research agent that digs through the web and market data and writes a fully cited report.

- **Plan → sweep → follow-up (LangGraph):** a local fine-tuned model (via Ollama) plans the searches, runs a parallel sweep of tools, then chases leads.
- **Write:** Claude writes the final report, citing every claim as `[E#]` with a sources list.
- **Models:** `get_llm.py` builds every model from a `provider:model` spec, so new providers plug in easily.
- **Free data only:** DuckDuckGo web and news, yfinance quotes and financials, SEC EDGAR filings, page scraping, and the LSTM forecaster below.

```bash
python research.py "Is NVIDIA still a buy after the latest earnings?" --ticker NVDA
```

### Unsloth fine-tune: `unsloth_finetuning/`
Fine-tunes Qwen2.5-3B with LoRA on Hermes function-calling and finance data, then exports a GGUF for Ollama. This is the brain of the research agent.
[Open in Colab](https://colab.research.google.com/github/adnanwaasi/my_weird_model/blob/main/unsloth_finetuning/unsloth_market_researcher.ipynb) (T4 GPU)

### LSTM stock predictor: `stock_market_prediction/`
Predicts the next-day return from 30 days of price and volume features. It's tested honestly against naive baselines (no fake "look how well it predicts price" charts).

```bash
python train.py --ticker AAPL
python predict.py --ticker AAPL
```

### Anti money laundering: `anti_money_laundering/`
GNN-based money laundering detection, my 5th sem project. The data-to-graph pipeline is from another dev (he's the goat); I changed the model architecture.

## How they fit together

```
unsloth_finetuning ──(GGUF → Ollama)──► market_research_agent ◄──(lstm_forecast tool)── stock_market_prediction
                                               │
                                        Claude writes the report
```

## Ideas for next

- F1 race prediction (GNN)
- something productive

Each project folder has its own README with setup details. Nothing here is investment advice.
