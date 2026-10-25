"""Research tools backed by free, keyless data sources.

Every tool returns a JSON string (what the LLM sees) and never raises: failures come
back as {"error": ...} so one dead source can't kill a research run.
"""

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import httpx
from langchain_core.tools import tool

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
# SEC asks automated clients to identify themselves: https://www.sec.gov/os/accessing-edgar-data
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "MarketResearchAgent/0.1 research-bot")
STOCK_PREDICTOR_DIR = Path(__file__).resolve().parent.parent / "stock_market_prediction"


def _ok(data) -> str:
    return json.dumps(data, default=str, ensure_ascii=False)


def _err(e: Exception) -> str:
    return json.dumps({"error": f"{type(e).__name__}: {e}"})


def _num(x):
    try:
        return None if x is None or x != x else round(float(x), 4)  # x != x catches NaN
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- web


@tool
def web_search(query: str, max_results: int = 8) -> str:
    """General web search (DuckDuckGo). Returns titles, URLs and snippets."""
    try:
        from ddgs import DDGS

        hits = DDGS().text(query, max_results=max_results)
        return _ok([{"title": h.get("title"), "url": h.get("href"), "snippet": h.get("body")} for h in hits])
    except Exception as e:
        return _err(e)


@tool
def news_search(query: str, max_results: int = 10) -> str:
    """Search recent news articles; returns dates, publishers, URLs and snippets."""
    try:
        from ddgs import DDGS

        hits = DDGS().news(query, max_results=max_results)
        return _ok(
            [
                {"date": h.get("date"), "source": h.get("source"), "title": h.get("title"),
                 "url": h.get("url"), "snippet": h.get("body")}
                for h in hits
            ]
        )
    except Exception as e:
        return _err(e)


@tool
def fetch_page(url: str, max_chars: int = 12000) -> str:
    """Download a web page and return its main readable text. Use on promising search results."""
    try:
        import trafilatura

        r = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=20, follow_redirects=True)
        r.raise_for_status()
        text = trafilatura.extract(r.text, include_comments=False, include_tables=True) or ""
        if not text:
            return _ok({"url": url, "error": "no readable text (paywall, JS-only page or PDF)"})
        return _ok({"url": url, "text": text[:max_chars], "truncated": len(text) > max_chars})
    except Exception as e:
        return _err(e)


# ---------------------------------------------------------------- market data (yfinance)


@tool
def stock_quote(ticker: str) -> str:
    """Company profile, price, valuation multiples, margins and analyst targets for a ticker (e.g. NVDA, RELIANCE.NS)."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        keys = [
            "longName", "sector", "industry", "country", "currency", "currentPrice", "marketCap",
            "enterpriseValue", "trailingPE", "forwardPE", "pegRatio", "priceToBook", "enterpriseToEbitda",
            "profitMargins", "operatingMargins", "returnOnEquity", "revenueGrowth", "earningsGrowth",
            "totalCash", "totalDebt", "freeCashflow", "dividendYield", "beta", "fiftyTwoWeekLow",
            "fiftyTwoWeekHigh", "recommendationKey", "targetMeanPrice", "targetLowPrice",
            "targetHighPrice", "numberOfAnalystOpinions", "fullTimeEmployees",
        ]
        out = {k: info.get(k) for k in keys if info.get(k) is not None}
        out["ticker"] = ticker.upper()
        out["businessSummary"] = (info.get("longBusinessSummary") or "")[:1500]
        out["source"] = f"https://finance.yahoo.com/quote/{ticker}"
        return _ok(out)
    except Exception as e:
        return _err(e)


@tool
def price_history(ticker: str, period: str = "1y") -> str:
    """Price performance, volatility, drawdown and moving-average trend. period: 1mo, 3mo, 6mo, 1y, 2y or 5y."""
    try:
        import numpy as np
        import yfinance as yf

        h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if h.empty:
            return _ok({"error": f"no price history for {ticker}"})
        close = h["Close"]
        rets = close.pct_change().dropna()

        def ret_over(days):
            return _num(close.iloc[-1] / close.iloc[-days - 1] - 1) if len(close) > days else None

        return _ok(
            {
                "ticker": ticker.upper(), "period": period,
                "start": str(close.index[0].date()), "end": str(close.index[-1].date()),
                "last_close": _num(close.iloc[-1]),
                "return_1m": ret_over(21), "return_3m": ret_over(63), "return_period": _num(close.iloc[-1] / close.iloc[0] - 1),
                "annualized_volatility": _num(rets.std() * np.sqrt(252)),
                "max_drawdown": _num((close / close.cummax() - 1).min()),
                "above_50d_ma": bool(close.iloc[-1] > close.rolling(50).mean().iloc[-1]),
                "above_200d_ma": bool(close.iloc[-1] > close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None,
                "avg_daily_volume": _num(h["Volume"].tail(20).mean()),
                "source": f"https://finance.yahoo.com/quote/{ticker}/history",
            }
        )
    except Exception as e:
        return _err(e)


@tool
def financials(ticker: str) -> str:
    """Annual and quarterly income statement, balance sheet and cash flow highlights for a ticker."""
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)
        wanted = {
            "income": (t.income_stmt, ["Total Revenue", "Gross Profit", "Operating Income", "Net Income", "Diluted EPS"]),
            "quarterly_income": (t.quarterly_income_stmt, ["Total Revenue", "Operating Income", "Net Income", "Diluted EPS"]),
            "balance_sheet": (t.balance_sheet, ["Total Assets", "Total Debt", "Cash And Cash Equivalents", "Stockholders Equity"]),
            "cashflow": (t.cashflow, ["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"]),
        }
        out = {"ticker": ticker.upper(), "source": f"https://finance.yahoo.com/quote/{ticker}/financials"}
        for name, (df, rows) in wanted.items():
            if df is None or df.empty:
                continue
            df = df.iloc[:, :4]  # latest 4 periods
            out[name] = {
                str(col.date()): {r: _num(df.at[r, col]) for r in rows if r in df.index}
                for col in df.columns
            }
        return _ok(out)
    except Exception as e:
        return _err(e)


# ---------------------------------------------------------------- SEC EDGAR


@lru_cache(maxsize=1)
def _sec_ticker_map() -> dict:
    r = httpx.get("https://www.sec.gov/files/company_tickers.json", headers={"User-Agent": SEC_USER_AGENT}, timeout=20)
    r.raise_for_status()
    return {v["ticker"].upper(): v for v in r.json().values()}


@tool
def sec_filings(ticker: str, forms: str = "10-K,10-Q,8-K", limit: int = 10) -> str:
    """Recent SEC filings (forms is comma-separated, e.g. '10-K,10-Q,8-K') with document URLs. US-listed companies only."""
    try:
        company = _sec_ticker_map().get(ticker.upper())
        if not company:
            return _ok({"error": f"{ticker} not found in SEC EDGAR (non-US listings are not covered)"})
        cik = int(company["cik_str"])
        r = httpx.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
                      headers={"User-Agent": SEC_USER_AGENT}, timeout=20)
        r.raise_for_status()
        recent = r.json()["filings"]["recent"]
        wanted = {f.strip().upper() for f in forms.split(",")}
        filings = []
        for form, filed, acc, doc, desc in zip(recent["form"], recent["filingDate"], recent["accessionNumber"],
                                               recent["primaryDocument"], recent["primaryDocDescription"]):
            if form.upper() in wanted:
                filings.append({
                    "form": form, "filed": filed, "description": desc,
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}",
                })
                if len(filings) >= limit:
                    break
        return _ok({"company": company["title"], "cik": cik, "filings": filings})
    except Exception as e:
        return _err(e)


# ---------------------------------------------------------------- LSTM forecast (../stock_market_prediction)


@tool
def lstm_forecast(ticker: str) -> str:
    """Next-day direction forecast from the local LSTM model. A weak statistical signal; needs a trained checkpoint."""
    try:
        if str(STOCK_PREDICTOR_DIR) not in sys.path:
            sys.path.insert(0, str(STOCK_PREDICTOR_DIR))
        from predict import forecast

        return _ok(forecast(ticker.upper()))
    except FileNotFoundError as e:
        return _ok({"error": str(e)})
    except Exception as e:
        return _err(e)


# ---------------------------------------------------------------- registry

TOOLS = [web_search, news_search, fetch_page, stock_quote, price_history, financials, sec_filings, lstm_forecast]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def call_tool(name: str, args: dict) -> str:
    """Run a tool by name; bad names or arguments come back as an error string, never raise."""
    if name not in TOOLS_BY_NAME:
        return _ok({"error": f"unknown tool {name!r}; available: {sorted(TOOLS_BY_NAME)}"})
    try:
        return TOOLS_BY_NAME[name].invoke(args or {})
    except Exception as e:  # e.g. argument validation errors from the model
        return _err(e)
