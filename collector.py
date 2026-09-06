"""
Phase 1 — Collecteur de données brutes via yfinance.
Pas d'agent, pas de raisonnement : on récupère et on structure, point.
Fonctionne pour actions et ETF (les options/calls viendront en extension,
yfinance les expose via Ticker.option_chain() mais c'est un autre morceau).
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import yfinance as yf

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_ticker_data(ticker: str) -> dict:
    """Récupère prix, historique et fondamentaux pour un ticker (action ou ETF)."""
    t = yf.Ticker(ticker)
    info = t.info  # dict fondamentaux (marche pour actions ET ETF, champs différents)

    hist = t.history(period="6mo")
    history_points = [
        {"date": str(idx.date()), "close": round(row["Close"], 2)}
        for idx, row in hist.iterrows()
    ]

    quote_type = info.get("quoteType", "UNKNOWN")  # "EQUITY", "ETF", ...

    data = {
        "ticker": ticker.upper(),
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "quote_type": quote_type,
        "name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "market_cap": info.get("marketCap"),
        "currency": info.get("currency"),
    }

    if quote_type == "EQUITY":
        data["fundamentals"] = {
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "profit_margins": info.get("profitMargins"),
            "gross_margins": info.get("grossMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "debt_to_equity": info.get("debtToEquity"),
            "total_cash": info.get("totalCash"),
            "total_debt": info.get("totalDebt"),
            "dividend_yield": info.get("dividendYield"),
        }
    elif quote_type == "ETF":
        data["fundamentals"] = {
            "expense_ratio": info.get("annualReportExpenseRatio"),
            "ytd_return": info.get("ytdReturn"),
            "three_year_return": info.get("threeYearAverageReturn"),
            "total_assets": info.get("totalAssets"),
            "category": info.get("category"),
        }
    else:
        data["fundamentals"] = {}
        data["note"] = f"Type '{quote_type}' pas encore géré finement (ex: options)."

    data["price_history_6mo"] = history_points
    return data


def display_summary(data: dict) -> None:
    print(f"\n{'=' * 50}")
    print(f"{data['ticker']} — {data['name']} ({data['quote_type']})")
    print(f"{'=' * 50}")
    print(f"Prix actuel : {data['current_price']} {data['currency']}")
    print(f"Secteur     : {data['sector']} / {data['industry']}")
    if data["market_cap"]:
        print(f"Market cap  : {data['market_cap']:,}")
    print("\nFondamentaux :")
    for key, value in data["fundamentals"].items():
        print(f"  - {key}: {value}")
    print(f"\n{len(data['price_history_6mo'])} points de prix sur 6 mois récupérés.")


def save_data(data: dict) -> Path:
    out_path = DATA_DIR / f"{data['ticker']}.json"
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else input("Ticker (ex: AAPL, SPY) : ").strip()
    result = fetch_ticker_data(ticker)
    display_summary(result)
    saved_path = save_data(result)
    print(f"\nDonnées sauvegardées dans {saved_path}")