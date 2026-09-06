# Multi-Agent Investment Analyst

A multi-agent financial analysis system that simulates an analyst desk. Nine specialized agents each produce a note from a different angle, then a tenth — the Portfolio Manager — cross-references their analyses to write a structured synthesis with bull thesis, bear thesis, and watchpoints.

**The system never recommends buying or selling.** This is a design principle, not just a disclaimer.

---

## Design Principles

- **Analysis, not decisions.** An LLM generates plausible text — it doesn't predict markets. The system informs reasoning; it never replaces human judgment.
- **Two theses, always.** Every final note presents the strongest bull AND bear arguments, backed by numbers. The reader decides.
- **Visible sources.** Raw data (JSON) is stored alongside analyses (Markdown) so everything is verifiable.
- **Public data only.** No brokerage connections, no order execution, no API keys in plaintext.

---

## Architecture

```
         ┌──────────────────────────────────────────────────────┐
         │                    WATCHLIST                         │
         │              (AAPL, MSFT, NVDA…)                    │
         └────────────────────┬─────────────────────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │     Collector     │  yfinance: prices, ratios,
                    │   (collector.py)  │  financial statements, OHLCV
                    └────────┬──────────┘
                             │
              ┌──────────────┼──────────────────────────┐
              │              │                          │
              ▼              ▼                          ▼
     ┌────────────┐  ┌──────────────┐  ┌──────────────────────────┐
     │Fundamental │  │  Sentiment   │  │  Optional agents         │
     │ (required) │  │  (required)  │  │                          │
     │            │  │  Tavily news │  │  • Balance sheet          │
     │ Ratios,    │  │  Tone, story │  │  • Technical (momentum)   │
     │ valuation  │  │  catalysts   │  │  • Sector cycle           │
     └─────┬──────┘  └──────┬───────┘  │  • Peers (benchmarking)   │
           │                │          │  • Earnings call           │
           │                │          └────────────┬──────────────┘
           │                │                       │
           │    ┌───────────┼───────────────────────┘
           │    │           │
           │    │    ┌──────┴──────────────────────┐
           │    │    │   Contextual agents         │
           │    │    │   (shared, not per-ticker)   │
           │    │    │                              │
           │    │    │  • Macro (indices, rates,    │
           │    │    │    currencies, commodities)   │
           │    │    │  • Geopolitical (conflicts,  │
           │    │    │    tariffs, mega-trends)      │
           │    │    └──────┬──────────────────────┘
           │    │           │
           ▼    ▼           ▼
     ┌─────────────────────────────┐
     │    PORTFOLIO MANAGER        │
     │     (manager_agent.py)      │
     │                             │
     │  Cross-references all 9     │
     │  analyses + history         │
     │                             │
     │  → Bull thesis              │
     │  → Bear thesis              │
     │  → Contradictions           │
     │  → Watchpoints              │
     └─────────────────────────────┘
```

### The 9 Analysts + Portfolio Manager

| # | Agent | File | Role | Required |
|---|-------|------|------|:--------:|
| 1 | Collector | `collector.py` | Fetches prices, ratios, financial statements via yfinance | ✓ |
| 2 | Fundamental | `fundamental_agent.py` | Valuation, profitability, balance sheet health | ✓ |
| 3 | Sentiment | `sentiment_agent.py` | News tone, dominant narrative, surprise catalysts | ✓ |
| 4 | Balance Sheet | `balance_sheet_agent.py` | Multi-year trajectory: revenue, margins, debt, FCF | |
| 5 | Technical | `technical_agent.py` | Price action, momentum, key levels | |
| 6 | Cycle | `cycle_agent.py` | Position within the sector cycle | |
| 7 | Peers | `peers_agent.py` | Benchmarking vs sector competitors | |
| 8 | Macro | `macro_agent.py` | Indices, rates, currencies, commodities (shared) | |
| 9 | Geopolitical | `geopolitical_agent.py` | Conflicts, tariffs, national strategies (shared, TTL cache) | |
| 10 | Earnings | `earnings_agent.py` | Earnings call transcripts, management guidance | |
| — | **Manager** | `manager_agent.py` | Final synthesis: cross-references all 9 notes, doesn't just summarize | |

Agents 1–3 are required (the pipeline stops if they fail). Agents 4–10 are optional — the Manager adapts its synthesis based on what's available.

---

## Key Features

- **History & trends**: `history_manager.py` compares data across runs. Agents don't just describe a snapshot — they comment on evolution ("VIX jumped +18%", "margins compressing for the 3rd consecutive quarter").
- **Smart caching**: the geopolitical agent uses a per-category TTL (7 days for conflicts, 180 days for national strategies). The earnings agent caches by quarter.
- **Temporal awareness**: every agent receives today's date in its system prompt. The earnings agent detects offset fiscal years. The sentiment agent weighs conclusions by article freshness.
- **Batch via watchlist**: `update.py` runs the pipeline across multiple tickers, sharing common steps (macro, geopolitical) across all of them.

---

## Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running
- An LLM model pulled in Ollama (default: `glm-4.7-flash`)
- A [Tavily](https://tavily.com/) API key (free tier, for news search)

### Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install -r requirements.txt

# Pull the LLM model
ollama pull glm-4.7-flash

# Configure the Tavily API key
# Create a .env file at the project root:
echo "TAVILY_API_KEY=tvly-your-key-here" > .env
```

---

## Usage

### Full analysis of a single ticker

```bash
python run_pipeline.py AAPL
```

The pipeline runs all 11 steps in order and produces a synthesis note in `data/`.

### Pipeline options

```bash
# Skip shared steps (if already done today)
python run_pipeline.py AAPL --quick

# Skip only macro
python run_pipeline.py AAPL --skip-macro

# Re-run only the synthesis (if analyses are up to date)
python run_pipeline.py AAPL --only-synthesis
```

### Batch (multiple tickers)

```bash
# Using the watchlist
python update.py

# From the command line
python update.py AAPL MSFT NVDA

# Quick mode (skip macro/geo/earnings)
python update.py --quick
```

The watchlist is configured in `watchlist.txt` (one ticker per line, `#` for comments).

### Individual agents

Each agent can run standalone:

```bash
python collector.py AAPL           # Raw data
python fundamental_agent.py AAPL   # Fundamental analysis
python sentiment_agent.py AAPL     # Sentiment analysis
python macro_agent.py              # Macro note (no ticker needed)
python geopolitical_agent.py       # Geopolitical note
python earnings_agent.py AAPL      # Earnings call analysis
```

---

## Tech Stack

| Component | Role |
|-----------|------|
| **Python** | Core language |
| **Ollama** | Local LLM inference (no paid cloud API) |
| **GLM-4** | Default model (`glm-4.7-flash`) |
| **yfinance** | Free market data (prices, ratios, financial statements) |
| **Tavily** | News and article search (API, free tier) |

### Multi-model angle

The architecture is designed for model comparison. Each agent has a `MODEL_NAME` constant — you can assign a different model per agent (GLM for structuring, DeepSeek for fundamental reasoning, etc.) and observe where analyses diverge. This is one of the project's distinctive features: measuring which model performs best on which type of analytical task.

---

## File Structure

```
.
├── collector.py              # Data collection (yfinance)
├── fundamental_agent.py      # Fundamental analysis
├── balance_sheet_agent.py    # Balance sheet & financial statements
├── technical_agent.py        # Technical analysis
├── cycle_agent.py            # Sector cycle analysis
├── peers_agent.py            # Peer comparison
├── sentiment_agent.py        # Sentiment analysis (Tavily)
├── macro_agent.py            # Macroeconomic context
├── geopolitical_agent.py     # Geopolitical watch (TTL cache)
├── earnings_agent.py         # Earnings call analysis
├── manager_agent.py          # Portfolio Manager / Synthesizer
├── history_manager.py        # Cross-run comparison
├── run_pipeline.py           # Single-ticker orchestrator
├── update.py                 # Batch orchestrator (watchlist)
├── watchlist.txt             # Tickers to track
├── requirements.txt          # Python dependencies
├── .env                      # Tavily API key (NOT versioned)
└── data/                     # Agent outputs (NOT versioned)
    ├── AAPL.json             # Raw collector data
    ├── AAPL_fundamental.md   # Fundamental note
    ├── AAPL_sentiment.md     # Sentiment note
    ├── AAPL_synthesis_*.md   # Synthesis notes (timestamped)
    ├── macro_brief_*.md      # Macro notes (dated)
    └── geopolitical/         # Geopolitical notes + caches
```

---

## Disclaimer

This project is an automated analysis tool for educational and exploratory purposes. The notes produced by the system **do not constitute investment advice**. Analyses are generated by language models from public data and may contain errors, questionable interpretations, or outdated information.

All investment decisions are the sole responsibility of the reader.

---

## License

MIT
