"""
Phase 3 — Agent Analyste Sentiment.
Récupère les news récentes via Tavily et demande à GLM une lecture du ton
ambiant + ce qui pourrait surprendre le marché (pas juste "positif/négatif").

Pré-requis :
  pip install ollama tavily-python python-dotenv
  .env dans poc/ avec : TAVILY_API_KEY=tvly-xxxx
  (si tu as nommé la variable autrement, change la ligne os.getenv ci-dessous)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import ollama
from dotenv import load_dotenv
from tavily import TavilyClient

# Charge le .env qui est dans poc/ (à côté de ce fichier)
load_dotenv(Path(__file__).parent / ".env")

MODEL_NAME = "glm-4.7-flash:latest"
DATA_DIR = Path(__file__).parent / "data"

SYSTEM_PROMPT = """Tu es un analyste sell-side spécialisé sentiment de marché.
Tu reçois des news récentes sur une société et tu produis une lecture nuancée
du ton ambiant — jamais un verdict "achète/vends".

Ta note doit contenir, en français concis :
1. TON GÉNÉRAL — dominante des news (positive / neutre / négative / mixte),
   avec une phrase de justification.
2. NARRATIF DOMINANT — quel est le récit que le marché raconte sur ce titre
   en ce moment ? (ex: "IA générative", "guerre des prix", "changement de CEO"...)
3. DÉJÀ DANS LE PRIX — quels éléments semblent déjà largement anticipés
   par le marché (donc peu susceptibles de faire bouger le cours).
4. POTENTIELS CATALYSEURS DE SURPRISE — événements ou signaux qui, s'ils se
   confirment, pourraient surprendre positivement ou négativement.
5. SOURCES CITÉES — liste des articles que tu as utilisés (titre + URL).

Sois prudent : le sentiment est bruité. Signale explicitement si les news sont
trop peu nombreuses ou trop anciennes pour tirer une conclusion fiable."""


def fetch_news(ticker: str, company_name: str | None = None, max_results: int = 8) -> list[dict]:
    """Récupère les news récentes via Tavily (7 derniers jours)."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY absente. Vérifie ton fichier poc/.env "
            "(ou change le nom de la variable dans le code)."
        )
    client = TavilyClient(api_key=api_key)

    query = f"{ticker} stock news"
    if company_name:
        query = f"{company_name} ({ticker}) latest news"

    result = client.search(
        query=query,
        search_depth="advanced",
        topic="news",
        days=7,
        max_results=max_results,
    )
    return result.get("results", [])


def load_ticker_data(ticker: str) -> dict:
    """On récupère le nom de la société depuis le JSON du collecteur si dispo."""
    path = DATA_DIR / f"{ticker.upper()}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def build_user_prompt(ticker: str, news: list[dict]) -> str:
    if not news:
        return f"Aucune news trouvée pour {ticker} sur les 7 derniers jours."

    formatted = []
    for i, article in enumerate(news, 1):
        formatted.append(
            f"[{i}] {article.get('title', 'Sans titre')}\n"
            f"    URL: {article.get('url')}\n"
            f"    Publié: {article.get('published_date', 'date inconnue')}\n"
            f"    Extrait: {article.get('content', '')[:500]}\n"
        )

    return (
        f"Voici les {len(news)} news les plus récentes sur {ticker} :\n\n"
        + "\n".join(formatted)
        + "\n\nProduis ta note d'analyse sentiment."
    )


def analyze(ticker: str) -> tuple[str, list[dict]]:
    data = load_ticker_data(ticker)
    company_name = data.get("name")

    print(f"→ Récupération des news pour {ticker} via Tavily...")
    news = fetch_news(ticker, company_name)
    print(f"  {len(news)} articles récupérés.")

    user_prompt = build_user_prompt(ticker, news)

    print(f"→ Envoi à {MODEL_NAME} pour analyse sentiment...\n")
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.4},  # un poil plus haut : c'est plus interprétatif
    )
    return response["message"]["content"], news


def save_output(ticker: str, analysis: str, news: list[dict]) -> None:
    ticker = ticker.upper()

    # L'analyse en markdown
    md_path = DATA_DIR / f"{ticker}_sentiment.md"
    md_path.write_text(analysis, encoding="utf-8")

    # Les news brutes en JSON (traçabilité — utile si le Gérant veut vérifier)
    news_path = DATA_DIR / f"{ticker}_news.json"
    news_path.write_text(
        json.dumps(
            {"fetched_at": datetime.now().isoformat(timespec="seconds"), "articles": news},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n→ Analyse sauvegardée dans {md_path}")
    print(f"→ News brutes sauvegardées dans {news_path}")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else input("Ticker à analyser : ").strip()
    analysis, news = analyze(ticker)

    print("=" * 60)
    print(f"NOTE SENTIMENT — {ticker.upper()}")
    print("=" * 60)
    print(analysis)

    save_output(ticker, analysis, news)