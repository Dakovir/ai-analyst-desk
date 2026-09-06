"""
Phase 4bis — Agent Analyste Macro (v2 — avec lookback historique).
Produit une note de contexte macro-économique INDÉPENDANTE du ticker.

NOUVEAUTÉ v2 : utilise history_manager pour comparer les indicateurs du jour
avec ceux du run précédent. Le LLM reçoit les deltas et peut commenter les
tendances ("le VIX a bondi de 18%", "l'or poursuit sa hausse pour la 3e
séance") au lieu de juste décrire un instantané.

Sortie datée : data/macro_brief_YYYYMMDD.md + data/macro_data_YYYYMMDD.json
Le Gérant (manager_agent.py) lit la note du jour la plus récente.

Pré-requis : pip install ollama tavily-python python-dotenv yfinance
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import ollama
import yfinance as yf
from dotenv import load_dotenv
from tavily import TavilyClient

# Import du gestionnaire d'historique
from history_manager import compute_macro_deltas, format_deltas_for_prompt

load_dotenv(Path(__file__).parent / ".env")

MODEL_NAME = "glm-4.7-flash:latest"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Les symboles à suivre, groupés par catégorie.
# ---------------------------------------------------------------------------
MACRO_SYMBOLS = {
    "Indices actions par zone": {
        "Amérique — S&P 500": "^GSPC",
        "Amérique — Nasdaq": "^IXIC",
        "LATAM — Bovespa (Brésil)": "^BVSP",
        "LATAM — IPC (Mexique)": "^MXX",
        "Asie — Nikkei 225 (Japon)": "^N225",
        "Asie — Hang Seng (Hong Kong)": "^HSI",
        "Asie — Shanghai Composite": "000001.SS",
        "Europe — Euro Stoxx 50": "^STOXX50E",
        "Europe — CAC 40": "^FCHI",
        "Europe — DAX": "^GDAXI",
        "Afrique — JSE Top 40 (Afrique du Sud)": "^JN0U.JO",
    },
    "Taux souverains US": {
        "US 3 mois": "^IRX",
        "US 5 ans": "^FVX",
        "US 10 ans": "^TNX",
        "US 30 ans": "^TYX",
    },
    "Taux de change": {
        "EUR/USD": "EURUSD=X",
        "USD/JPY": "JPY=X",
        "GBP/USD": "GBPUSD=X",
        "Dollar Index (DXY)": "DX-Y.NYB",
    },
    "Énergie": {
        "Pétrole WTI": "CL=F",
        "Pétrole Brent": "BZ=F",
        "Gaz naturel": "NG=F",
    },
    "Matières premières": {
        "Or": "GC=F",
        "Argent": "SI=F",
        "Cuivre": "HG=F",
    },
    "Indices de la peur (volatilité)": {
        "VIX (S&P 500)": "^VIX",
        "VXN (Nasdaq)": "^VXN",
    },
}

# ---------------------------------------------------------------------------
# 2. Requêtes Tavily
# ---------------------------------------------------------------------------
MACRO_QUERIES = {
    "Politique monétaire US (Fed)": "Federal Reserve interest rate decision monetary policy outlook",
    "Politique monétaire Europe (BCE)": "European Central Bank ECB interest rates eurozone economy",
    "Asie (BoJ, Chine)": "Bank of Japan yen China economy stimulus markets",
    "Marchés émergents (LATAM, Afrique)": "emerging markets Latin America Africa economy currencies",
    "Sentiment de foule / positionnement": "stock market investor sentiment fear greed euphoria positioning overbought",
}


def fetch_indicators() -> dict:
    """Récupère prix + variation sur la dernière séance pour chaque symbole."""
    results = {}
    total = sum(len(v) for v in MACRO_SYMBOLS.values())
    done = 0

    for category, symbols in MACRO_SYMBOLS.items():
        results[category] = {}
        for label, symbol in symbols.items():
            done += 1
            print(f"  [{done}/{total}] {label} ({symbol})...", end=" ")
            try:
                hist = yf.Ticker(symbol).history(period="5d")
                if len(hist) >= 2:
                    last = float(hist["Close"].iloc[-1])
                    prev = float(hist["Close"].iloc[-2])
                    change_pct = (last - prev) / prev * 100 if prev else 0.0
                    results[category][label] = {
                        "value": round(last, 2),
                        "change_pct": round(change_pct, 2),
                    }
                    print(f"{last:.2f} ({change_pct:+.2f}%)")
                else:
                    results[category][label] = {
                        "value": None,
                        "note": "données indisponibles",
                    }
                    print("indispo")
            except Exception as e:
                results[category][label] = {
                    "value": None,
                    "note": f"erreur: {e}",
                }
                print("erreur")
    return results


def fetch_macro_news() -> dict:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY absente. Vérifie ton fichier poc/.env")
    client = TavilyClient(api_key=api_key)

    news = {}
    for theme, query in MACRO_QUERIES.items():
        print(f"  News : {theme}...")
        try:
            res = client.search(
                query=query,
                search_depth="advanced",
                topic="news",
                days=7,
                max_results=4,
            )
            news[theme] = [
                {
                    "title": a.get("title"),
                    "url": a.get("url"),
                    "content": (a.get("content") or "")[:400],
                }
                for a in res.get("results", [])
            ]
        except Exception as e:
            news[theme] = [{"error": str(e)}]
    return news


SYSTEM_PROMPT = """Tu es un stratégiste macro-économique senior dans une société
de gestion. Tu reçois des chiffres de marché, des news récentes, ET L'ÉVOLUTION
PAR RAPPORT AU RUN PRÉCÉDENT (quand disponible).

IMPORTANT sur l'historique : quand des deltas sont fournis, UTILISE-LES.
Ne te contente pas de décrire un instantané — commente les TENDANCES.
Exemples : "le VIX a bondi de +18% depuis la dernière lecture, signalant un
regain de nervosité", "l'or poursuit sa hausse (+2.3%) dans un contexte de
fuite vers la qualité", "le 10 ans US se détend de -15 bp, le marché price
un pivot dovish de la Fed".

Ta note doit être en français, structurée dans cet ordre exact :

## RÉGIME DE MARCHÉ
Risk-on ou risk-off ? Justifie en 2-3 phrases avec les chiffres ET leur
évolution récente (VIX, taux, or, dollar).

## PAR ZONE GÉOGRAPHIQUE
Un paragraphe court par zone, basé sur indices + news + tendance :
- **Amérique du Nord** (US)
- **Europe**
- **Asie** (Japon, Chine)
- **Amérique latine (LATAM)**
- **Afrique**
Si tu manques de données fiables, dis-le.

## POLITIQUE MONÉTAIRE ET TAUX
Fed, BCE, BoJ. Sens des taux souverains US (courbe qui se pentifie /
s'aplatit ?). Évolution par rapport au run précédent.

## DEVISES
Dollar (DXY), EUR/USD, USD/JPY. Force/faiblesse du dollar et implications.

## ÉNERGIE ET MATIÈRES PREMIÈRES
Pétrole, gaz, or, cuivre. L'or et le cuivre sont des signaux.
Commente les MOUVEMENTS, pas juste les niveaux.

## INDICES DE LA PEUR
VIX : niveau + évolution. Complaisance / normal / stress.

## SENTIMENT DE FOULE
Le marché est-il EUPHORIQUE (trop de monde positif → prudence contrariante)
ou en CAPITULATION (pessimisme excessif → opportunité potentielle) ?

## SYNTHÈSE TRANSVERSALE
3-4 phrases : fil rouge macro du moment, quels types d'actifs ou secteurs
ce contexte favorise ou pénalise (sans nommer de titre).

Reste factuel, cite les chiffres, signale ce qui manque."""


def build_user_prompt(indicators: dict, news: dict,
                      deltas_text: str) -> str:
    return (
        "## CHIFFRES DE MARCHÉ (variation = dernière séance)\n"
        f"```json\n{json.dumps(indicators, indent=2, ensure_ascii=False)}\n```\n\n"
        f"## ÉVOLUTION PAR RAPPORT AU RUN PRÉCÉDENT\n{deltas_text}\n\n"
        "## NEWS MACRO RÉCENTES (7 derniers jours)\n"
        f"```json\n{json.dumps(news, indent=2, ensure_ascii=False)}\n```\n\n"
        "Rédige maintenant ta note de contexte macro en suivant strictement "
        "la structure et en commentant les tendances."
    )


def generate_brief() -> tuple[str, dict, dict]:
    print("→ Récupération des chiffres de marché via yfinance...\n")
    indicators = fetch_indicators()

    # NOUVEAUTÉ : calcul des deltas par rapport au run précédent
    print("\n→ Comparaison avec l'historique...")
    deltas = compute_macro_deltas(indicators)
    deltas_text = format_deltas_for_prompt(deltas)
    if deltas:
        compared_to = deltas.get("_compared_to", "?")
        print(f"  Comparaison avec les données du {compared_to}")
    else:
        print("  Pas d'historique disponible (premier run)")

    print("\n→ Récupération des news macro via Tavily...")
    news = fetch_macro_news()

    print(f"\n→ Envoi à {MODEL_NAME} pour la synthèse macro...\n")
    date_ctx = f"Date du jour : {datetime.now().strftime('%d/%m/%Y %H:%M')}.\n\n"
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": date_ctx + SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(
                indicators, news, deltas_text
            )},
        ],
        options={"temperature": 0.3},
    )
    return response["message"]["content"], indicators, news


def save_brief(brief: str, indicators: dict, news: dict) -> Path:
    day = datetime.now().strftime("%Y%m%d")
    header = (
        f"# Note de contexte macro — {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
        f"*Générée par l'agent macro v2 (modèle : {MODEL_NAME})*\n\n---\n\n"
    )
    brief_path = DATA_DIR / f"macro_brief_{day}.md"
    brief_path.write_text(header + brief, encoding="utf-8")

    raw_path = DATA_DIR / f"macro_data_{day}.json"
    raw_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "indicators": indicators,
                "news": news,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return brief_path


if __name__ == "__main__":
    brief, indicators, news = generate_brief()

    print("=" * 60)
    print("NOTE DE CONTEXTE MACRO")
    print("=" * 60)
    print(brief)

    saved = save_brief(brief, indicators, news)
    print(f"\n→ Note macro sauvegardée dans {saved}")