"""
Phase 2 — Agent Analyste Fondamental.
Lit le JSON produit par collector.py et demande à GLM (via Ollama)
une lecture argumentée des chiffres — pas un verdict, une analyse.

Pré-requis : pip install ollama
Le collector.py doit avoir tourné avant (data/<TICKER>.json doit exister).
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import ollama

from history_manager import (
    get_previous_fundamental,
    compute_fundamental_deltas,
    format_fundamental_deltas,
)

MODEL_NAME = "glm-4.7-flash:latest"
DATA_DIR = Path(__file__).parent / "data"

SYSTEM_PROMPT = """Tu es un analyste fondamental sell-side expérimenté.
Tu reçois les chiffres bruts d'une société ou d'un ETF et tu produis une
lecture argumentée — jamais un verdict "achète/vends".

Quand un historique de comparaison est fourni (évolution depuis la collecte
précédente), UTILISE-LE. Commente les changements : "le P/E a augmenté de
25x à 28x, signe d'une revalorisation", "la dette a baissé de 15%, le bilan
se renforce". C'est ce suivi dans le temps qui fait la valeur de l'analyse.

Ta note doit contenir, en français concis :
1. VALORISATION — le titre est-il cher, dans la moyenne, décoté ? Compare
   les ratios aux normes du secteur quand tu les connais. COMMENTE
   L'ÉVOLUTION si l'historique est disponible.
2. RENTABILITÉ — les marges sont-elles solides ? En amélioration ?
3. SOLIDITÉ FINANCIÈRE — bilan sain ? Dette maîtrisée par rapport à la trésorerie ?
4. DYNAMIQUE — croissance du chiffre d'affaires, momentum. Si des données
   précédentes existent, commente la TENDANCE.
5. POINTS DE VIGILANCE — ce qui interpelle, ce qu'il faut creuser.

Reste factuel, cite les chiffres que tu utilises, et signale explicitement
quand un chiffre manque ou paraît aberrant. Pas de conclusion "à acheter"."""


def load_ticker_data(ticker: str) -> dict:
    path = DATA_DIR / f"{ticker.upper()}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Pas de données pour {ticker}. Lance d'abord : python collector.py {ticker}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_user_prompt(data: dict) -> str:
    """On envoie à l'agent uniquement ce dont il a besoin — pas les 128 points de prix."""
    ticker = data["ticker"]
    payload = {
        "ticker": ticker,
        "name": data["name"],
        "quote_type": data["quote_type"],
        "sector": data.get("sector"),
        "industry": data.get("industry"),
        "current_price": data["current_price"],
        "currency": data["currency"],
        "market_cap": data["market_cap"],
        "fundamentals": data["fundamentals"],
    }

    # Comparaison avec la collecte précédente
    deltas = compute_fundamental_deltas(ticker, data)
    deltas_text = format_fundamental_deltas(deltas)

    # Analyse précédente (pour que GLM voie s'il y en a eu une)
    prev_analysis = get_previous_fundamental(ticker)
    prev_section = ""
    if prev_analysis:
        # On tronque à 1500 chars pour ne pas exploser le contexte
        prev_section = (
            "\n\n## TON ANALYSE PRÉCÉDENTE SUR CE TITRE\n"
            f"{prev_analysis[:1500]}\n"
            "(Fin de l'extrait de l'analyse précédente.)\n"
        )

    return (
        "Voici les données à analyser :\n\n"
        f"```json\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n```\n\n"
        f"## ÉVOLUTION DEPUIS LA DERNIÈRE COLLECTE\n{deltas_text}\n"
        f"{prev_section}\n"
        "Produis ta note d'analyse fondamentale en commentant les évolutions."
    )


def analyze(ticker: str) -> str:
    data = load_ticker_data(ticker)
    user_prompt = build_user_prompt(data)

    print(f"→ Envoi à {MODEL_NAME} pour analyse de {data['ticker']}...\n")
    date_ctx = f"Date du jour : {datetime.now().strftime('%d/%m/%Y %H:%M')}.\n\n"
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": date_ctx + SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.3},  # peu de créativité, on veut du factuel
    )
    return response["message"]["content"]


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else input("Ticker à analyser : ").strip()
    analysis = analyze(ticker)

    print("=" * 60)
    print(f"NOTE FONDAMENTALE — {ticker.upper()}")
    print("=" * 60)
    print(analysis)

    # Sauvegarde à côté du JSON pour pouvoir la relire / la donner au Gérant plus tard
    out_path = DATA_DIR / f"{ticker.upper()}_fundamental.md"
    out_path.write_text(analysis, encoding="utf-8")
    print(f"\n→ Analyse sauvegardée dans {out_path}")