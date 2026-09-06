"""
Agent Analyste Bilan & Compte de Résultat — Deep dive financier multi-annuel.

Contrairement à l'analyste fondamental (qui travaille sur les ratios snapshot
de yfinance.info), cet agent lit les VRAIS états financiers : compte de résultat,
bilan, et flux de trésorerie sur 4-5 ans. Il regarde les tendances : la marge
s'améliore-t-elle ? La dette se creuse ? Le FCF est-il en croissance ?

C'est l'analyse que ferait un analyste equity researcher avec les 10-K/10-Q.

Pré-requis : pip install ollama
Le collector.py v2 doit avoir tourné (data/<TICKER>.json avec income_statement,
balance_sheet, cash_flow).
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import ollama

from history_manager import get_previous_analysis

MODEL_NAME = "glm-4.7-flash:latest"
DATA_DIR = Path(__file__).parent / "data"

SYSTEM_PROMPT = """Tu es un analyste equity research senior. Tu reçois les états
financiers réels d'une société (compte de résultat, bilan, flux de trésorerie)
sur plusieurs exercices, et tu produis une analyse APPROFONDIE de la trajectoire
financière — pas un simple commentaire de ratios.

Ta note doit contenir, en français concis :

1. TRAJECTOIRE DU CHIFFRE D'AFFAIRES — croissance annuelle, accélération ou
   décélération, saisonnalité (si trimestres disponibles). Cite les chiffres
   année par année.

2. PROFITABILITÉ ET MARGES — marge brute, marge opérationnelle, marge nette.
   La tendance est-elle à l'expansion ou à la compression des marges ?
   Compare les marges du dernier exercice aux exercices précédents.

3. STRUCTURE DU BILAN — actifs totaux, capitaux propres, ratio dette/fonds
   propres. La dette augmente-t-elle plus vite que les actifs ?
   Current ratio (actifs courants / passifs courants) si les données permettent.

4. DETTE ET TRÉSORERIE — dette totale vs trésorerie. Le ratio de couverture
   s'améliore-t-il ou se dégrade ? La société est-elle en position de
   rembourser ou de s'endetter davantage ?

5. FLUX DE TRÉSORERIE — Operating Cash Flow, CAPEX, Free Cash Flow.
   Le FCF est-il en croissance ? Le ratio FCF/revenu est-il stable ?
   Y a-t-il un signal d'investissement massif (CAPEX qui explose) ?

6. QUALITÉ DES BÉNÉFICES — le résultat net est-il soutenu par le cash flow
   opérationnel ? (Si le résultat augmente mais le cash flow stagne, c'est
   un red flag comptable.)

7. SIGNAUX D'ALERTE — éléments inhabituels : goodwill disproportionné,
   stock-based compensation massive, write-downs, changements comptables.

Reste factuel, cite les chiffres année par année quand c'est pertinent.
Signale explicitement les données manquantes. Pas de conclusion "achète/vends"."""


def load_ticker_data(ticker: str) -> dict:
    path = DATA_DIR / f"{ticker.upper()}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Pas de données pour {ticker}. "
            f"Lance d'abord : python collector.py {ticker}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_user_prompt(data: dict) -> str:
    ticker = data["ticker"]
    mc = data.get("market_cap")
    mc_str = f"{mc:,.0f}" if mc else "N/A"

    sections = [
        f"## SOCIÉTÉ : {data.get('name')} ({ticker})",
        f"Secteur : {data.get('sector')} / {data.get('industry')}",
        f"Prix actuel : {data.get('current_price')} {data.get('currency')}",
        f"Market cap : {mc_str}",
        "",
    ]

    # Compte de résultat annuel
    income = data.get("income_statement", {})
    if income:
        sections.append("## COMPTE DE RÉSULTAT (annuel)")
        sections.append(
            f"```json\n{json.dumps(income, indent=2, ensure_ascii=False)}\n```"
        )
    else:
        sections.append("(Compte de résultat annuel non disponible)")

    # Résultats trimestriels
    quarterly = data.get("quarterly_income", {})
    if quarterly:
        sections.append("\n## RÉSULTATS TRIMESTRIELS (4 derniers)")
        sections.append(
            f"```json\n{json.dumps(quarterly, indent=2, ensure_ascii=False)}\n```"
        )

    # Bilan
    bs = data.get("balance_sheet", {})
    if bs:
        sections.append("\n## BILAN (annuel)")
        sections.append(
            f"```json\n{json.dumps(bs, indent=2, ensure_ascii=False)}\n```"
        )
    else:
        sections.append("(Bilan non disponible)")

    # Cash flow
    cf = data.get("cash_flow", {})
    if cf:
        sections.append("\n## FLUX DE TRÉSORERIE (annuel)")
        sections.append(
            f"```json\n{json.dumps(cf, indent=2, ensure_ascii=False)}\n```"
        )
    else:
        sections.append("(Flux de trésorerie non disponible)")

    # Analyse précédente
    prev = get_previous_analysis(ticker, "balance_sheet")
    if prev:
        sections.append(
            "\n## TON ANALYSE PRÉCÉDENTE SUR CE TITRE\n"
            f"{prev[:1500]}\n"
            "(Fin de l'extrait de l'analyse précédente.)\n"
        )

    sections.append(
        "\nProduis ta note d'analyse approfondie des états financiers "
        "en commentant les tendances multi-annuelles."
    )
    return "\n".join(sections)


def analyze(ticker: str) -> str:
    data = load_ticker_data(ticker)

    has_statements = any(
        data.get(k) for k in ["income_statement", "balance_sheet", "cash_flow"]
    )
    if not has_statements:
        return (
            f"Aucun état financier disponible pour {ticker}. "
            "Le collecteur v2 est nécessaire. "
            f"Relance : python collector.py {ticker}"
        )

    user_prompt = build_user_prompt(data)

    print(
        f"→ Envoi à {MODEL_NAME} pour analyse des "
        f"états financiers de {ticker.upper()}...\n"
    )
    date_ctx = f"Date du jour : {datetime.now().strftime('%d/%m/%Y %H:%M')}.\n\n"
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": date_ctx + SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.3},
    )
    return response["message"]["content"]


def save_output(ticker: str, analysis: str) -> Path:
    ticker = ticker.upper()
    out_path = DATA_DIR / f"{ticker}_balance_sheet.md"
    out_path.write_text(analysis, encoding="utf-8")
    print(f"\n→ Analyse bilan sauvegardée dans {out_path}")
    return out_path


if __name__ == "__main__":
    ticker = (
        sys.argv[1]
        if len(sys.argv) > 1
        else input("Ticker à analyser : ").strip()
    )
    analysis = analyze(ticker)

    print("=" * 60)
    print(f"ANALYSE BILAN & ÉTATS FINANCIERS — {ticker.upper()}")
    print("=" * 60)
    print(analysis)

    save_output(ticker, analysis)