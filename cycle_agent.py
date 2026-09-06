"""
Agent Analyste Cycle — Positionnement dans le cycle sectoriel.

Analyse la trajectoire CAPEX, dette, marges sur plusieurs années pour
déterminer où en est l'entreprise dans son cycle :
  - EXPANSION : CAPEX accélère, marges montent, dette peut croître
  - MATURITÉ  : CAPEX stable, marges au plateau, FCF fort
  - COMPRESSION : Marges en baisse, dette élevée, CAPEX qui ralentit
  - CONTRACTION : Restructuration, baisse de CA, deleveraging

C'est typiquement l'analyse qu'il manquait pour les semi-conducteurs :
CAPEX qui s'emballe → dette qui monte → marges qui se compriment →
fin de cycle et retournement.

Pré-requis : pip install ollama
Le collector.py v2 doit avoir tourné (états financiers nécessaires).
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import ollama

from history_manager import get_previous_analysis

MODEL_NAME = "glm-4.7-flash:latest"
DATA_DIR = Path(__file__).parent / "data"


def load_ticker_data(ticker: str) -> dict:
    path = DATA_DIR / f"{ticker.upper()}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Pas de données pour {ticker}. "
            f"Lance d'abord : python collector.py {ticker}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def compute_cycle_metrics(data: dict) -> dict:
    """Calcule les métriques de cycle à partir des états financiers.
    Retourne un dict structuré avec les trajectoires annuelles."""

    income = data.get("income_statement", {})
    balance = data.get("balance_sheet", {})
    cashflow = data.get("cash_flow", {})

    if not income and not cashflow:
        return {"error": "Pas d'états financiers disponibles"}

    # Trier les dates (du plus ancien au plus récent)
    dates = sorted(
        set(
            list(income.keys())
            + list(balance.keys())
            + list(cashflow.keys())
        )
    )

    metrics = {"dates": dates, "annual_data": {}}

    for date in dates:
        year_data = {}
        inc = income.get(date, {})
        bal = balance.get(date, {})
        cf = cashflow.get(date, {})

        # Revenue
        revenue = inc.get("Total Revenue") or inc.get("Operating Revenue")
        year_data["revenue"] = revenue

        # CAPEX (négatif dans yfinance → on le rend positif pour lisibilité)
        capex_raw = cf.get("Capital Expenditure")
        capex = abs(capex_raw) if capex_raw is not None else None
        year_data["capex"] = capex

        # CAPEX / Revenue ratio
        if revenue and capex:
            year_data["capex_to_revenue_pct"] = round(capex / revenue * 100, 2)

        # Marges
        gross = inc.get("Gross Profit")
        operating = inc.get("Operating Income") or inc.get("EBIT")
        net_income = inc.get("Net Income")

        if revenue and gross:
            year_data["gross_margin_pct"] = round(gross / revenue * 100, 2)
        if revenue and operating:
            year_data["operating_margin_pct"] = round(
                operating / revenue * 100, 2
            )
        if revenue and net_income:
            year_data["net_margin_pct"] = round(
                net_income / revenue * 100, 2
            )

        # Dette
        total_debt = bal.get("Total Debt") or bal.get("Long Term Debt")
        equity = bal.get("Stockholders Equity") or bal.get(
            "Total Equity Gross Minority Interest"
        )
        total_assets = bal.get("Total Assets")

        year_data["total_debt"] = total_debt
        year_data["equity"] = equity

        if total_debt and equity and equity > 0:
            year_data["debt_to_equity"] = round(total_debt / equity, 2)
        if total_debt and total_assets and total_assets > 0:
            year_data["debt_to_assets_pct"] = round(
                total_debt / total_assets * 100, 2
            )

        # Cash flow
        ocf = cf.get("Operating Cash Flow")
        fcf = cf.get("Free Cash Flow")
        # Fallback : FCF = OCF + CAPEX (CAPEX est négatif)
        if fcf is None and ocf is not None and capex_raw is not None:
            fcf = ocf + capex_raw

        year_data["operating_cash_flow"] = ocf
        year_data["free_cash_flow"] = fcf

        if revenue and fcf:
            year_data["fcf_to_revenue_pct"] = round(fcf / revenue * 100, 2)

        # R&D si disponible
        rd = inc.get("Research And Development")
        if revenue and rd:
            year_data["rd_to_revenue_pct"] = round(abs(rd) / revenue * 100, 2)

        # SBC (Stock-Based Compensation)
        sbc = cf.get("Stock Based Compensation")
        if revenue and sbc:
            year_data["sbc_to_revenue_pct"] = round(
                abs(sbc) / revenue * 100, 2
            )

        # Nettoyer les None
        year_data = {k: v for k, v in year_data.items() if v is not None}
        metrics["annual_data"][date] = year_data

    # Calculer les tendances (variation entre premier et dernier exercice)
    if len(dates) >= 2:
        first = metrics["annual_data"].get(dates[0], {})
        last = metrics["annual_data"].get(dates[-1], {})

        trends = {}
        for key in [
            "capex_to_revenue_pct",
            "gross_margin_pct",
            "operating_margin_pct",
            "net_margin_pct",
            "debt_to_equity",
            "fcf_to_revenue_pct",
        ]:
            v1 = first.get(key)
            v2 = last.get(key)
            if v1 is not None and v2 is not None:
                diff = v2 - v1
                if diff > 0:
                    direction = "HAUSSE"
                elif diff < 0:
                    direction = "BAISSE"
                else:
                    direction = "STABLE"
                trends[key] = {
                    "premier_exercice": v1,
                    "dernier_exercice": v2,
                    "variation": round(diff, 2),
                    "tendance": direction,
                }

        # CAGR du revenue
        rev_first = first.get("revenue")
        rev_last = last.get("revenue")
        n_years = len(dates) - 1
        if rev_first and rev_last and rev_first > 0 and n_years > 0:
            cagr = ((rev_last / rev_first) ** (1 / n_years) - 1) * 100
            trends["revenue_cagr_pct"] = round(cagr, 2)

        metrics["trends"] = trends

    return metrics


SYSTEM_PROMPT = """Tu es un analyste de cycles sectoriels et de marchés de capitaux.
Tu reçois des métriques de cycle calculées à partir des états financiers d'une
société sur plusieurs années, et tu détermines sa POSITION DANS LE CYCLE.

Les 4 phases classiques d'un cycle :
  EXPANSION — CAPEX accélère, CA en croissance forte, marges en expansion,
              la société investit massivement (parfois en s'endettant)
  MATURITÉ  — CAPEX stabilisé, marges au plateau, FCF abondant, croissance
              qui ralentit mais reste positive
  COMPRESSION — Marges qui commencent à se contracter, dette élevée, CAPEX
                qui ralentit, le marché se normalise
  CONTRACTION — CA qui baisse, restructuration, deleveraging, parfois
                destruction de valeur avant la reprise

Ce cadre est PARTICULIÈREMENT important pour les secteurs cycliques :
semi-conducteurs, énergie, construction, matières premières, automobile.
Pour les secteurs non-cycliques (pharma, utilities, consommation de base),
adapte ton analyse en expliquant pourquoi le cycle classique s'applique
différemment.

Ta note doit contenir, en français concis :

1. INTENSITÉ CAPITALISTIQUE — l'évolution du ratio CAPEX/CA sur la période.
   Le CAPEX accélère-t-il (signe d'expansion) ou décélère (fin de cycle) ?
   Si R&D est disponible, commente aussi l'effort d'innovation.

2. TRAJECTOIRE DES MARGES — les marges (brute, opérationnelle, nette) sont-
   elles en expansion, stables, ou en compression ? La compression des marges
   avec un CAPEX élevé est le signal classique de fin de cycle.

3. DYNAMIQUE DE LA DETTE — la dette augmente-t-elle en proportion des fonds
   propres ? L'endettement sert-il à financer la croissance (positif) ou
   à compenser une baisse de cash flow (négatif) ?

4. QUALITÉ DU CASH FLOW — le FCF suit-il le résultat net ? Le ratio FCF/CA
   est-il stable, en hausse (bonne gestion) ou en baisse (le CAPEX mange le
   cash) ?

5. DIAGNOSTIC DE PHASE — dans quelle phase du cycle la société se situe ?
   Justifie avec les tendances chiffrées. Sois SPÉCIFIQUE :
   "début d'expansion" ≠ "fin d'expansion" ≠ "transition vers maturité".

6. CE QUI VIENDRAIT ENSUITE — si la société est en phase X, quels seraient
   les signaux qui confirmeraient un passage en phase suivante ? C'est le
   point d'inflexion à surveiller.

Reste factuel, cite les chiffres année par année. Pas de conseil d'achat/vente."""


def build_user_prompt(ticker: str, metrics: dict, meta: dict) -> str:
    prev = get_previous_analysis(ticker, "cycle")
    prev_section = ""
    if prev:
        prev_section = (
            "\n\n## TON ANALYSE DE CYCLE PRÉCÉDENTE\n"
            f"{prev[:1500]}\n"
            "(Fin de l'extrait. La phase a-t-elle évolué ?)\n"
        )

    return (
        f"## {meta.get('name', ticker)} ({ticker.upper()})\n"
        f"Secteur : {meta.get('sector')}, Industrie : {meta.get('industry')}\n\n"
        "## MÉTRIQUES DE CYCLE CALCULÉES\n"
        f"```json\n{json.dumps(metrics, indent=2, ensure_ascii=False)}\n```\n"
        f"{prev_section}\n"
        "Produis ton diagnostic de positionnement dans le cycle."
    )


def analyze(ticker: str) -> str:
    data = load_ticker_data(ticker)

    print(f"→ Calcul des métriques de cycle pour {ticker.upper()}...")
    metrics = compute_cycle_metrics(data)

    if "error" in metrics:
        return f"Erreur : {metrics['error']}"

    n_years = len(metrics.get("dates", []))
    print(f"  {n_years} exercice(s) analysé(s)")

    if metrics.get("trends"):
        for k, v in metrics["trends"].items():
            if isinstance(v, dict):
                print(
                    f"  {k}: {v.get('premier_exercice')} → "
                    f"{v.get('dernier_exercice')} ({v.get('tendance')})"
                )
            else:
                print(f"  {k}: {v}")

    meta = {
        "name": data.get("name"),
        "sector": data.get("sector"),
        "industry": data.get("industry"),
    }
    user_prompt = build_user_prompt(ticker, metrics, meta)

    print(f"\n→ Envoi à {MODEL_NAME} pour diagnostic de cycle...\n")
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
    out_path = DATA_DIR / f"{ticker}_cycle.md"
    out_path.write_text(analysis, encoding="utf-8")
    print(f"\n→ Analyse de cycle sauvegardée dans {out_path}")
    return out_path


if __name__ == "__main__":
    ticker = (
        sys.argv[1]
        if len(sys.argv) > 1
        else input("Ticker à analyser : ").strip()
    )
    analysis = analyze(ticker)

    print("=" * 60)
    print(f"ANALYSE DE CYCLE — {ticker.upper()}")
    print("=" * 60)
    print(analysis)

    save_output(ticker, analysis)