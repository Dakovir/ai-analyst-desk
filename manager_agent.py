"""
Phase 4 — Agent Gérant / Synthétiseur (v4 — desk complet, 9 analystes).

Le gérant reçoit maintenant les notes de NEUF analystes de son équipe :
  1. Analyste fondamental  (ratios snapshot, valorisation)
  2. Analyste bilan        (états financiers multi-annuels)
  3. Analyste cycle        (positionnement CAPEX/marges/dette dans le cycle)
  4. Analyste pairs        (benchmarking sectoriel vs peers)
  5. Analyste technique    (price action, momentum, signaux techniques)
  6. Analyste sentiment    (news, ton ambiant, catalyseurs de surprise)
  7. Stratégiste macro     (indices, taux, devises, matières premières)
  8. Analyste géopolitique (programmes, conflits, méga-tendances)
  9. Analyste earnings     (conférences de résultats, guidance management)

+ L'historique des synthèses précédentes pour ce ticker.

Ne conclut JAMAIS "achète/vends" — c'est le principe de design du projet.

Pré-requis : les agents fondamental et sentiment doivent avoir tourné.
Tous les autres sont optionnels (le gérant s'adapte s'ils manquent).
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import ollama

from history_manager import get_previous_syntheses

MODEL_NAME = "glm-4.7-flash:latest"
DATA_DIR = Path(__file__).parent / "data"
GEO_DIR = DATA_DIR / "geopolitical"


# ─────────────────────────────────────────────────────────────────────
# Chargement des analyses
# ─────────────────────────────────────────────────────────────────────

def load_analysis(ticker: str, kind: str) -> str:
    """Charge une analyse obligatoire (fondamental, sentiment)."""
    path = DATA_DIR / f"{ticker.upper()}_{kind}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Analyse {kind} manquante pour {ticker}. "
            f"Lance d'abord : python {kind}_agent.py {ticker}"
        )
    return path.read_text(encoding="utf-8")


def load_optional_analysis(ticker: str, kind: str) -> str | None:
    """Charge une analyse optionnelle (balance_sheet, technical, cycle, peers, earnings)."""
    path = DATA_DIR / f"{ticker.upper()}_{kind}.md"
    if path.exists():
        print(f"  {kind.replace('_', ' ').title()} trouvé : {path.name}")
        return path.read_text(encoding="utf-8")
    return None


def load_latest_macro() -> str | None:
    """Récupère la note macro la plus récente."""
    briefs = sorted(DATA_DIR.glob("macro_brief_*.md"))
    if not briefs:
        return None
    latest = briefs[-1]
    print(f"  Macro trouvé : {latest.name}")
    return latest.read_text(encoding="utf-8")


def load_latest_geopolitical() -> str | None:
    """Récupère la note géopolitique la plus récente."""
    if not GEO_DIR.exists():
        return None
    briefs = sorted(GEO_DIR.glob("geopolitical_brief_*.md"))
    if not briefs:
        return None
    latest = briefs[-1]
    print(f"  Géopolitique trouvé : {latest.name}")
    return latest.read_text(encoding="utf-8")


def load_ticker_meta(ticker: str) -> dict:
    path = DATA_DIR / f"{ticker.upper()}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "name": data.get("name"),
        "sector": data.get("sector"),
        "industry": data.get("industry"),
        "current_price": data.get("current_price"),
        "currency": data.get("currency"),
    }


# ─────────────────────────────────────────────────────────────────────
# System prompt élargi — 9 analystes
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un gérant de portefeuille senior qui dirige un desk de
NEUF analystes. Tu reçois leurs notes et tu dois les confronter pour produire
la note de synthèse la plus complète possible. Tu reçois aussi tes propres
synthèses précédentes sur ce titre quand elles existent.

Tes neuf analystes :
  1. Fondamental — ratios snapshot, valorisation
  2. Bilan & états financiers — trajectoire multi-annuelle (CA, marges, dette, FCF)
  3. Cycle sectoriel — positionnement dans le cycle (expansion/maturité/compression/contraction)
  4. Pairs / Comparatif — benchmarking vs concurrents du même secteur
  5. Technique — price action, momentum, signaux techniques
  6. Sentiment — news, ton ambiant, catalyseurs
  7. Macro — indices, taux, devises, matières premières
  8. Géopolitique — programmes, conflits, méga-tendances
  9. Earnings call — ce que dit le management

Ton rôle N'EST PAS de dire "achète" ou "vends". Ton rôle est de :
- confronter les NEUF analyses (ou celles disponibles),
- construire la meilleure thèse haussière ET la meilleure thèse baissière,
- signaler les contradictions ENTRE analystes (c'est là qu'est la valeur),
- ne pas simplement résumer chaque note — les CROISER.

RÈGLE IMPORTANTE : ne recopie pas les notes en entier. Sélectionne ce qui est
PERTINENT pour CE titre et explique les liens causaux entre les analyses.

Structure ta note en français, dans cet ordre exact :

## RÉSUMÉ EXÉCUTIF
4-5 phrases : qui, quoi, tension centrale, positionnement dans le cycle,
et verdict technique vs fondamental.

## ANALYSE FONDAMENTALE & BILAN
Croise les notes fondamental + bilan + cycle. La trajectoire des états
financiers confirme-t-elle ou contredit-elle la photo instantanée des ratios ?
Le positionnement dans le cycle explique-t-il la valorisation ?
Signale si les marges se compriment alors que le CAPEX accélère (fin de cycle).

## POSITIONNEMENT VS PAIRS
Le titre est-il cher ou décoté par rapport à ses concurrents ? Sa rentabilité
et sa croissance justifient-elles un premium ? Compare avec les conclusions
de l'analyste fondamental sur la valorisation.

## ANALYSE TECHNIQUE
Tendance, momentum, niveaux clés. L'analyse technique confirme-t-elle ou
contredit-elle la thèse fondamentale ? Un titre fondamentalement solide en
tendance baissière technique est un signal différent d'un titre solide en
breakout haussier.

## CE QUE DIT LE MANAGEMENT (EARNINGS)
Messages clés du dernier earnings call. La guidance valide-t-elle ou
contredit-elle les chiffres du bilan et la lecture du cycle ?
Si pas d'earnings disponible, signale-le.

## CONTEXTE MACRO PERTINENT
Uniquement les facteurs macro qui touchent CE titre, avec le lien explicite.
Si le macro est neutre, dis-le. 3-5 puces max.

## CONTEXTE GÉOPOLITIQUE PERTINENT
Uniquement les facteurs géopolitiques qui touchent CE titre ou son secteur.
Si rien n'est pertinent, dis-le en une phrase. 3-5 puces max.

## THÈSE HAUSSIÈRE (BULL)
Meilleurs arguments positifs, en piochant dans les NEUF analyses.
Cite les chiffres. 4-6 puces.

## THÈSE BAISSIÈRE (BEAR)
Meilleurs arguments prudents/négatifs, en piochant dans les neuf.
Cite les chiffres. 4-6 puces.

## CONTRADICTIONS ET TENSIONS
Le cœur de la note. Où les analyses DIVERGENT-elles ? Inclus :
- fondamental vs technique (valorisation attractive mais momentum négatif ?)
- cycle vs pairs (en fin de cycle mais moins cher que les pairs ?)
- bilan vs sentiment (trajectoire financière solide mais newsflow négatif ?)
- management vs marché (guidance optimiste mais sentiment dégradé ?)
- macro vs micro (fondamentaux solides mais taux défavorables ?)
- géopolitique vs fondamentaux (bons chiffres mais exposition tarifs/conflits ?)

## ÉVOLUTION DE L'ANALYSE
Si des synthèses précédentes existent : qu'est-ce qui a changé depuis
la dernière note ? La thèse s'est-elle renforcée, affaiblie, inversée ?
Si c'est la première note, signale-le.

## POINTS DE VIGILANCE
Ce qu'il faut surveiller, classé par horizon :
- Court terme (semaines) : niveaux techniques, catalyseurs
- Moyen terme (trimestres) : earnings, évolution des marges, cycle
- Long terme : macro, géopolitique, positionnement sectoriel

## DISCLAIMER
Note produite par un système d'IA multi-agents à partir de données publiques.
Elle ne constitue pas un conseil en investissement. Les analyses peuvent
contenir des erreurs ou des interprétations discutables. Toute décision
d'investissement est sous la responsabilité exclusive du lecteur."""


# ─────────────────────────────────────────────────────────────────────
# Construction du prompt utilisateur
# ─────────────────────────────────────────────────────────────────────

def build_user_prompt(
    ticker: str,
    fundamental: str,
    sentiment: str,
    balance_sheet: str | None,
    technical: str | None,
    cycle: str | None,
    peers: str | None,
    macro: str | None,
    geopolitical: str | None,
    earnings: str | None,
    previous_syntheses: list[dict],
    meta: dict,
) -> str:
    meta_line = ""
    if meta:
        meta_line = (
            f"Contexte : {meta.get('name')} ({ticker.upper()}), "
            f"secteur {meta.get('sector')}"
        )
        if meta.get("industry"):
            meta_line += f", industrie {meta['industry']}"
        meta_line += (
            f", cours actuel {meta.get('current_price')} "
            f"{meta.get('currency')}.\n\n"
        )

    sep = "=" * 60
    parts = [meta_line]

    # --- Les 9 analyses (ou celles disponibles) ---
    parts.append(
        f"{sep}\n## NOTE DE L'ANALYSTE FONDAMENTAL\n{sep}\n{fundamental}\n"
    )

    if balance_sheet:
        parts.append(
            f"{sep}\n## NOTE DE L'ANALYSTE BILAN & ÉTATS FINANCIERS\n{sep}\n"
            f"{balance_sheet}\n"
        )

    if cycle:
        parts.append(
            f"{sep}\n## NOTE DE L'ANALYSTE CYCLE SECTORIEL\n{sep}\n{cycle}\n"
        )

    if peers:
        parts.append(
            f"{sep}\n## NOTE DE L'ANALYSTE COMPARATIF (PAIRS)\n{sep}\n{peers}\n"
        )

    if technical:
        parts.append(
            f"{sep}\n## NOTE DE L'ANALYSTE TECHNIQUE\n{sep}\n{technical}\n"
        )

    parts.append(
        f"{sep}\n## NOTE DE L'ANALYSTE SENTIMENT\n{sep}\n{sentiment}\n"
    )

    if macro:
        parts.append(
            f"{sep}\n## NOTE DU STRATÉGISTE MACRO\n{sep}\n{macro}\n"
        )
    else:
        parts.append(
            "(Aucune note macro disponible. Signale-le dans la section "
            "CONTEXTE MACRO PERTINENT et concentre-toi sur le titre.)\n"
        )

    if geopolitical:
        parts.append(
            f"{sep}\n## NOTE DE L'ANALYSTE GÉOPOLITIQUE\n{sep}\n"
            f"{geopolitical}\n"
        )
    else:
        parts.append(
            "(Aucune note géopolitique disponible. Signale-le dans la "
            "section CONTEXTE GÉOPOLITIQUE PERTINENT.)\n"
        )

    if earnings:
        parts.append(
            f"{sep}\n## NOTE EARNINGS CALL\n{sep}\n{earnings}\n"
        )
    else:
        parts.append(
            "(Aucune analyse d'earnings call disponible pour ce ticker. "
            "Signale-le dans la section CE QUE DIT LE MANAGEMENT.)\n"
        )

    # --- Notes manquantes : résumé ---
    missing = []
    if not balance_sheet:
        missing.append("bilan/états financiers")
    if not cycle:
        missing.append("cycle sectoriel")
    if not peers:
        missing.append("comparatif pairs")
    if not technical:
        missing.append("technique")
    if missing:
        parts.append(
            f"\n(Notes manquantes : {', '.join(missing)}. "
            "Adapte ta synthèse en conséquence — ne laisse pas de "
            "section vide, signale ce qui manque.)\n"
        )

    # --- Historique des synthèses précédentes ---
    if previous_syntheses:
        parts.append(
            f"{sep}\n## TES SYNTHÈSES PRÉCÉDENTES SUR CE TITRE\n{sep}"
        )
        for synth in previous_syntheses[-2:]:
            parts.append(
                f"\n--- Synthèse du {synth['date']} ---\n"
                f"{synth['content'][:2000]}\n"
            )
    else:
        parts.append(
            "(Première synthèse pour ce titre — pas d'historique. "
            "Signale-le dans la section ÉVOLUTION DE L'ANALYSE.)\n"
        )

    parts.append(
        f"\n{sep}\n"
        "Rédige maintenant ta note de synthèse en suivant STRICTEMENT "
        "la structure demandée. Croise les analyses, ne les résume pas "
        "séparément. Sois spécifique sur les liens causaux."
    )
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────
# Synthèse et sauvegarde
# ─────────────────────────────────────────────────────────────────────

def synthesize(ticker: str) -> str:
    print(f"→ Chargement des analyses pour {ticker.upper()}...\n")

    # Obligatoires
    fundamental = load_analysis(ticker, "fundamental")
    sentiment = load_analysis(ticker, "sentiment")

    # Optionnels — per-ticker
    balance_sheet = load_optional_analysis(ticker, "balance_sheet")
    technical = load_optional_analysis(ticker, "technical")
    cycle = load_optional_analysis(ticker, "cycle")
    peers = load_optional_analysis(ticker, "peers")
    earnings = load_optional_analysis(ticker, "earnings")

    # Optionnels — partagés
    macro = load_latest_macro()
    geopolitical = load_latest_geopolitical()

    # Compteur
    available = 2  # fundamental + sentiment
    for a in [balance_sheet, technical, cycle, peers, earnings, macro, geopolitical]:
        if a is not None:
            available += 1
    print(f"\n  → {available}/9 analyses disponibles pour la synthèse")

    # Historique
    previous = get_previous_syntheses(ticker)
    if previous:
        print(f"  Historique : {len(previous)} synthèse(s) précédente(s)")
    else:
        print("  Historique : première synthèse pour ce titre")

    meta = load_ticker_meta(ticker)

    user_prompt = build_user_prompt(
        ticker, fundamental, sentiment,
        balance_sheet, technical, cycle, peers,
        macro, geopolitical, earnings,
        previous, meta
    )

    print(f"\n→ Envoi à {MODEL_NAME} pour synthèse Gérant "
          f"sur {ticker.upper()}...\n")
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


def save_note(ticker: str, note: str) -> Path:
    ticker = ticker.upper()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = DATA_DIR / f"{ticker}_synthesis_{timestamp}.md"

    header = (
        f"# Note de synthèse — {ticker}\n"
        f"*Générée le {datetime.now().strftime('%d/%m/%Y à %H:%M')} "
        f"par le système multi-agents v4 (modèle : {MODEL_NAME})*\n\n"
        "---\n\n"
    )
    out_path.write_text(header + note, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    ticker = (
        sys.argv[1]
        if len(sys.argv) > 1
        else input("Ticker à synthétiser : ").strip()
    )
    note = synthesize(ticker)

    print("=" * 60)
    print(f"NOTE DE SYNTHÈSE — {ticker.upper()}")
    print("=" * 60)
    print(note)

    saved = save_note(ticker, note)
    print(f"\n→ Note finale sauvegardée dans {saved}")