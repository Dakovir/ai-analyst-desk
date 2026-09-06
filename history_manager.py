"""
Module utilitaire — Gestionnaire d'historique.
Permet aux agents de RÉUTILISER les données des runs précédents
pour détecter des tendances et évolutions au lieu de repartir de zéro.

Importé par les autres agents, pas exécuté seul.

Usage typique :
  from history_manager import compute_macro_deltas, format_deltas_for_prompt
  deltas = compute_macro_deltas(current_indicators)
  deltas_text = format_deltas_for_prompt(deltas)
  # → injecté dans le prompt pour que GLM commente les tendances
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


# ─────────────────────────────────────────────────────────────────────
# MACRO — comparaison des indicateurs entre runs
# ─────────────────────────────────────────────────────────────────────

def get_previous_macro_data(days_back: int = 30) -> list[dict]:
    """Récupère les données macro brutes (JSON) des N derniers jours.
    Triées par date croissante (la plus ancienne en premier)."""
    files = sorted(DATA_DIR.glob("macro_data_*.json"))
    results = []
    cutoff = datetime.now() - timedelta(days=days_back)

    for f in files:
        date_str = f.stem.replace("macro_data_", "")
        try:
            file_date = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            continue
        if file_date >= cutoff:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                data["_date"] = date_str
                data["_file_date"] = file_date
                results.append(data)
            except (json.JSONDecodeError, OSError):
                continue

    return results


def compute_macro_deltas(current_indicators: dict,
                         days_back: int = 30) -> dict:
    """Compare les indicateurs actuels avec ceux du run précédent le plus récent.

    Retourne un dict :
      { "_compared_to": "20260901",
        "Indices actions par zone": {
            "Amérique — S&P 500": { "current": 5500.12, "previous": 5480.00,
                                     "delta_pct": +0.37 },
            ...
        },
        ...
      }
    Si aucun historique n'existe, retourne {}.
    """
    previous_runs = get_previous_macro_data(days_back)
    if not previous_runs:
        return {}

    # Le run le plus récent AVANT aujourd'hui
    today_str = datetime.now().strftime("%Y%m%d")
    previous_runs = [r for r in previous_runs if r["_date"] != today_str]
    if not previous_runs:
        return {}

    prev = previous_runs[-1]
    prev_indicators = prev.get("indicators", {})
    prev_date = prev.get("_date", "inconnu")

    deltas = {"_compared_to": prev_date}

    for category, symbols in current_indicators.items():
        deltas[category] = {}
        prev_cat = prev_indicators.get(category, {})

        for label, current_data in symbols.items():
            if not isinstance(current_data, dict):
                continue
            current_val = current_data.get("value")
            if current_val is None:
                continue

            prev_entry = prev_cat.get(label, {})
            prev_val = prev_entry.get("value") if isinstance(prev_entry, dict) else None

            if prev_val is None:
                deltas[category][label] = {
                    "current": current_val,
                    "previous": None,
                    "note": "pas de donnée précédente",
                }
                continue

            delta_pct = ((current_val - prev_val) / prev_val * 100) if prev_val else 0.0
            deltas[category][label] = {
                "current": round(current_val, 2),
                "previous": round(prev_val, 2),
                "delta_pct": round(delta_pct, 2),
            }

    return deltas


def format_deltas_for_prompt(deltas: dict) -> str:
    """Formate les deltas en texte lisible pour injection dans un prompt LLM."""
    if not deltas:
        return "(Aucun historique disponible — c'est le premier run.)\n"

    compared_to = deltas.get("_compared_to", "?")
    lines = [f"ÉVOLUTION DEPUIS LE DERNIER RUN ({compared_to}) :\n"]

    for category, symbols in deltas.items():
        if category.startswith("_") or not isinstance(symbols, dict):
            continue
        if not symbols:
            continue

        lines.append(f"\n{category} :")
        for label, data in symbols.items():
            if data.get("previous") is None:
                lines.append(f"  • {label}: {data['current']} (nouveau)")
            else:
                d = data["delta_pct"]
                arrow = "↑" if d > 0 else "↓" if d < 0 else "→"
                lines.append(
                    f"  • {label}: {data['current']} "
                    f"(vs {data['previous']}, {arrow} {d:+.2f}%)"
                )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# FONDAMENTAL — comparaison avec l'analyse précédente
# ─────────────────────────────────────────────────────────────────────

def get_previous_analysis(ticker: str, kind: str) -> str | None:
    """Récupère la dernière analyse sauvegardée pour un ticker et un type donné.

    Fonction générique utilisée par tous les agents.
    kind correspond au suffixe du fichier : fundamental, sentiment,
    balance_sheet, technical, cycle, peers, earnings.
    """
    path = DATA_DIR / f"{ticker.upper()}_{kind}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def get_previous_fundamental(ticker: str) -> str | None:
    """Backward-compat wrapper — utilise get_previous_analysis."""
    return get_previous_analysis(ticker, "fundamental")


def get_previous_ticker_data(ticker: str) -> dict | None:
    """Récupère les données JSON brutes précédemment collectées pour un ticker.
    Permet de comparer les fondamentaux chiffrés entre deux runs."""
    path = DATA_DIR / f"{ticker.upper()}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def compute_fundamental_deltas(ticker: str, current_data: dict) -> dict:
    """Compare les fondamentaux actuels avec la collecte précédente.
    Retourne un dict des changements clés (ou {} si pas d'historique)."""
    prev = get_previous_ticker_data(ticker)
    if not prev:
        return {}

    prev_fundamentals = prev.get("fundamentals", {})
    curr_fundamentals = current_data.get("fundamentals", {})

    if not prev_fundamentals or not curr_fundamentals:
        return {}

    deltas = {
        "_previous_date": prev.get("fetched_at", "inconnu"),
        "_previous_price": prev.get("current_price"),
        "_current_price": current_data.get("current_price"),
    }

    for key in curr_fundamentals:
        curr_val = curr_fundamentals.get(key)
        prev_val = prev_fundamentals.get(key)
        if curr_val is not None and prev_val is not None:
            try:
                curr_f = float(curr_val)
                prev_f = float(prev_val)
                change = curr_f - prev_f
                deltas[key] = {
                    "current": round(curr_f, 4),
                    "previous": round(prev_f, 4),
                    "change": round(change, 4),
                }
            except (ValueError, TypeError):
                pass

    return deltas


def format_fundamental_deltas(deltas: dict) -> str:
    """Formate les deltas fondamentaux en texte pour le prompt."""
    if not deltas:
        return "(Première analyse pour ce ticker — pas de comparaison.)\n"

    prev_date = deltas.get("_previous_date", "?")
    prev_price = deltas.get("_previous_price")
    curr_price = deltas.get("_current_price")

    lines = [f"ÉVOLUTION DEPUIS LA DERNIÈRE COLLECTE ({prev_date[:10]}) :"]

    if prev_price and curr_price:
        price_chg = ((curr_price - prev_price) / prev_price * 100) if prev_price else 0
        lines.append(f"  Prix : {curr_price} (vs {prev_price}, {price_chg:+.2f}%)")

    for key, data in deltas.items():
        if key.startswith("_") or not isinstance(data, dict):
            continue
        lines.append(
            f"  {key}: {data['current']} (vs {data['previous']}, "
            f"variation {data['change']:+.4f})"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# SENTIMENT — récupérer l'analyse précédente pour comparaison
# ─────────────────────────────────────────────────────────────────────

def get_previous_sentiment(ticker: str) -> str | None:
    """Backward-compat wrapper — utilise get_previous_analysis."""
    return get_previous_analysis(ticker, "sentiment")


# ─────────────────────────────────────────────────────────────────────
# SYNTHÈSES — historique des notes du gérant
# ─────────────────────────────────────────────────────────────────────

def get_previous_syntheses(ticker: str, max_count: int = 3) -> list[dict]:
    """Récupère les N dernières synthèses du gérant pour un ticker.
    Utile pour que le gérant voie l'évolution de ses propres conclusions."""
    pattern = f"{ticker.upper()}_synthesis_*.md"
    files = sorted(DATA_DIR.glob(pattern))[-max_count:]
    results = []
    for f in files:
        date_part = f.stem.replace(f"{ticker.upper()}_synthesis_", "")
        results.append({
            "date": date_part,
            "content": f.read_text(encoding="utf-8"),
            "filename": f.name,
        })
    return results


# ─────────────────────────────────────────────────────────────────────
# MACRO — récupérer les notes précédentes (pas juste les chiffres)
# ─────────────────────────────────────────────────────────────────────

def get_previous_macro_briefs(max_count: int = 3) -> list[dict]:
    """Récupère les N dernières notes macro rédigées."""
    files = sorted(DATA_DIR.glob("macro_brief_*.md"))[-max_count:]
    results = []
    for f in files:
        date_part = f.stem.replace("macro_brief_", "")
        results.append({
            "date": date_part,
            "content": f.read_text(encoding="utf-8"),
            "filename": f.name,
        })
    return results