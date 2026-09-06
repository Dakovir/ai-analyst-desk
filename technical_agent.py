"""
Agent Analyste Technique — Price action, momentum, signaux techniques.

Calcule les indicateurs techniques en Python (pas par le LLM, c'est du calcul)
puis les transmet à GLM pour interprétation dans le contexte du titre.

Indicateurs calculés :
  - SMA 20, SMA 50 (moyennes mobiles simples)
  - RSI 14 (Relative Strength Index)
  - Bandes de Bollinger (20 jours, 2 écarts-types)
  - Momentum : rendements 1 mois et 3 mois
  - Tendance de volume (volume moyen vs volume récent)
  - Support/résistance approximatifs (extrema locaux)

Pré-requis : pip install ollama
Le collector.py v2 doit avoir tourné (pour avoir OHLCV dans price_history_6mo).
Fonctionne aussi avec le v1 (close seul), mais avec moins d'indicateurs.
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import ollama

from history_manager import get_previous_analysis

MODEL_NAME = "glm-4.7-flash:latest"
DATA_DIR = Path(__file__).parent / "data"


# ─────────────────────────────────────────────────────────────────────
# Calcul des indicateurs techniques (en Python, pas par le LLM)
# ─────────────────────────────────────────────────────────────────────


def _sma(closes: list[float], period: int) -> float | None:
    """Simple Moving Average sur les N derniers jours."""
    if len(closes) < period:
        return None
    return round(sum(closes[-period:]) / period, 2)


def _rsi(closes: list[float], period: int = 14) -> float | None:
    """Relative Strength Index (Wilder's smoothing)."""
    if len(closes) < period + 1:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    # Seed avec la première fenêtre
    gains = [max(d, 0) for d in deltas[:period]]
    losses = [max(-d, 0) for d in deltas[:period]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Smoothing exponentiel (Wilder)
    for d in deltas[period:]:
        if d > 0:
            avg_gain = (avg_gain * (period - 1) + d) / period
            avg_loss = (avg_loss * (period - 1)) / period
        else:
            avg_gain = (avg_gain * (period - 1)) / period
            avg_loss = (avg_loss * (period - 1) + abs(d)) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 1)


def _bollinger_bands(
    closes: list[float], period: int = 20, num_std: int = 2
) -> dict | None:
    """Bandes de Bollinger."""
    if len(closes) < period:
        return None

    window = closes[-period:]
    sma = sum(window) / period
    variance = sum((x - sma) ** 2 for x in window) / period
    std = math.sqrt(variance)

    return {
        "upper": round(sma + num_std * std, 2),
        "middle": round(sma, 2),
        "lower": round(sma - num_std * std, 2),
        "bandwidth_pct": (
            round((num_std * 2 * std / sma) * 100, 2) if sma else None
        ),
    }


def _find_support_resistance(
    highs: list[float], lows: list[float], window: int = 5
) -> dict:
    """Trouve des niveaux de support/résistance approximatifs
    (extrema locaux sur la moitié la plus récente des données)."""
    half = len(highs) // 2
    recent_highs = highs[half:]
    recent_lows = lows[half:]

    resistances = []
    supports = []

    for i in range(window, len(recent_highs) - window):
        chunk_h = recent_highs[i - window : i + window + 1]
        if recent_highs[i] == max(chunk_h):
            resistances.append(round(recent_highs[i], 2))

        chunk_l = recent_lows[i - window : i + window + 1]
        if recent_lows[i] == min(chunk_l):
            supports.append(round(recent_lows[i], 2))

    return {
        "resistance_levels": sorted(set(resistances), reverse=True)[:3],
        "support_levels": sorted(set(supports))[:3],
    }


def compute_indicators(price_data: list[dict]) -> dict:
    """Calcule tous les indicateurs techniques à partir de l'historique OHLCV."""
    if not price_data or len(price_data) < 20:
        return {"error": "Pas assez de données de prix (minimum 20 jours)"}

    closes = [p["close"] for p in price_data]
    highs = [p.get("high", p["close"]) for p in price_data]
    lows = [p.get("low", p["close"]) for p in price_data]
    volumes = [p["volume"] for p in price_data if p.get("volume")]

    current_price = closes[-1]

    # Moyennes mobiles
    sma_20 = _sma(closes, 20)
    sma_50 = _sma(closes, 50)

    sma_signals = {}
    if sma_20:
        above_20 = current_price > sma_20
        sma_signals["prix_vs_sma20"] = "AU-DESSUS" if above_20 else "EN-DESSOUS"
        sma_signals["ecart_sma20_pct"] = round(
            (current_price - sma_20) / sma_20 * 100, 2
        )
    if sma_50:
        above_50 = current_price > sma_50
        sma_signals["prix_vs_sma50"] = "AU-DESSUS" if above_50 else "EN-DESSOUS"
        sma_signals["ecart_sma50_pct"] = round(
            (current_price - sma_50) / sma_50 * 100, 2
        )
    if sma_20 and sma_50:
        sma_signals["golden_cross"] = sma_20 > sma_50

    # RSI
    rsi = _rsi(closes, 14)
    rsi_signal = None
    if rsi is not None:
        if rsi > 70:
            rsi_signal = "SURACHETÉ"
        elif rsi < 30:
            rsi_signal = "SURVENDU"
        else:
            rsi_signal = "NEUTRE"

    # Bollinger
    bollinger = _bollinger_bands(closes, 20, 2)
    bollinger_signal = None
    if bollinger:
        if current_price > bollinger["upper"]:
            bollinger_signal = (
                "AU-DESSUS de la bande haute (pression haussière ou surachat)"
            )
        elif current_price < bollinger["lower"]:
            bollinger_signal = (
                "EN-DESSOUS de la bande basse (pression baissière ou survente)"
            )
        else:
            bollinger_signal = "DANS les bandes (conditions normales)"

    # Momentum
    momentum = {}
    if len(closes) >= 21:
        ret_1m = (current_price - closes[-21]) / closes[-21] * 100
        momentum["rendement_1mois"] = round(ret_1m, 2)
    if len(closes) >= 63:
        ret_3m = (current_price - closes[-63]) / closes[-63] * 100
        momentum["rendement_3mois"] = round(ret_3m, 2)

    period_high = max(highs)
    period_low = min(lows)
    momentum["plus_haut_6mois"] = round(period_high, 2)
    momentum["plus_bas_6mois"] = round(period_low, 2)
    momentum["distance_du_plus_haut_pct"] = round(
        (current_price - period_high) / period_high * 100, 2
    )

    # Volume
    volume_analysis = {}
    if len(volumes) >= 5:
        avg_vol = sum(volumes) / len(volumes)
        recent_vols = volumes[-5:]
        recent_vol = sum(recent_vols) / len(recent_vols)
        volume_analysis["volume_moyen"] = int(avg_vol)
        volume_analysis["volume_recent_5j"] = int(recent_vol)
        volume_analysis["ratio_vol_recent_vs_moyen"] = (
            round(recent_vol / avg_vol, 2) if avg_vol else None
        )
        if recent_vol > avg_vol * 1.5:
            volume_analysis["signal"] = "Volume ÉLEVÉ (activité inhabituelle)"
        elif recent_vol < avg_vol * 0.5:
            volume_analysis["signal"] = "Volume FAIBLE (manque de conviction)"
        else:
            volume_analysis["signal"] = "Volume normal"

    # Support / Résistance
    sr = _find_support_resistance(highs, lows)

    return {
        "prix_actuel": current_price,
        "nombre_jours_analyse": len(closes),
        "moyennes_mobiles": {
            "sma_20": sma_20,
            "sma_50": sma_50,
            **sma_signals,
        },
        "rsi_14": {"valeur": rsi, "signal": rsi_signal},
        "bollinger_20_2": {**(bollinger or {}), "signal": bollinger_signal},
        "momentum": momentum,
        "volume": volume_analysis,
        "support_resistance": sr,
    }


# ─────────────────────────────────────────────────────────────────────
# Prompt LLM
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un analyste technique expérimenté. Tu reçois des indicateurs
techniques PRÉ-CALCULÉS (pas besoin de les recalculer) et tu produis une lecture
de la situation technique du titre — pas un verdict "achète/vends".

Ta note doit contenir, en français concis :

1. TENDANCE PRIMAIRE — le titre est-il en tendance haussière, baissière, ou
   en range ? Justifie avec les SMA (prix au-dessus/en-dessous, golden cross
   ou death cross).

2. MOMENTUM — la dynamique est-elle en accélération ou essoufflement ?
   Utilise le RSI et les rendements 1 mois / 3 mois. Un RSI extrême
   (>70 ou <30) est un signal à commenter.

3. VOLATILITÉ — les bandes de Bollinger sont-elles larges (incertitude) ou
   serrées (compression → mouvement imminent) ? Le prix touche-t-il les bandes ?

4. VOLUME — le volume récent confirme-t-il ou infirme la tendance ?
   Un mouvement haussier avec volume faible est suspect.

5. NIVEAUX CLÉS — supports et résistances identifiés. Proximité avec les
   plus hauts/bas de la période.

6. SCÉNARIOS TECHNIQUES — formule 2 scénarios :
   - Scénario HAUSSIER : les conditions pour une continuation/reprise
   - Scénario BAISSIER : les conditions pour un retournement/cassure

Reste factuel, cite les chiffres. N'utilise pas de jargon obscur sans
l'expliquer. Pas de conseil d'achat/vente."""


def load_ticker_data(ticker: str) -> dict:
    path = DATA_DIR / f"{ticker.upper()}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Pas de données pour {ticker}. "
            f"Lance d'abord : python collector.py {ticker}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_user_prompt(ticker: str, indicators: dict, meta: dict) -> str:
    prev = get_previous_analysis(ticker, "technical")
    prev_section = ""
    if prev:
        prev_section = (
            "\n\n## TON ANALYSE TECHNIQUE PRÉCÉDENTE\n"
            f"{prev[:1500]}\n"
            "(Fin de l'extrait. Compare avec la situation actuelle.)\n"
        )

    return (
        f"## {meta.get('name', ticker)} ({ticker.upper()})\n"
        f"Secteur : {meta.get('sector')}, Industrie : {meta.get('industry')}\n\n"
        "## INDICATEURS TECHNIQUES CALCULÉS\n"
        f"```json\n{json.dumps(indicators, indent=2, ensure_ascii=False)}\n```\n"
        f"{prev_section}\n"
        "Produis ta note d'analyse technique."
    )


def analyze(ticker: str) -> str:
    data = load_ticker_data(ticker)
    price_data = data.get("price_history_6mo", [])

    print(f"→ Calcul des indicateurs techniques pour {ticker.upper()}...")
    indicators = compute_indicators(price_data)

    if "error" in indicators:
        return f"Erreur : {indicators['error']}"

    mm = indicators["moyennes_mobiles"]
    print(
        f"  RSI: {indicators['rsi_14']['valeur']}, "
        f"SMA20: {mm['sma_20']}, "
        f"SMA50: {mm['sma_50']}"
    )

    meta = {
        "name": data.get("name"),
        "sector": data.get("sector"),
        "industry": data.get("industry"),
    }
    user_prompt = build_user_prompt(ticker, indicators, meta)

    print(f"→ Envoi à {MODEL_NAME} pour interprétation technique...\n")
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
    out_path = DATA_DIR / f"{ticker}_technical.md"
    out_path.write_text(analysis, encoding="utf-8")
    print(f"\n→ Analyse technique sauvegardée dans {out_path}")
    return out_path


if __name__ == "__main__":
    ticker = (
        sys.argv[1]
        if len(sys.argv) > 1
        else input("Ticker à analyser : ").strip()
    )
    analysis = analyze(ticker)

    print("=" * 60)
    print(f"ANALYSE TECHNIQUE — {ticker.upper()}")
    print("=" * 60)
    print(analysis)

    save_output(ticker, analysis)