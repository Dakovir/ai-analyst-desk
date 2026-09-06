"""
Orchestrateur du pipeline complet — lance tous les agents dans le bon ordre.

Usage :
  python run_pipeline.py AAPL          → pipeline complet
  python run_pipeline.py AAPL --skip-macro   → sans macro (si déjà fait aujourd'hui)
  python run_pipeline.py AAPL --skip-geo     → sans géopolitique
  python run_pipeline.py AAPL --only-synthesis → relance seulement le gérant

L'ordre est important :
  1. Collecteur        → data/<TICKER>.json
  2. Fondamental       → data/<TICKER>_fundamental.md
  3. Sentiment         → data/<TICKER>_sentiment.md
  4. Macro             → data/macro_brief_YYYYMMDD.md  (indépendant du ticker)
  5. Géopolitique      → data/geopolitical/geopolitical_brief_YYYYMMDD.md (cache TTL)
  6. Earnings          → data/<TICKER>_earnings.md
  7. Gérant/Synthèse   → data/<TICKER>_synthesis_YYYYMMDD_HHMM.md

Chaque étape affiche un bandeau clair pour suivre la progression.
Si une étape optionnelle échoue, le pipeline continue (le gérant s'adapte).
"""

import sys
import time
from datetime import datetime


def banner(step: int, total: int, title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  ÉTAPE {step}/{total} — {title}")
    print(f"{'=' * 60}\n")


def run_pipeline(ticker: str, skip_macro: bool = False,
                 skip_geo: bool = False, only_synthesis: bool = False) -> None:
    ticker = ticker.upper()
    start_time = time.time()
    total_steps = 7

    print(f"\n{'#' * 60}")
    print(f"  PIPELINE D'ANALYSE — {ticker}")
    print(f"  Lancé le {datetime.now().strftime('%d/%m/%Y à %H:%M')}")
    print(f"{'#' * 60}")

    if only_synthesis:
        print("\n  Mode --only-synthesis : saut direct au gérant.\n")
        banner(7, total_steps, "GÉRANT / SYNTHÈSE")
        try:
            from manager_agent import synthesize, save_note
            note = synthesize(ticker)
            print(note)
            saved = save_note(ticker, note)
            print(f"\n→ Note finale : {saved}")
        except Exception as e:
            print(f"  ✗ ERREUR gérant : {e}")
        elapsed = time.time() - start_time
        print(f"\n✓ Pipeline terminé en {elapsed:.0f}s")
        return

    # --- Étape 1 : Collecteur ---
    banner(1, total_steps, "COLLECTEUR (yfinance)")
    try:
        from collector import fetch_ticker_data, save_data, display_summary
        data = fetch_ticker_data(ticker)
        display_summary(data)
        save_data(data)
        print("  ✓ Données collectées.")
    except Exception as e:
        print(f"  ✗ ERREUR collecteur : {e}")
        print("  Le pipeline s'arrête (les autres agents ont besoin des données).")
        return

    # --- Étape 2 : Fondamental ---
    banner(2, total_steps, "ANALYSTE FONDAMENTAL")
    try:
        from fundamental_agent import analyze as fund_analyze
        analysis = fund_analyze(ticker)
        from fundamental_agent import DATA_DIR as FUND_DIR
        out = FUND_DIR / f"{ticker}_fundamental.md"
        out.write_text(analysis, encoding="utf-8")
        print(f"  ✓ Analyse fondamentale sauvegardée dans {out.name}")
    except Exception as e:
        print(f"  ✗ ERREUR fondamental : {e}")
        print("  Le pipeline s'arrête (le gérant a besoin du fondamental).")
        return

    # --- Étape 3 : Sentiment ---
    banner(3, total_steps, "ANALYSTE SENTIMENT")
    try:
        from sentiment_agent import analyze as sent_analyze, save_output as sent_save
        sent_analysis, news = sent_analyze(ticker)
        sent_save(ticker, sent_analysis, news)
        print("  ✓ Analyse sentiment sauvegardée.")
    except Exception as e:
        print(f"  ✗ ERREUR sentiment : {e}")
        print("  Le pipeline s'arrête (le gérant a besoin du sentiment).")
        return

    # --- Étape 4 : Macro (optionnel) ---
    banner(4, total_steps, "STRATÉGISTE MACRO")
    if skip_macro:
        print("  ⏭ Skip (--skip-macro). Le gérant utilisera la note la plus récente.")
    else:
        try:
            from macro_agent import generate_brief, save_brief
            brief, indicators, macro_news = generate_brief()
            save_brief(brief, indicators, macro_news)
            print("  ✓ Note macro sauvegardée.")
        except Exception as e:
            print(f"  ⚠ Erreur macro (non bloquante) : {e}")
            print("  Le gérant fonctionnera sans ou avec une note précédente.")

    # --- Étape 5 : Géopolitique (optionnel, avec cache) ---
    banner(5, total_steps, "ANALYSTE GÉOPOLITIQUE")
    if skip_geo:
        print("  ⏭ Skip (--skip-geo). Le gérant utilisera la note la plus récente.")
    else:
        try:
            from geopolitical_agent import generate_brief as geo_brief, save_brief as geo_save
            geo_text = geo_brief(force_refresh=False)  # Utilise le cache
            geo_save(geo_text)
            print("  ✓ Note géopolitique sauvegardée.")
        except Exception as e:
            print(f"  ⚠ Erreur géopolitique (non bloquante) : {e}")
            print("  Le gérant fonctionnera sans contexte géopolitique.")

    # --- Étape 6 : Earnings (optionnel, avec cache) ---
    banner(6, total_steps, "ANALYSTE EARNINGS CALL")
    try:
        from earnings_agent import analyze as earn_analyze, save_output as earn_save
        earn_analysis, articles = earn_analyze(ticker)
        earn_save(ticker, earn_analysis, articles)
        print("  ✓ Analyse earnings sauvegardée.")
    except Exception as e:
        print(f"  ⚠ Erreur earnings (non bloquante) : {e}")
        print("  Le gérant fonctionnera sans les earnings.")

    # --- Étape 7 : Gérant / Synthèse ---
    banner(7, total_steps, "GÉRANT / SYNTHÈSE FINALE")
    try:
        from manager_agent import synthesize, save_note
        note = synthesize(ticker)
        print("=" * 60)
        print(f"NOTE DE SYNTHÈSE — {ticker}")
        print("=" * 60)
        print(note)
        saved = save_note(ticker, note)
        print(f"\n→ Note finale : {saved}")
    except Exception as e:
        print(f"  ✗ ERREUR gérant : {e}")

    elapsed = time.time() - start_time
    print(f"\n{'#' * 60}")
    print(f"  ✓ PIPELINE TERMINÉ en {elapsed:.0f} secondes")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    ticker = args[0] if args else input("Ticker à analyser : ").strip()

    run_pipeline(
        ticker,
        skip_macro="--skip-macro" in flags,
        skip_geo="--skip-geo" in flags,
        only_synthesis="--only-synthesis" in flags,
    )