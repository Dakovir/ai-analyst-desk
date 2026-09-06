"""
Outil de mise à jour multi-ticker (batch) — lance le pipeline pour
chaque ticker d'une watchlist.

Usage :
  python update.py                   → lit watchlist.txt, pipeline complet
  python update.py --quick           → skip macro/geo/earnings (déjà fait)
  python update.py --file ma_liste.txt  → watchlist personnalisée
  python update.py AAPL MSFT TSLA    → tickers en ligne de commande

Le fichier watchlist.txt contient un ticker par ligne.
Les lignes vides et les commentaires (#) sont ignorés.

Optimisation : les étapes partagées (macro, géopolitique) ne tournent
qu'UNE SEULE FOIS, avant la boucle per-ticker.
"""

import sys
import time
from datetime import datetime
from pathlib import Path


WATCHLIST_FILE = Path(__file__).parent / "watchlist.txt"


def load_watchlist(filepath: Path) -> list[str]:
    """Charge la watchlist depuis un fichier texte."""
    if not filepath.exists():
        print(f"  ✗ Fichier watchlist introuvable : {filepath}")
        print("  Crée un fichier watchlist.txt avec un ticker par ligne.")
        return []

    tickers = []
    for line in filepath.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Prend le premier mot (ignore les commentaires après le ticker)
        ticker = line.split()[0].upper()
        if ticker not in tickers:
            tickers.append(ticker)
    return tickers


def run_shared_steps(skip_macro: bool, skip_geo: bool) -> dict:
    """Lance les étapes partagées (macro, géopolitique) une seule fois.
    Retourne un dict avec le statut de chaque étape."""
    results = {"macro": "skipped", "geo": "skipped"}

    if not skip_macro:
        print(f"\n{'=' * 60}")
        print("  ÉTAPE PARTAGÉE — STRATÉGISTE MACRO")
        print(f"{'=' * 60}\n")
        try:
            from macro_agent import generate_brief, save_brief
            brief, indicators, macro_news = generate_brief()
            save_brief(brief, indicators, macro_news)
            print("  ✓ Note macro sauvegardée.")
            results["macro"] = "ok"
        except Exception as e:
            print(f"  ⚠ Erreur macro : {e}")
            results["macro"] = f"erreur: {e}"

    if not skip_geo:
        print(f"\n{'=' * 60}")
        print("  ÉTAPE PARTAGÉE — ANALYSTE GÉOPOLITIQUE")
        print(f"{'=' * 60}\n")
        try:
            from geopolitical_agent import (
                generate_brief as geo_brief,
                save_brief as geo_save,
            )
            geo_text = geo_brief(force_refresh=False)
            geo_save(geo_text)
            print("  ✓ Note géopolitique sauvegardée.")
            results["geo"] = "ok"
        except Exception as e:
            print(f"  ⚠ Erreur géopolitique : {e}")
            results["geo"] = f"erreur: {e}"

    return results


def run_ticker_pipeline(
    ticker: str,
    skip_earnings: bool = False,
) -> dict:
    """Lance les étapes per-ticker (sans macro/geo, déjà faites).
    Retourne un dict avec le statut de chaque étape."""
    results = {}

    # 1. Collecteur
    print(f"\n  [1/8] Collecteur...")
    try:
        from collector import fetch_ticker_data, save_data
        data = fetch_ticker_data(ticker)
        save_data(data)
        results["collector"] = "ok"
    except Exception as e:
        results["collector"] = f"erreur: {e}"
        return results  # Bloquant

    # 2. Fondamental
    print(f"  [2/8] Fondamental...")
    try:
        from fundamental_agent import analyze as fund_analyze, DATA_DIR
        analysis = fund_analyze(ticker)
        out = DATA_DIR / f"{ticker}_fundamental.md"
        out.write_text(analysis, encoding="utf-8")
        results["fundamental"] = "ok"
    except Exception as e:
        results["fundamental"] = f"erreur: {e}"
        return results  # Bloquant

    # 3. Sentiment
    print(f"  [3/8] Sentiment...")
    try:
        from sentiment_agent import analyze as sent_analyze, save_output as sent_save
        sent_analysis, news = sent_analyze(ticker)
        sent_save(ticker, sent_analysis, news)
        results["sentiment"] = "ok"
    except Exception as e:
        results["sentiment"] = f"erreur: {e}"
        return results  # Bloquant

    # 4. Bilan
    print(f"  [4/8] Bilan & états financiers...")
    try:
        from balance_sheet_agent import analyze as bs_analyze, save_output as bs_save
        bs_analysis = bs_analyze(ticker)
        bs_save(ticker, bs_analysis)
        results["balance_sheet"] = "ok"
    except Exception as e:
        results["balance_sheet"] = f"erreur: {e}"

    # 5. Technique
    print(f"  [5/8] Technique...")
    try:
        from technical_agent import analyze as tech_analyze, save_output as tech_save
        tech_analysis = tech_analyze(ticker)
        tech_save(ticker, tech_analysis)
        results["technical"] = "ok"
    except Exception as e:
        results["technical"] = f"erreur: {e}"

    # 6. Cycle
    print(f"  [6/8] Cycle sectoriel...")
    try:
        from cycle_agent import analyze as cycle_analyze, save_output as cycle_save
        cycle_analysis = cycle_analyze(ticker)
        cycle_save(ticker, cycle_analysis)
        results["cycle"] = "ok"
    except Exception as e:
        results["cycle"] = f"erreur: {e}"

    # 7. Pairs
    print(f"  [7/8] Pairs / Comparatif...")
    try:
        from peers_agent import analyze as peers_analyze, save_output as peers_save
        peers_analysis = peers_analyze(ticker)
        peers_save(ticker, peers_analysis)
        results["peers"] = "ok"
    except Exception as e:
        results["peers"] = f"erreur: {e}"

    # 8. Earnings
    if skip_earnings:
        results["earnings"] = "skipped"
    else:
        print(f"  [8/8] Earnings...")
        try:
            from earnings_agent import (
                analyze as earn_analyze,
                save_output as earn_save,
            )
            earn_analysis, articles = earn_analyze(ticker)
            earn_save(ticker, earn_analysis, articles)
            results["earnings"] = "ok"
        except Exception as e:
            results["earnings"] = f"erreur: {e}"

    # Synthèse
    print(f"  [SYNTHÈSE] Gérant...")
    try:
        from manager_agent import synthesize, save_note
        note = synthesize(ticker)
        saved = save_note(ticker, note)
        results["synthesis"] = "ok"
        results["synthesis_file"] = str(saved)
    except Exception as e:
        results["synthesis"] = f"erreur: {e}"

    return results


def print_summary(all_results: dict, elapsed: float) -> None:
    """Affiche un résumé final de tous les tickers traités."""
    print(f"\n{'#' * 60}")
    print(f"  RÉSUMÉ DU BATCH — {len(all_results)} ticker(s)")
    print(f"  Durée totale : {elapsed:.0f} secondes")
    print(f"{'#' * 60}\n")

    ok_count = 0
    fail_count = 0

    for ticker, results in all_results.items():
        synth = results.get("synthesis", "non lancé")
        if synth == "ok":
            ok_count += 1
            status = "✓"
            detail = results.get("synthesis_file", "")
        else:
            fail_count += 1
            status = "✗"
            detail = synth

        # Compter les analyses réussies
        agents_ok = sum(1 for k, v in results.items()
                        if v == "ok" and k not in ("synthesis", "synthesis_file"))
        agents_total = sum(1 for k in results
                          if k not in ("synthesis", "synthesis_file"))

        print(f"  {status} {ticker:8s} — {agents_ok}/{agents_total} agents OK"
              f"  {'→ ' + detail if detail else ''}")

    print(f"\n  Total : {ok_count} synthèse(s) OK, {fail_count} en erreur")


def main():
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    quick = "--quick" in flags
    skip_macro = "--skip-macro" in flags or quick
    skip_geo = "--skip-geo" in flags or quick
    skip_earnings = "--skip-earnings" in flags or quick

    # Charger les tickers
    watchlist_path = WATCHLIST_FILE
    for i, f in enumerate(flags):
        if f == "--file" and i + 1 < len(flags):
            # Chercher le prochain argument non-flag
            pass
    # Chercher --file <path>
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--file" and i < len(sys.argv) - 1:
            watchlist_path = Path(sys.argv[i + 1])
            break

    if args:
        # Tickers passés en ligne de commande
        tickers = [t.upper() for t in args if not t.startswith("--")]
        # Retirer le fichier après --file
        for i, a in enumerate(sys.argv[1:], 1):
            if a == "--file" and i < len(sys.argv) - 1:
                next_arg = sys.argv[i + 1]
                tickers = [t for t in tickers if t != next_arg.upper()]
    else:
        tickers = load_watchlist(watchlist_path)

    if not tickers:
        print("Aucun ticker à traiter.")
        print("Usage : python update.py AAPL MSFT")
        print("   ou : python update.py (avec watchlist.txt)")
        return

    start_time = time.time()

    print(f"\n{'#' * 60}")
    print(f"  BATCH UPDATE — {len(tickers)} ticker(s)")
    print(f"  {', '.join(tickers)}")
    print(f"  {datetime.now().strftime('%d/%m/%Y à %H:%M')}")
    if quick:
        print("  Mode --quick : macro/geo/earnings skippés")
    print(f"{'#' * 60}")

    # Étapes partagées (une seule fois)
    shared = run_shared_steps(skip_macro, skip_geo)

    # Boucle per-ticker
    all_results = {}
    for i, ticker in enumerate(tickers, 1):
        print(f"\n{'─' * 60}")
        print(f"  TICKER {i}/{len(tickers)} — {ticker}")
        print(f"{'─' * 60}")

        results = run_ticker_pipeline(ticker, skip_earnings=skip_earnings)
        all_results[ticker] = results

    elapsed = time.time() - start_time
    print_summary(all_results, elapsed)


if __name__ == "__main__":
    main()