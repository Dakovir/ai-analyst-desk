"""
Agent Earnings Call — Récupère et analyse les conférences de résultats.

Les earnings calls sont le moment où le management parle : chiffres réels vs
consensus, guidance future, ton de la direction, questions des analystes.
C'est de l'info publique (publiée en PDF ou transcript sur Seeking Alpha,
Motley Fool, sites IR) mais dense et souvent longue — d'où l'intérêt d'un
agent qui résume les points clés.

Cache par ticker + trimestre : un earnings call publié ne change plus,
inutile de re-requêter Tavily pour le même trimestre.

Sortie : data/<TICKER>_earnings.md  (analyse GLM)
         data/<TICKER>_earnings_raw.json  (données brutes + métadonnées cache)

Usage :
  python earnings_agent.py AAPL
  python earnings_agent.py AAPL --force    → ignore le cache

Pré-requis : pip install ollama tavily-python python-dotenv
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import ollama
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv(Path(__file__).parent / ".env")

MODEL_NAME = "glm-4.7-flash:latest"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Gestion du cache trimestriel
# ─────────────────────────────────────────────────────────────────────

def get_current_quarter() -> str:
    """Retourne le trimestre courant (ex: '2026Q3')."""
    now = datetime.now()
    quarter = (now.month - 1) // 3 + 1
    return f"{now.year}Q{quarter}"


def is_cache_valid(ticker: str) -> bool:
    """Le cache est valide si on a déjà les données pour ce trimestre.
    Les earnings du Q2 tombent pendant le Q3, donc on vérifie par trimestre
    courant — un changement de trimestre déclenche un nouveau fetch."""
    cache_path = DATA_DIR / f"{ticker.upper()}_earnings_raw.json"
    if not cache_path.exists():
        return False

    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        return cache.get("quarter") == get_current_quarter()
    except (json.JSONDecodeError, KeyError):
        return False


# ─────────────────────────────────────────────────────────────────────
# Récupération via Tavily
# ─────────────────────────────────────────────────────────────────────

def fetch_earnings_data(ticker: str,
                        company_name: str | None = None) -> list[dict]:
    """Recherche les transcripts et résumés d'earnings calls via Tavily."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY absente. Vérifie ton fichier poc/.env")
    client = TavilyClient(api_key=api_key)

    name = company_name or ticker

    # Deux requêtes complémentaires : transcript brut + couverture analystes
    queries = [
        f"{name} {ticker} earnings call transcript quarterly results latest",
        f"{name} {ticker} quarterly earnings results revenue EPS guidance analyst",
    ]

    all_results = []
    for query in queries:
        print(f"  → {query[:65]}...")
        try:
            res = client.search(
                query=query,
                search_depth="advanced",
                topic="news",
                days=30,
                max_results=6,
            )
            for article in res.get("results", []):
                all_results.append({
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "content": (article.get("content") or "")[:800],
                    "published_date": article.get("published_date"),
                })
        except Exception as e:
            all_results.append({"error": str(e)})

    # Déduplications par URL
    seen_urls = set()
    unique = []
    for r in all_results:
        url = r.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(r)
        elif not url:
            unique.append(r)

    return unique


def load_ticker_data(ticker: str) -> dict:
    """Récupère le nom de la société depuis le JSON du collecteur."""
    path = DATA_DIR / f"{ticker.upper()}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


# ─────────────────────────────────────────────────────────────────────
# Prompt LLM
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un analyste sell-side spécialisé dans l'analyse des
conférences de résultats (earnings calls). Tu reçois des extraits de transcripts
et d'articles couvrant les derniers résultats d'une société.

Ta note doit contenir, en français concis :

1. DERNIERS RÉSULTATS PUBLIÉS — chiffres clés annoncés : chiffre d'affaires,
   BPA (EPS), marges principales, croissance. Par rapport au consensus (beat /
   miss / en ligne). Indique le trimestre et la date de publication.

2. MESSAGES CLÉS DU MANAGEMENT — ce que la direction a mis en avant pendant
   le call, le ton général (confiant / prudent / défensif / évasif sur certains
   sujets), les priorités stratégiques affichées.

3. GUIDANCE — prévisions données par le management pour le ou les prochains
   trimestres/l'exercice. La guidance a-t-elle été relevée, maintenue ou
   abaissée par rapport à la précédente ?

4. QUESTIONS DES ANALYSTES — les sujets qui ont concentré le plus de questions
   (ça révèle les inquiétudes du marché). Résume 2-3 échanges significatifs.

5. SIGNAUX À SURVEILLER — ce qui pourrait surprendre au prochain trimestre,
   basé sur ce que le management a dit ou n'a PAS dit (les non-réponses sont
   parfois le signal le plus fort).

6. SOURCES CITÉES — liste des articles et transcripts utilisés (titre + URL).

Si les informations sont insuffisantes pour certaines sections, signale-le
explicitement et explique pourquoi (pas encore de publication ce trimestre,
données indisponibles, etc.). Pas de conseil d'achat/vente.

IMPORTANT — REPÉRAGE TEMPOREL :
- Cite TOUJOURS le trimestre ET l'année exacte des résultats (ex: "Q2 FY2026",
  jamais "le dernier trimestre" sans date).
- Attention aux exercices fiscaux décalés : certaines sociétés (Apple, e.l.f.
  Beauty, Nike, etc.) ont un exercice qui ne suit pas l'année civile. Si les
  dates des résultats ne correspondent pas aux trimestres civils, SIGNALE que
  l'exercice est décalé et précise la correspondance (ex: "Q1 FY2027 = avril-
  juin 2026 en année civile").
- Vérifie les dates des articles : un article de plus de 60 jours est
  potentiellement OBSOLÈTE — signale-le explicitement.
- Ne projette JAMAIS un trimestre futur comme s'il était déjà publié. Si les
  résultats du trimestre en cours ne sont pas encore sortis, dis-le."""


def build_user_prompt(ticker: str, articles: list[dict]) -> str:
    valid = [a for a in articles if "error" not in a]

    if not valid:
        return (
            f"Aucun transcript ou article sur les earnings de {ticker} trouvé "
            "sur les 30 derniers jours. Signale-le dans ta note et indique "
            "quand le prochain reporting est attendu si tu le sais."
        )

    formatted = []
    for i, article in enumerate(valid, 1):
        formatted.append(
            f"[{i}] {article.get('title', 'Sans titre')}\n"
            f"    URL: {article.get('url')}\n"
            f"    Publié: {article.get('published_date', 'date inconnue')}\n"
            f"    Extrait: {article.get('content', '')}\n"
        )

    return (
        f"Voici les {len(valid)} sources trouvées sur les earnings calls "
        f"de {ticker.upper()} :\n\n"
        + "\n".join(formatted)
        + "\n\nProduis ta note d'analyse des earnings calls."
    )


# ─────────────────────────────────────────────────────────────────────
# Analyse principale
# ─────────────────────────────────────────────────────────────────────

def analyze(ticker: str, force_refresh: bool = False) -> tuple[str, list[dict]]:
    data = load_ticker_data(ticker)
    company_name = data.get("name")

    # Vérifier le cache
    if not force_refresh and is_cache_valid(ticker):
        print(f"  ✓ Cache earnings valide pour {ticker.upper()} "
              f"(trimestre {get_current_quarter()})")
        cache = json.loads(
            (DATA_DIR / f"{ticker.upper()}_earnings_raw.json")
            .read_text(encoding="utf-8")
        )
        articles = cache.get("articles", [])
    else:
        print(f"→ Récupération des earnings calls pour {ticker.upper()} "
              "via Tavily...")
        articles = fetch_earnings_data(ticker, company_name)
        valid_count = len([a for a in articles if "error" not in a])
        print(f"  {valid_count} articles uniques récupérés.")

    user_prompt = build_user_prompt(ticker, articles)

    print(f"→ Envoi à {MODEL_NAME} pour analyse des earnings...\n")
    date_ctx = f"Date du jour : {datetime.now().strftime('%d/%m/%Y %H:%M')}.\n\n"
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": date_ctx + SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.3},
    )
    return response["message"]["content"], articles


def save_output(ticker: str, analysis: str, articles: list[dict]) -> None:
    ticker = ticker.upper()

    md_path = DATA_DIR / f"{ticker}_earnings.md"
    md_path.write_text(analysis, encoding="utf-8")

    raw_path = DATA_DIR / f"{ticker}_earnings_raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
                "quarter": get_current_quarter(),
                "ticker": ticker,
                "articles": articles,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n→ Analyse earnings sauvegardée dans {md_path}")
    print(f"→ Données brutes sauvegardées dans {raw_path}")


if __name__ == "__main__":
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--force"]
    ticker = args[0] if args else input("Ticker à analyser : ").strip()

    if force:
        print("⚠ Mode --force : cache ignoré.\n")

    analysis, articles = analyze(ticker, force_refresh=force)

    print("=" * 60)
    print(f"NOTE EARNINGS — {ticker.upper()}")
    print("=" * 60)
    print(analysis)

    save_output(ticker, analysis, articles)