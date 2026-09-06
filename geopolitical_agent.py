"""
Agent Géopolitique — Veille stratégique avec cache intelligent.

L'économie et la géopolitique sont inséparables : une guerre au Moyen-Orient
fait monter le baril et profite à Total, les droits de douane de Trump pèsent
sur les exportateurs chinois, le plan Chine 2049 oriente les semi-conducteurs
pour 20 ans. Ces méga-tendances ne changent pas tous les jours — d'où un
système de cache avec TTL (time-to-live) par catégorie.

Catégories et TTL :
  - Programmes présidentiels       → 90 jours (change aux élections)
  - Politiques commerciales        → 30 jours (tarifs par rounds)
  - Stratégies nationales LT       → 180 jours (plans décennaux, très stables)
  - Conflits majeurs               → 7 jours (situation volatile)

Sortie : data/geopolitical/geopolitical_brief_YYYYMMDD.md
         data/geopolitical/<catégorie>_cache.json (données brutes + TTL)

Usage :
  python geopolitical_agent.py              → utilise le cache si valide
  python geopolitical_agent.py --force      → force le refresh de tout

Pré-requis : pip install ollama tavily-python python-dotenv
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import ollama
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv(Path(__file__).parent / ".env")

MODEL_NAME = "glm-4.7-flash:latest"
DATA_DIR = Path(__file__).parent / "data"
GEO_DIR = DATA_DIR / "geopolitical"
GEO_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Configuration des catégories géopolitiques
# ─────────────────────────────────────────────────────────────────────

GEOPOLITICAL_CATEGORIES = {
    "programmes_presidentiels": {
        "label": "Programmes présidentiels (grandes puissances)",
        "ttl_days": 90,
        "queries": [
            "Trump administration economic policy tariffs trade agenda 2025 2026",
            "US president fiscal policy infrastructure spending budget priorities",
            "European Union political leadership economic reform industrial policy",
            "China Xi Jinping economic strategy technology self-sufficiency",
        ],
        "max_results_per_query": 5,
    },
    "politiques_commerciales": {
        "label": "Politiques commerciales et droits de douane",
        "ttl_days": 30,
        "queries": [
            "US tariffs trade war China Europe retaliatory duties latest",
            "international trade agreements sanctions economic restrictions 2025 2026",
            "supply chain reshoring nearshoring friendshoring trends",
        ],
        "max_results_per_query": 5,
    },
    "strategies_nationales": {
        "label": "Stratégies nationales long terme",
        "ttl_days": 180,
        "queries": [
            "China Made in China 2025 technology plan semiconductor self-sufficiency 2035 2049",
            "European Green Deal industrial strategy digital sovereignty",
            "India Make in India economic strategy manufacturing growth",
            "US CHIPS Act IRA industrial policy semiconductor reshoring",
        ],
        "max_results_per_query": 4,
    },
    "conflits_majeurs": {
        "label": "Conflits géopolitiques majeurs et impacts marchés",
        "ttl_days": 7,
        "queries": [
            "Ukraine Russia war economic impact energy prices Europe sanctions latest",
            "Middle East Iran Israel conflict oil prices supply disruption",
            "Taiwan China tension semiconductor supply chain geopolitical risk",
            "geopolitical risk financial markets impact investment outlook",
        ],
        "max_results_per_query": 5,
    },
}


# ─────────────────────────────────────────────────────────────────────
# Gestion du cache
# ─────────────────────────────────────────────────────────────────────

def is_cache_valid(category_key: str) -> bool:
    """Vérifie si le cache d'une catégorie est encore dans son TTL."""
    cache_path = GEO_DIR / f"{category_key}_cache.json"
    if not cache_path.exists():
        return False

    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(cache["fetched_at"])
        ttl_days = GEOPOLITICAL_CATEGORIES[category_key]["ttl_days"]
        return datetime.now() - fetched_at < timedelta(days=ttl_days)
    except (KeyError, ValueError, json.JSONDecodeError):
        return False


def load_cache(category_key: str) -> dict | None:
    """Charge le cache d'une catégorie."""
    cache_path = GEO_DIR / f"{category_key}_cache.json"
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def fetch_category(category_key: str, config: dict) -> dict:
    """Récupère les données pour une catégorie via Tavily."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY absente. Vérifie ton fichier poc/.env")
    client = TavilyClient(api_key=api_key)

    all_results = []
    for query in config["queries"]:
        print(f"    → {query[:65]}...")
        try:
            res = client.search(
                query=query,
                search_depth="advanced",
                topic="news",
                days=min(config["ttl_days"], 30),  # Tavily plafonne à 30 jours
                max_results=config["max_results_per_query"],
            )
            for article in res.get("results", []):
                all_results.append({
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "content": (article.get("content") or "")[:600],
                    "query_source": query[:50],
                })
        except Exception as e:
            all_results.append({"error": str(e), "query_source": query[:50]})

    cache_data = {
        "category": category_key,
        "label": config["label"],
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "ttl_days": config["ttl_days"],
        "expires_at": (
            datetime.now() + timedelta(days=config["ttl_days"])
        ).isoformat(timespec="seconds"),
        "result_count": len([r for r in all_results if "error" not in r]),
        "results": all_results,
    }

    # Sauvegarde du cache
    cache_path = GEO_DIR / f"{category_key}_cache.json"
    cache_path.write_text(
        json.dumps(cache_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return cache_data


def fetch_all_categories(force_refresh: bool = False) -> dict:
    """Récupère toutes les catégories en utilisant le cache quand valide."""
    all_data = {}
    stats = {"cached": 0, "refreshed": 0}

    for key, config in GEOPOLITICAL_CATEGORIES.items():
        print(f"\n  [{config['label']}]")

        if not force_refresh and is_cache_valid(key):
            cache = load_cache(key)
            if cache:
                expires = cache.get("expires_at", "?")[:10]
                count = cache.get("result_count", "?")
                print(f"    ✓ Cache valide ({count} résultats, expire {expires})")
                all_data[key] = cache
                stats["cached"] += 1
                continue

        print(f"    ⟳ Cache expiré ou absent — requête Tavily...")
        all_data[key] = fetch_category(key, config)
        n = all_data[key].get("result_count", 0)
        print(f"    ✓ {n} résultats récupérés et mis en cache.")
        stats["refreshed"] += 1

    print(f"\n  Bilan : {stats['cached']} catégories en cache, "
          f"{stats['refreshed']} rafraîchies.")
    return all_data


# ─────────────────────────────────────────────────────────────────────
# Prompt LLM
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un analyste géopolitique senior dans une société de gestion
d'actifs. Tu reçois des informations sur les grandes tendances géopolitiques
mondiales et tu produis une note de veille stratégique — jamais un conseil
d'investissement.

Ta note doit être en français, structurée dans cet ordre exact :

## CARTE DES POUVOIRS
Qui dirige quoi ? Position des grandes puissances (US, Chine, Europe, Russie)
et l'orientation économique de chaque leadership. 2-3 phrases par puissance.

## POLITIQUES COMMERCIALES EN COURS
Droits de douane actifs, sanctions, accords commerciaux. Quels flux sont
perturbés ? Quels secteurs sont directement touchés ? Sois spécifique
(cite les taux de tarifs, les secteurs ciblés, les pays sanctionnés).

## STRATÉGIES NATIONALES LONG TERME
Les grands plans industriels/technologiques : CHIPS Act (US), Made in China
2025/2035, Green Deal (Europe), Make in India. Ce qui avance, ce qui est
bloqué, ce qui a été renforcé ou abandonné.

## CONFLITS ET TENSIONS ACTIFS
Les foyers de tension qui pèsent sur les marchés. Pour CHAQUE conflit :
- la situation actuelle
- l'impact économique DIRECT (énergie, supply chain, devises, matières premières)
- les secteurs/zones les plus exposés

## MÉGA-TENDANCES À HORIZON 2-5 ANS
Les mouvements de fond : relocalisation industrielle, course à l'IA et aux
semi-conducteurs, transition énergétique, démondialisation partielle,
recomposition des alliances commerciales. Distingue ce qui est structurel
(irréversible) de ce qui est conjoncturel (réversible si le contexte change).

## IMPLICATIONS SECTORIELLES
Sans nommer de titres, quels SECTEURS sont favorisés ou pénalisés par ce
contexte géopolitique ? Donne le lien causal explicite (ex: "la défense
européenne profite du réarmement post-Ukraine car les budgets OTAN augmentent
vers 2% du PIB").

Reste factuel, cite tes sources quand possible. Signale ce qui est incertain
ou spéculatif. Pas de conseil d'achat/vente."""


def build_user_prompt(all_data: dict) -> str:
    sections = []
    for key, data in all_data.items():
        label = data.get("label", key)
        results = data.get("results", [])
        valid_results = [r for r in results if "error" not in r]
        cached_date = data.get("fetched_at", "")[:10]

        if valid_results:
            # On limite pour ne pas exploser le contexte
            trimmed = valid_results[:15]
            formatted = json.dumps(trimmed, indent=2, ensure_ascii=False)
            sections.append(
                f"### {label} (données du {cached_date})\n"
                f"```json\n{formatted}\n```"
            )
        else:
            sections.append(f"### {label}\n(Aucune donnée disponible)")

    return (
        "## DONNÉES GÉOPOLITIQUES COLLECTÉES\n\n"
        + "\n\n".join(sections)
        + "\n\nRédige maintenant ta note de veille géopolitique en suivant "
        "strictement la structure demandée."
    )


# ─────────────────────────────────────────────────────────────────────
# Génération et sauvegarde
# ─────────────────────────────────────────────────────────────────────

def generate_brief(force_refresh: bool = False) -> str:
    print("→ Récupération des données géopolitiques...\n")
    all_data = fetch_all_categories(force_refresh)

    user_prompt = build_user_prompt(all_data)

    print(f"\n→ Envoi à {MODEL_NAME} pour la synthèse géopolitique...\n")
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


def save_brief(brief: str) -> Path:
    day = datetime.now().strftime("%Y%m%d")
    header = (
        f"# Veille géopolitique — {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
        f"*Générée par l'agent géopolitique (modèle : {MODEL_NAME})*\n\n---\n\n"
    )
    brief_path = GEO_DIR / f"geopolitical_brief_{day}.md"
    brief_path.write_text(header + brief, encoding="utf-8")
    return brief_path


# ─────────────────────────────────────────────────────────────────────
# Fonction utilitaire pour le manager_agent
# ─────────────────────────────────────────────────────────────────────

def load_latest_geopolitical() -> str | None:
    """Récupère la note géopolitique la plus récente (appelé par le Gérant)."""
    briefs = sorted(GEO_DIR.glob("geopolitical_brief_*.md"))
    if not briefs:
        return None
    return briefs[-1].read_text(encoding="utf-8")


if __name__ == "__main__":
    force = "--force" in sys.argv
    if force:
        print("⚠ Mode --force : tous les caches seront rafraîchis.\n")

    brief = generate_brief(force_refresh=force)

    print("=" * 60)
    print("VEILLE GÉOPOLITIQUE")
    print("=" * 60)
    print(brief)

    saved = save_brief(brief)
    print(f"\n→ Note géopolitique sauvegardée dans {saved}")