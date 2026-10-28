"""DuckDuckGo search + page fetch. Returns condensed text for the LLM to summarize.

Standalone test:  python -m tools.web_search "weather in Paris"
"""
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS
except ImportError:  # older package name
    from duckduckgo_search import DDGS

log = logging.getLogger("jarvis.web")

SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web (DuckDuckGo) and read the top pages. Use for news, weather, sports, prices, "
            "recent events, and any current or factual information you are not certain about."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": (
                    "A concise, unambiguous search query in English. Always spell out full place names "
                    "with country (e.g. 'weather in Paris, France today') and include the year for news."
                ),
            },
            },
            "required": ["query"],
        },
    },
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


def _fetch_text(url: str, timeout: float, limit: int) -> str:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=timeout)
        r.raise_for_status()
        if "html" not in r.headers.get("content-type", "html"):
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg"]):
            tag.decompose()
        root = soup.find("article") or soup.find("main") or soup.body or soup
        paras = [p.get_text(" ", strip=True) for p in root.find_all(["p", "li", "h1", "h2", "h3"])]
        text = " ".join(p for p in paras if len(p) > 40)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:limit]
    except Exception as e:
        log.debug("fetch failed for %s: %s", url, e)
        return ""


def run(query: str, cfg: dict) -> str:
    query = (query or "").strip()
    if not query:
        return "No query given."
    max_results = int(cfg.get("max_results", 5))
    pages = int(cfg.get("pages_to_fetch", 2))
    limit = int(cfg.get("chars_per_page", 1800))
    timeout = float(cfg.get("timeout_s", 8))

    log.info("web search: %r", query)
    results = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"Search failed ({e}). Tell the user the search is unavailable."
    if not results:
        return "No results found."

    urls = [r.get("href") or r.get("url") for r in results[:pages]]
    with ThreadPoolExecutor(max_workers=max(1, pages)) as pool:
        texts = list(pool.map(lambda u: _fetch_text(u, timeout, limit) if u else "", urls))

    out = [f"Search results for: {query}"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        snippet = r.get("body", "")
        out.append(f"[{i}] {title}: {snippet}")
    for i, (u, t) in enumerate(zip(urls, texts), 1):
        if t:
            out.append(f"\nPage excerpt [{i}] ({u}):\n{t}")
    return "\n".join(out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    q = " ".join(sys.argv[1:]) or "latest news"
    print(run(q, {}))
