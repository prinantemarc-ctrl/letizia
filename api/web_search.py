"""
Recherche web en complément de l'index RAG.
Priorité aux pages visit-corsica.com, avec fetches parallèles.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import httpx
import trafilatura

logger = logging.getLogger(__name__)

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None


def _is_visit_corsica(url: str) -> bool:
    try:
        return "visit-corsica.com" in urlparse(url).netloc.lower()
    except Exception:
        return False


def _http_url(url: str) -> bool:
    try:
        return urlparse(url).scheme in ("http", "https")
    except Exception:
        return False


def _ddg_text_results(query: str, max_results: int) -> list[dict[str, str]]:
    if DDGS is None or max_results <= 0:
        return []
    rows: list[dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                href = (r.get("href") or r.get("url") or "").strip()
                if not href or not _http_url(href):
                    continue
                rows.append({
                    "url": href,
                    "title": (r.get("title") or "").strip(),
                    "snippet": (r.get("body") or "").strip(),
                })
    except Exception as e:  # noqa: BLE001
        logger.warning("DDG échouée (%s): %s", query[:60], e)
    return rows


def _fetch_page_text(url: str, timeout: float) -> str:
    headers = {"User-Agent": "VisitCorsicaChatbot/1.0"}
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as c:
            r = c.get(url)
            r.raise_for_status()
        text = trafilatura.extract(r.text, url=url, include_comments=False, include_tables=False)
        return (text or "").strip()
    except Exception as e:  # noqa: BLE001
        logger.debug("Fetch ignoré %s : %s", url, e)
        return ""


def gather_web_context(
    message: str,
    *,
    max_ddg: int,
    max_fetch: int,
    fetch_timeout: float,
) -> list[tuple[str, str]]:
    if not message.strip():
        return []

    site_hits = _ddg_text_results(f"site:visit-corsica.com {message}", max_ddg)
    seen: set[str] = {h["url"] for h in site_hits}
    merged = list(site_hits)

    if len(merged) < 2:
        for h in _ddg_text_results(f"Corse tourisme {message}", max_ddg):
            if h["url"] not in seen:
                merged.append(h)
                seen.add(h["url"])

    merged.sort(key=lambda x: (not _is_visit_corsica(x["url"]), x["url"]))
    to_fetch = merged[:max_fetch]

    # Fetch pages en parallèle
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_fetch) as pool:
        futures = {pool.submit(_fetch_page_text, h["url"], fetch_timeout): h for h in to_fetch}
        for fut in as_completed(futures):
            hit = futures[fut]
            body = ""
            try:
                body = fut.result()
            except Exception:  # noqa: BLE001
                pass
            if len(body) < 120:
                body = hit.get("snippet") or ""
            if len(body.strip()) >= 40:
                results[hit["url"]] = body.strip()[:4000]

    out: list[tuple[str, str]] = []
    for h in to_fetch:
        if h["url"] in results:
            out.append((h["url"], results[h["url"]]))
    return out
