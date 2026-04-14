"""
Crawl du contenu français visit-corsica.com via sitemap_fr.xml.

Respecte Crawl-delay (défaut 10 s) et les chemins Disallow du robots.txt.
Usage:
  python -m scraper.scrape --out data/raw/pages_fr.jsonl
  python -m scraper.scrape --max-pages 20 # test rapide
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx
import trafilatura
from lxml import etree

from scraper.robots import parse_robots_disallow, path_is_disallowed

BASE = "https://www.visit-corsica.com"
SITEMAP_FR = f"{BASE}/sitemap_fr.xml"
ROBOTS_URL = f"{BASE}/robots.txt"
USER_AGENT = "VisitCorsicaKnowledgeBot/1.0 (+https://example.com; crawl pour chatbot RAG interne)"


def fetch_text(client: httpx.Client, url: str, timeout: float = 60.0) -> str | None:
    r = client.get(url, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    if "text/html" not in (r.headers.get("content-type") or "").lower():
        return None
    extracted = trafilatura.extract(
        r.text,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    return (extracted or "").strip() or None


def parse_sitemap_fr(xml_bytes: bytes) -> list[str]:
    root = etree.fromstring(xml_bytes)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = root.xpath("//sm:url/sm:loc/text()", namespaces=ns)
    return [u.strip() for u in locs if u.strip()]


def read_crawl_delay(robots_text: str) -> float | None:
    for line in robots_text.splitlines():
        m = re.match(r"(?i)Crawl-delay:\s*(\d+(?:\.\d+)?)", line.strip())
        if m:
            return float(m.group(1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape visit-corsica.com (FR)")
    parser.add_argument("--out", type=Path, default=Path("data/raw/pages_fr.jsonl"))
    parser.add_argument("--max-pages", type=int, default=0, help="0 = pas de limite")
    parser.add_argument("--delay", type=float, default=-1, help="secondes entre requêtes; -1 = depuis robots.txt")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(headers={"User-Agent": USER_AGENT}, verify=True) as client:
        robots_resp = client.get(ROBOTS_URL, timeout=30.0)
        robots_resp.raise_for_status()
        robots_body = robots_resp.text
        disallow = parse_robots_disallow(robots_body)
        delay = args.delay
        if delay < 0:
            delay = read_crawl_delay(robots_body) or 1.0

        sm = client.get(SITEMAP_FR, timeout=60.0)
        sm.raise_for_status()
        urls = parse_sitemap_fr(sm.content)
        urls = [u for u in urls if "/fr/" in u or u.rstrip("/") == f"{BASE}/fr"]
        urls = [u for u in urls if not path_is_disallowed(u, disallow)]

        if args.max_pages:
            urls = urls[: args.max_pages]

        print(f"URLs à traiter: {len(urls)} (délai {delay}s)", file=sys.stderr)

        count_ok = 0
        with args.out.open("w", encoding="utf-8") as f:
            for i, url in enumerate(urls):
                try:
                    text = fetch_text(client, url)
                    rec = {
                        "url": url,
                        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "text": text,
                        "error": None if text else "empty_or_non_html",
                    }
                    if text:
                        count_ok += 1
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                except Exception as e:  # noqa: BLE001
                    rec = {
                        "url": url,
                        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "text": None,
                        "error": str(e),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"ERR {url}: {e}", file=sys.stderr)

                if i + 1 < len(urls):
                    time.sleep(delay)

        print(f"Terminé. Pages avec texte: {count_ok}/{len(urls)} -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
