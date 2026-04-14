"""Lecture minimale de robots.txt pour filtrer les URL interdites."""

from __future__ import annotations

import re
from urllib.parse import urlparse


def parse_robots_disallow(robots_text: str) -> list[str]:
    """Retourne les préfixes de chemins Disallow (user-agent: * uniquement)."""
    lines = robots_text.splitlines()
    in_star = False
    disallows: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m_ua = re.match(r"(?i)User-agent:\s*(.+)", line)
        if m_ua:
            in_star = m_ua.group(1).strip() == "*"
            continue
        m_dis = re.match(r"(?i)Disallow:\s*(.*)", line)
        if m_dis and in_star:
            path = m_dis.group(1).strip()
            if path:
                disallows.append(path)
    return disallows


def path_is_disallowed(url: str, disallow_prefixes: list[str]) -> bool:
    parsed = urlparse(url)
    path = parsed.path or "/"
    for prefix in disallow_prefixes:
        if path.startswith(prefix) or (prefix.startswith("/") and path.startswith(prefix)):
            return True
    return False
