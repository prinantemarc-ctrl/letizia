"""Découpage de texte en chunks pour l'indexation."""

from __future__ import annotations


def chunk_text(
    text: str,
    max_chars: int = 900,
    overlap: int = 120,
    min_chars: int = 80,
) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""

    def flush_buf() -> None:
        nonlocal buf
        if len(buf) >= min_chars:
            chunks.append(buf.strip())
        buf = ""

    for p in paras:
        if len(p) > max_chars:
            flush_buf()
            start = 0
            while start < len(p):
                end = min(start + max_chars, len(p))
                piece = p[start:end].strip()
                if piece:
                    chunks.append(piece)
                start = end - overlap if end < len(p) else end
            continue

        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= max_chars:
            buf = f"{buf}\n\n{p}"
        else:
            flush_buf()
            buf = p

    flush_buf()
    return chunks
