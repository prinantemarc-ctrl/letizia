"""Classification IA des questions visiteurs via Claude (appel rapide, non-streaming)."""

from __future__ import annotations

import json
import logging

import anthropic

logger = logging.getLogger(__name__)

CLASSIFY_PROMPT = """Analyse cette conversation (question visiteur + debut de reponse) et retourne un JSON avec exactement ces champs :

- "category": un parmi [hebergement, plages, randonnee, gastronomie, transport, culture, budget, meteo, itineraire, activites, general]
- "region": un parmi [Balagne, Extreme-Sud, Alta-Rocca, Cap-Corse, Castagniccia, Ajaccio, Bastia, Cortenais, Cote-orientale, Ouest, Sartenais, non-specifique]
- "season": un parmi [printemps, ete, automne, hiver, non-specifique]
- "travel_style": un parmi [famille, couple, solo, groupe, sportif, slow, non-specifique]

Reponds UNIQUEMENT avec le JSON, rien d'autre. Pas de markdown, pas de commentaire.

Question: {question}
Reponse (debut): {answer_preview}"""


def classify_question(
    question: str,
    answer_preview: str,
    *,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Retourne {category, region, season, travel_style} ou des valeurs par defaut si erreur."""
    defaults = {
        "category": "general",
        "region": "non-specifique",
        "season": "non-specifique",
        "travel_style": "non-specifique",
    }
    if not api_key:
        return defaults

    try:
        client = anthropic.Anthropic(api_key=api_key, max_retries=1, timeout=15.0)
        msg = client.messages.create(
            model=model,
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": CLASSIFY_PROMPT.format(
                    question=question[:500],
                    answer_preview=answer_preview[:300],
                ),
            }],
        )
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        for k in defaults:
            if k not in data or not isinstance(data[k], str):
                data[k] = defaults[k]
        return data
    except Exception as e:  # noqa: BLE001
        logger.warning("Classification echouee: %s", e)
        return defaults
