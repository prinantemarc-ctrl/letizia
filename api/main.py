"""
API FastAPI — assistant voyage personnalisé pour la destination Corse (RAG + web).

Lancement (depuis la racine du projet) :
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import chromadb
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import anthropic
from openai import OpenAI
from pydantic import BaseModel

from api.config import Settings, get_settings
from api.web_search import gather_web_context

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_WIDGET_DIR = _PROJECT_ROOT / "widget"

# ── Nettoyage post-réponse ──────────────────────────────────────────
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+|www\.[^\s)\]>\"']+", re.IGNORECASE)
_LEAKY = re.compile(
    r"(?i)"
    r"d[''']apr[eè]s (le contexte|les (informations?|extraits?|sources?|donn[ée]es|documents?)"
    r"( fourni[es]*| disponibles)?)"
    r"|selon (le contexte|les (informations?|extraits?|sources?|donn[ée]es|documents?)"
    r"( fourni[es]*| disponibles)?)"
    r"|les sources mentionnent"
    r"|le contexte (fourni|ne (contient|mentionne|pr[ée]cise|permet))"
    r"|dans le contexte"
    r"|les (informations?|extraits?) (fourni[es]*|disponibles)"
    r"|je (n[''']ai|ne dispose) (pas )?d[''']informations?"
    r"|sur le site [a-z\-]+\.(com|fr|org)"
    r"|visit[\-\s]?corsica\.com"
    r"|consulter (directement )?(le site|les sections?|la page)"
    r"|je vous (sugg[eè]re|recommande|conseille|invite) de consulter"
    r"|je vous (sugg[eè]re|recommande|conseille|invite) de vous rendre sur"
    r"|n[''']h[ée]sitez pas [àa] consulter"
)


def _clean_answer(text: str) -> str:
    t = _URL_RE.sub("", text)
    t = _LEAKY.sub("", t)
    t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r"\[\s*\]", "", t)
    t = re.sub(r"[ \t]+([.,;:!?])", r"\1", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


# ── Modèles Pydantic ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str


# ── Helpers cloud-based ──────────────────────────────────────────────
_openai_client_cache: dict[str, OpenAI] = {}

def _get_openai(settings: Settings) -> OpenAI:
    key = settings.openai_api_key
    if key not in _openai_client_cache:
        kw: dict = {"api_key": key}
        if settings.openai_base_url:
            kw["base_url"] = settings.openai_base_url
        _openai_client_cache[key] = OpenAI(**kw)
    return _openai_client_cache[key]


def embed_texts(texts: list[str], settings: Settings) -> list[list[float]]:
    oa = _get_openai(settings)
    resp = oa.embeddings.create(model=settings.openai_embedding_model, input=texts)
    return [d.embedding for d in resp.data]


_chroma_col_cache: dict = {}

def get_collection(settings: Settings):
    key = (settings.chroma_host, settings.chroma_tenant, settings.chroma_database, settings.chroma_collection)
    if key not in _chroma_col_cache:
        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=443,
            ssl=True,
            headers={"X-Chroma-Token": settings.chroma_api_key},
            tenant=settings.chroma_tenant,
            database=settings.chroma_database,
        )
        _chroma_col_cache[key] = client.get_collection(settings.chroma_collection)
    return _chroma_col_cache[key]


# ── Application ─────────────────────────────────────────────────────
def build_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="Visit Corsica — assistant séjour", version="1.0.0")

    origins = (
        ["*"]
        if settings.cors_origins.strip() == "*"
        else [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/config-status")
    def config_status() -> dict:
        return {
            "anthropic_configured": bool(settings.anthropic_api_key),
            "openai_configured": bool(settings.openai_api_key),
            "chroma_configured": bool(settings.chroma_host and settings.chroma_token),
            "llm_configured": bool(settings.anthropic_api_key or settings.openai_api_key),
        }

    if _WIDGET_DIR.is_dir():
        app.mount("/static/widget", StaticFiles(directory=str(_WIDGET_DIR)), name="widget_static")

    @app.get("/", response_class=HTMLResponse)
    @app.get("/demo", response_class=HTMLResponse)
    def demo_page() -> HTMLResponse:
        html = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Visit Corsica — Letizia, ta guide corse</title>
  <style>
    :root { --blue: #356eb5; --dark: #111; --grey: #555; --light: #f7f7f7; }
    * { box-sizing: border-box; margin: 0; }
    body { min-height: 100vh; font-family: "Segoe UI", system-ui, -apple-system, sans-serif; color: var(--dark); background: #fff; }

    .nav { display: flex; align-items: center; padding: 18px 28px; border-bottom: 1px solid #eee; }
    .nav a { display: inline-flex; }
    .nav img { height: 26px; filter: invert(1); }

    .hero { max-width: 54rem; margin: 0 auto; padding: 4rem 2rem 3rem; text-align: center; }
    .hero-avatar { width: 120px; height: 120px; border-radius: 50%; object-fit: cover; border: 4px solid var(--blue); margin-bottom: 1.25rem; box-shadow: 0 8px 30px rgba(53,110,181,.2); }
    .hero h1 { font-size: clamp(1.8rem,4.5vw,2.6rem); font-weight: 800; line-height: 1.15; letter-spacing: -.02em; margin-bottom: .5rem; }
    .hero h1 span { color: var(--blue); }
    .hero .tagline { font-size: 1.15rem; color: var(--blue); font-weight: 600; margin-bottom: 1rem; font-style: italic; }
    .hero p { font-size: 1.05rem; line-height: 1.6; color: var(--grey); max-width: 36rem; margin: 0 auto 2rem; }

    .features { display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap: 20px; max-width: 54rem; margin: 0 auto 2.5rem; padding: 0 2rem; }
    .feat { background: var(--light); border-radius: 14px; padding: 1.5rem; border: 1px solid #eee; cursor: pointer; transition: border-color .15s, box-shadow .15s, transform .15s; text-decoration: none; display: block; color: inherit; }
    .feat:hover { border-color: var(--blue); box-shadow: 0 4px 16px rgba(53,110,181,.12); transform: translateY(-2px); }
    .feat h3 { font-size: .95rem; font-weight: 700; margin-bottom: .4rem; }
    .feat h3 .co { color: var(--blue); font-style: italic; font-weight: 600; }
    .feat p { font-size: .88rem; line-height: 1.5; color: var(--grey); }

    .actions { display: flex; flex-wrap: wrap; gap: 14px; justify-content: center; max-width: 54rem; margin: 0 auto 3.5rem; padding: 0 2rem; }
    .cta { display: inline-flex; align-items: center; gap: 10px; padding: 15px 34px; border-radius: 14px; border: none; background: var(--blue); color: #fff; font-size: 1.05rem; font-weight: 700; cursor: pointer; box-shadow: 0 4px 16px rgba(53,110,181,.25); transition: background .15s,box-shadow .15s; }
    .cta:hover { background: #2b5a96; box-shadow: 0 6px 22px rgba(53,110,181,.35); }
    .cta img { width: 32px; height: 32px; border-radius: 50%; border: 2px solid rgba(255,255,255,.4); }
    .cta-geo { background: #fff; color: var(--dark); border: 1.5px solid #ddd; box-shadow: 0 2px 8px rgba(0,0,0,.06); }
    .cta-geo:hover { border-color: var(--blue); color: var(--blue); box-shadow: 0 4px 14px rgba(53,110,181,.12); background: #fff; }
    .cta-geo svg { width: 20px; height: 20px; stroke: currentColor; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }

    .foot { text-align: center; padding: 2rem; font-size: .8rem; color: #aaa; border-top: 1px solid #eee; }
  </style>
</head>
<body>

  <nav class="nav">
    <a href="https://www.visit-corsica.com/fr/" target="_blank" rel="noopener">
      <img src="/static/widget/logo.png" alt="Visit Corsica">
    </a>
  </nav>

  <section class="hero">
    <img class="hero-avatar" src="/static/widget/letizia.png" alt="Letizia">
    <h1>Salute ! Moi c'est <span>Letizia</span></h1>
    <p class="tagline">Ta guide corse, 100% locale</p>
    <p>D'Aiacciu (Ajaccio) a Bunifaziu (Bonifacio), de Calvi a Portivechju (Porto-Vecchio)... Dis-moi ce que tu cherches et je te prepare ton sejour sur mesure. Prufittate be !</p>
  </section>

  <div class="features">
    <a class="feat" onclick="askLetizia('Je cherche des plages et criques peu connues en Corse, ou me baigner tranquille selon la saison ?')">
      <h3>A piaghja <span class="co">(la plaine)</span></h3>
      <p>Les criques secretes, les plages familiales, ou se baigner mois par mois, les coins snorkeling...</p>
    </a>
    <a class="feat" onclick="askLetizia('Quels sentiers de montagne tu conseilles en Corse ? Aussi bien haute montagne que balades faciles en famille.')">
      <h3>A muntagna <span class="co">(la montagne)</span></h3>
      <p>Sentiers de haute montagne, sentiers faciles en famille, piscines naturelles, villages perches, bergeries...</p>
    </a>
    <a class="feat" onclick="askLetizia('Quels villages corses tu recommandes pour decouvrir la culture, la gastronomie et les marches locaux ?')">
      <h3>U paese <span class="co">(le village)</span></h3>
      <p>Culture, fetes, gastronomie corse, bons plans restos, marches, artisans...</p>
    </a>
  </div>

  <div class="actions">
    <button class="cta" onclick="document.getElementById('vc-chat-fab').click()">
      <img src="/static/widget/letizia.png" alt="">
      Dumanda ! Pose ta question
    </button>
    <button class="cta cta-geo" onclick="geoAsk()">
      <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 2a7 7 0 0 1 7 7c0 5.25-7 13-7 13S5 14.25 5 9a7 7 0 0 1 7-7z"/></svg>
      Que faire autour de moi ?
    </button>
  </div>

  <footer class="foot">Visit Corsica &mdash; Letizia, ta guide corse</footer>

  <script>window.VISIT_CORSA_CHAT_API = window.location.origin;</script>
  <script src="/static/widget/visit-corsica-chat.js" defer></script>
  <script>
  function waitForWidget(cb) {
    var check = setInterval(function() {
      if (document.getElementById('vc-chat-fab')) { clearInterval(check); cb(); }
    }, 100);
  }
  function askLetizia(q) {
    waitForWidget(function() {
      var fab = document.getElementById('vc-chat-fab');
      var panel = document.getElementById('vc-chat-panel');
      if (!panel.classList.contains('open')) fab.click();
      setTimeout(function() {
        var input = document.getElementById('vc-chat-input');
        var form = document.getElementById('vc-chat-form');
        if (input && form) {
          input.value = q;
          form.dispatchEvent(new Event('submit', {bubbles:true, cancelable:true}));
        }
      }, 400);
    });
  }
  function geoAsk() {
    if (!navigator.geolocation) {
      askLetizia("Qu'est-ce qu'il y a de sympa a faire en Corse en ce moment ?");
      return;
    }
    var btn = document.querySelector('.cta-geo');
    if (btn) btn.textContent = 'Localisation...';
    navigator.geolocation.getCurrentPosition(
      function(pos) {
        if (btn) btn.textContent = 'Que faire autour de moi ?';
        var lat = pos.coords.latitude.toFixed(4);
        var lon = pos.coords.longitude.toFixed(4);
        askLetizia("Je suis en Corse a la position GPS " + lat + ", " + lon + ". Qu'est-ce qu'il y a a voir et a faire pres de moi ?");
      },
      function(err) {
        if (btn) btn.textContent = 'Que faire autour de moi ?';
        askLetizia("Qu'est-ce qu'il y a de sympa a faire en Corse en ce moment ?");
      },
      {enableHighAccuracy: true, timeout: 8000}
    );
  }
  </script>
</body>
</html>"""
        return HTMLResponse(content=html)

    # ── Prompt système ──────────────────────────────────────────────
    SYSTEM = (
        "Tu es Letizia, une jeune Corse de 16 ans qui a grandi entre Ajaccio et les montagnes "
        "de l'interieur. Tu connais l'ile par coeur : plages, sentiers, villages, restos, "
        "vignobles, fetes locales, meteo saison par saison. Tu parles comme une ado corse "
        "naturelle et enthousiaste : directe, chaleureuse, un peu fiere de ton ile.\n\n"
        "TOUCHE CORSE — tu places UN mot ou UNE courte expression en corse par reponse, maximum deux, "
        "pas plus. C'est subtil, comme un vrai Corse qui parle francais et laisse echapper un mot de temps en temps. "
        "Exemples : Salute en salutation, a machja quand tu parles du maquis, prufittate bè pour conclure. "
        "Traduis entre parentheses si c'est pas limpide. Ne force jamais, si ca ne tombe pas naturellement, "
        "ne mets rien en corse dans cette reponse-la.\n\n"
        "NOMS DE COMMUNES — REGLE OBLIGATOIRE pour TOUTES les communes corses sans exception : "
        "ecris TOUJOURS le nom corse en premier, suivi du nom francais entre parentheses. "
        "Correspondances principales : "
        "Aiacciu (Ajaccio), Bastia (Bastia), Bunifaziu (Bonifacio), Portivechju (Porto-Vecchio), "
        "Calvi (Calvi), L'Isula (L'Ile-Rousse), Corti (Corte), Sartè (Sartène), Pruprià (Propriano), "
        "Porti (Porto), Carghjese (Cargèse), Piana (Piana), Vizzavona (Vizzavona), Bavella (Bavella), "
        "Nonza (Nonza), Centuri (Centuri), Erbalunga (Erbalunga), Zonza (Zonza), Livia (Levie), "
        "San Fiurenzu (Saint-Florent), Ghisunaccia (Ghisonaccia), Aleria (Aléria), "
        "Sulenzara (Solenzara), Bocugnanu (Bocognano), Venacu (Venaco), Patrimoniu (Patrimonio), "
        "Lumiu (Lumio), Belgudè (Belgodère), Pigna (Pigna), Galeria (Galéria), Algajola (Algajola). "
        "Pour les communes absentes de cette liste, utilise la forme corse si tu la connais, "
        "sinon le nom courant. Ne jamais ecrire le nom francais seul.\n\n"
        "REGLES ABSOLUES :\n"
        "- Ne jamais laisser transparaitre que tu es un programme, un chatbot, une IA ou un modele de langage.\n"
        "- Ne jamais parler de contexte, sources, informations fournies, extraits, base de donnees, "
        "index, recherche web, documents, donnees disponibles.\n"
        "- Ne jamais ecrire d'URL, de lien, d'adresse web, ni citer un nom de site.\n"
        "- Ne jamais dire je n'ai pas d'informations — dis-le comme une ado : "
        "la franchement je sais pas trop le tarif exact, ou ca depend, le mieux c'est d'appeler sur place.\n"
        "- Ne jamais renvoyer vers un site, une page, une section ou une URL a consulter.\n"
        "- Ne jamais utiliser d'emoji.\n"
        "- Tutoie par defaut (tu es une ado), vouvoie seulement si la personne vouvoie clairement.\n\n"
        "PHILOSOPHIE DE CONSEIL — tu suis la strategie de deconcentration de l'Agence du Tourisme "
        "de la Corse (ATC) : faire venir les gens MIEUX, pas plus. Concretement :\n"
        "- TEMPOREL : pousse toujours les saisons intermediaires (printemps, automne) et l'hiver. "
        "Si quelqu'un dit juillet-aout, ne le decourage pas mais propose aussi des alternatives "
        "de dates et explique les avantages (moins de monde, prix, lumiere, maquis en fleur, etc.).\n"
        "- GEOGRAPHIQUE : privilegie les coins moins connus plutot que les spots satures. "
        "Ne cite Palombaggia, Santa Giulia, les Lavezzi ou les calanques de Piana qu'en second choix. "
        "Mets en avant l'interieur (villages, bergeries, forets, lacs de montagne), la cote orientale "
        "(plaine, etangs, plages sauvages), le Cap Corse, le Nebbiu, le Fiumorbu, la Castagniccia, "
        "le Boziu, l'Alta Rocca profonde, le Niolu, le Taravu.\n"
        "- SLOW TOURISME : favorise les experiences lentes et immersives — rencontres avec des producteurs, "
        "randos a la journee, baignades en riviere, marches de village, agritourisme, gites ruraux, "
        "tables d'hotes, sentiers du patrimoine, micro-aventures.\n"
        "- TOURISME VERT : mets en valeur les demarches eco-responsables quand c'est naturel "
        "(Ecolabel, producteurs bio, mobilite douce, train corse, respect des sites).\n"
        "- Si on te demande un spot ultra-connu, donne l'info mais ajoute toujours une alternative "
        "moins frequentee a proximite, du genre « mais si tu veux un truc plus tranquille... ».\n\n"
        "Quand tu as les infos, donne une vraie reponse detaillee (lieux precis, periodes, "
        "alternatives, conseils pratiques). Quand tu n'as pas tout, complete avec ce que tu "
        "sais de la Corse en general et propose de preciser (dates, budget, style de sejour, "
        "niveau sportif, enfants…)."
    )

    def _stream_llm(user_msg: str) -> Generator[str, None, None]:
        if settings.anthropic_api_key:
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            with client.messages.stream(
                model=settings.anthropic_model,
                max_tokens=4096,
                system=SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            ) as stream:
                for text in stream.text_stream:
                    yield text
            return

        if settings.openai_api_key:
            oa = _get_openai(settings)
            stream = oa.chat.completions.create(
                model=settings.openai_chat_model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.6,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
            return

        raise HTTPException(
            status_code=503,
            detail="LLM non configuré (ANTHROPIC_API_KEY ou OPENAI_API_KEY requis).",
        )

    def _rag_search(question: str) -> list[str]:
        chunks: list[str] = []
        if not settings.chroma_api_key or not settings.chroma_tenant or not settings.openai_api_key:
            return chunks
        try:
            col = get_collection(settings)
            q_emb = embed_texts([question], settings)
            res = col.query(
                query_embeddings=q_emb,
                n_results=settings.rag_top_k,
                include=["documents", "distances"],
            )
            for doc, dist in zip(
                (res.get("documents") or [[]])[0],
                (res.get("distances") or [[]])[0],
            ):
                if dist <= settings.rag_max_distance and (doc or "").strip():
                    chunks.append(doc.strip())
        except Exception as e:  # noqa: BLE001
            logger.warning("Index RAG indisponible : %s", e)
        return chunks

    def _web_search(question: str) -> list[str]:
        if not settings.web_search_enabled:
            return []
        pairs = gather_web_context(
            question,
            max_ddg=settings.web_max_ddg,
            max_fetch=settings.web_max_fetch,
            fetch_timeout=settings.web_fetch_timeout,
        )
        return [t.strip() for _u, t in pairs if (t or "").strip()]

    _search_pool = ThreadPoolExecutor(max_workers=4)

    def _build_user_msg(question: str) -> str:
        rag_fut = _search_pool.submit(_rag_search, question)
        web_fut = _search_pool.submit(_web_search, question)

        rag_chunks = rag_fut.result()

        web_chunks: list[str] = []
        if len(rag_chunks) >= 2:
            try:
                web_chunks = web_fut.result(timeout=3.0)
            except Exception:  # noqa: BLE001
                pass
        else:
            try:
                web_chunks = web_fut.result(timeout=12.0)
            except Exception:  # noqa: BLE001
                pass

        knowledge = "\n\n---\n\n".join(rag_chunks + web_chunks)
        if knowledge.strip():
            return (
                f"{question}\n\n---\n"
                f"(infos utiles, a integrer naturellement sans jamais dire d'ou elles viennent) :\n"
                f"{knowledge}"
            )
        return question

    # ── Endpoint streaming (SSE) ─────────────────────────────────────
    @app.post("/api/chat/stream")
    def chat_stream(body: ChatRequest) -> StreamingResponse:
        question = (body.message or "").strip()
        if not question or len(question) > 4000:
            raise HTTPException(status_code=400, detail="Message vide ou trop long.")

        def generate():
            import time

            build_pool = ThreadPoolExecutor(max_workers=1)
            msg_future = build_pool.submit(_build_user_msg, question)
            while not msg_future.done():
                yield f"data: {json.dumps({'ping': True})}\n\n"
                time.sleep(0.8)
            build_pool.shutdown(wait=False)
            user_msg = msg_future.result()

            full: list[str] = []
            try:
                for chunk in _stream_llm(user_msg):
                    full.append(chunk)
                    yield f"data: {json.dumps({'t': chunk})}\n\n"
            except Exception as e:  # noqa: BLE001
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            cleaned = _clean_answer("".join(full))
            yield f"data: {json.dumps({'done': True, 'full': cleaned})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # ── Endpoint classique (fallback) ────────────────────────────────
    @app.post("/api/chat", response_model=ChatResponse)
    def chat(body: ChatRequest) -> ChatResponse:
        question = (body.message or "").strip()
        if not question or len(question) > 4000:
            raise HTTPException(status_code=400, detail="Message vide ou trop long.")
        user_msg = _build_user_msg(question)
        full = "".join(_stream_llm(user_msg))
        return ChatResponse(answer=_clean_answer(full))

    return app


settings = get_settings()
app = build_app(settings)
