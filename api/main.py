"""
API FastAPI — assistant voyage personnalisé pour la destination Corse (RAG + web).

Lancement (depuis la racine du projet) :
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timedelta, timezone
from pathlib import Path

import chromadb
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import anthropic
from openai import OpenAI
from pydantic import BaseModel

from api.classify import classify_question
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
_chroma_client_cache: dict = {}

def _get_chroma_client(settings: Settings):
    key = (settings.chroma_host, settings.chroma_tenant, settings.chroma_database)
    if key not in _chroma_client_cache:
        _chroma_client_cache[key] = chromadb.HttpClient(
            host=settings.chroma_host,
            port=443,
            ssl=True,
            headers={"X-Chroma-Token": settings.chroma_api_key},
            tenant=settings.chroma_tenant,
            database=settings.chroma_database,
        )
    return _chroma_client_cache[key]


def get_collection(settings: Settings):
    key = ("rag", settings.chroma_collection)
    if key not in _chroma_col_cache:
        client = _get_chroma_client(settings)
        _chroma_col_cache[key] = client.get_collection(settings.chroma_collection)
    return _chroma_col_cache[key]


def get_log_collection(settings: Settings):
    key = ("log", settings.log_collection)
    if key not in _chroma_col_cache:
        client = _get_chroma_client(settings)
        _chroma_col_cache[key] = client.get_or_create_collection(
            name=settings.log_collection,
            metadata={"hnsw:space": "cosine"},
        )
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
            "chroma_configured": bool(settings.chroma_api_key and settings.chroma_tenant),
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
  <link rel="icon" type="image/png" href="/static/widget/favicon.png">
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

    @media(max-width:480px) {
      .hero { padding: 2.5rem 1.25rem 2rem; }
      .hero-avatar { width: 90px; height: 90px; }
      .hero .tagline { font-size: 1rem; }
      .hero p { font-size: .95rem; }
      .features { padding: 0 1rem; gap: 12px; grid-template-columns: 1fr; }
      .actions { padding: 0 1rem; flex-direction: column; align-items: stretch; }
      .cta { justify-content: center; padding: 14px 20px; font-size: .95rem; }
      .nav { padding: 14px 16px; }
    }
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
    <p><em>D'Aiacciu a Luri, da Sant'Antuninu a Quenza, dimmi cio ch'e tu cerchi, e t'aiutaraghju !</em><br>D'Ajaccio a Luri, de Sant'Antonino a Quenza, dis-moi ce que tu cherches et je t'aiderai !</p>
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
            client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key,
                max_retries=1,
                timeout=55.0,
            )
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

    # ── Logging async des conversations ────────────────────────────────
    def _log_conversation(question: str, answer: str, user_agent: str = "") -> None:
        try:
            device = "mobile" if any(k in user_agent.lower() for k in ("mobile", "android", "iphone")) else "desktop"
            tags = classify_question(
                question, answer[:300],
                api_key=settings.anthropic_api_key,
                model=settings.anthropic_model,
            )
            col = get_log_collection(settings)
            emb = embed_texts([question], settings)
            col.add(
                ids=[str(uuid.uuid4())],
                documents=[question],
                embeddings=emb,
                metadatas=[{
                    "question": question[:500],
                    "answer_preview": answer[:200],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "category": tags.get("category", "general"),
                    "region": tags.get("region", "non-specifique"),
                    "season": tags.get("season", "non-specifique"),
                    "travel_style": tags.get("travel_style", "non-specifique"),
                    "device": device,
                }],
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Log conversation echoue: %s", e)

    # ── Endpoint streaming (SSE) ─────────────────────────────────────
    @app.post("/api/chat/stream")
    def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
        question = (body.message or "").strip()
        if not question or len(question) > 4000:
            raise HTTPException(status_code=400, detail="Message vide ou trop long.")
        ua = request.headers.get("user-agent", "")

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

            if cleaned:
                _search_pool.submit(_log_conversation, question, cleaned, ua)

        return StreamingResponse(generate(), media_type="text/event-stream")

    # ── Endpoint classique (fallback) ────────────────────────────────
    @app.post("/api/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        question = (body.message or "").strip()
        if not question or len(question) > 4000:
            raise HTTPException(status_code=400, detail="Message vide ou trop long.")
        user_msg = _build_user_msg(question)
        full = "".join(_stream_llm(user_msg))
        cleaned = _clean_answer(full)
        if cleaned:
            ua = request.headers.get("user-agent", "")
            _search_pool.submit(_log_conversation, question, cleaned, ua)
        return ChatResponse(answer=cleaned)

    # ── Admin: vérification clé ──────────────────────────────────────
    def _check_admin(key: str) -> None:
        if not settings.admin_key or key != settings.admin_key:
            raise HTTPException(status_code=403, detail="Acces refuse.")

    # ── Admin: stats JSON ────────────────────────────────────────────
    @app.get("/api/admin/stats")
    def admin_stats(key: str = Query("")) -> JSONResponse:
        _check_admin(key)
        try:
            col = get_log_collection(settings)
            all_data = col.get(include=["metadatas"])
            metas = all_data.get("metadatas") or []
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, status_code=500)

        now = datetime.now(timezone.utc)
        today_str = now.strftime("%Y-%m-%d")
        week_ago = now - timedelta(days=7)

        cat_counter: Counter = Counter()
        region_counter: Counter = Counter()
        season_counter: Counter = Counter()
        style_counter: Counter = Counter()
        device_counter: Counter = Counter()
        today_count = 0
        week_count = 0
        recent: list[dict] = []

        for m in metas:
            cat_counter[m.get("category", "general")] += 1
            region_counter[m.get("region", "non-specifique")] += 1
            season_counter[m.get("season", "non-specifique")] += 1
            style_counter[m.get("travel_style", "non-specifique")] += 1
            device_counter[m.get("device", "desktop")] += 1
            ts = m.get("timestamp", "")
            if ts.startswith(today_str):
                today_count += 1
            try:
                if datetime.fromisoformat(ts) >= week_ago:
                    week_count += 1
            except Exception:  # noqa: BLE001
                pass
            recent.append({
                "question": m.get("question", "")[:200],
                "category": m.get("category", ""),
                "region": m.get("region", ""),
                "season": m.get("season", ""),
                "style": m.get("travel_style", ""),
                "device": m.get("device", ""),
                "ts": ts,
            })

        recent.sort(key=lambda x: x["ts"], reverse=True)

        return JSONResponse({
            "total": len(metas),
            "today": today_count,
            "this_week": week_count,
            "by_category": dict(cat_counter.most_common()),
            "by_region": dict(region_counter.most_common()),
            "by_season": dict(season_counter.most_common()),
            "by_style": dict(style_counter.most_common()),
            "by_device": dict(device_counter.most_common()),
            "recent": recent[:50],
        })

    # ── Admin: dashboard HTML ────────────────────────────────────────
    @app.get("/admin", response_class=HTMLResponse)
    def admin_dashboard(key: str = Query("")) -> HTMLResponse:
        _check_admin(key)
        html = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" type="image/png" href="/static/widget/favicon.png">
<title>Letizia — Analytics</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
:root{--blue:#356eb5;--dark:#0f1117;--card:#1a1d27;--border:#2a2d3a;--text:#e1e4ed;--muted:#8b8fa3;--green:#22c55e;--amber:#f59e0b;--pink:#ec4899;--cyan:#06b6d4;--purple:#a855f7;--red:#ef4444}
*{box-sizing:border-box;margin:0}
body{font-family:"Inter","Segoe UI",system-ui,sans-serif;background:var(--dark);color:var(--text);min-height:100vh}

.top{background:#13151d;border-bottom:1px solid var(--border);padding:16px 24px;display:flex;align-items:center;justify-content:space-between}
.top-left{display:flex;align-items:center;gap:12px}
.top-left img{height:28px;border-radius:50%;border:2px solid var(--blue)}
.top h1{font-size:1rem;font-weight:700;color:#fff}
.top h1 span{color:var(--blue);font-weight:400}
.live{display:flex;align-items:center;gap:6px;font-size:.75rem;color:var(--green)}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

.wrap{max-width:80rem;margin:0 auto;padding:20px}

.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-bottom:24px}
.kpi{background:var(--card);border-radius:14px;padding:22px 20px;border:1px solid var(--border);position:relative;overflow:hidden}
.kpi::before{content:"";position:absolute;top:0;left:0;right:0;height:3px}
.kpi:nth-child(1)::before{background:var(--blue)}
.kpi:nth-child(2)::before{background:var(--green)}
.kpi:nth-child(3)::before{background:var(--amber)}
.kpi:nth-child(4)::before{background:var(--pink)}
.kpi:nth-child(5)::before{background:var(--cyan)}
.kpi .val{font-size:2rem;font-weight:800;line-height:1}
.kpi:nth-child(1) .val{color:var(--blue)}
.kpi:nth-child(2) .val{color:var(--green)}
.kpi:nth-child(3) .val{color:var(--amber)}
.kpi:nth-child(4) .val{color:var(--pink)}
.kpi:nth-child(5) .val{color:var(--cyan)}
.kpi .lbl{font-size:.78rem;color:var(--muted);margin-top:6px}

.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px}
@media(max-width:900px){.grid2,.grid3{grid-template-columns:1fr}}

.card{background:var(--card);border-radius:14px;padding:20px;border:1px solid var(--border)}
.card h3{font-size:.85rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:16px}
.card canvas{width:100%!important;max-height:260px}

.tbl{background:var(--card);border-radius:14px;padding:20px;border:1px solid var(--border);margin-bottom:16px}
.tbl h3{font-size:.85rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:14px}
.tbl table{width:100%;border-collapse:collapse;font-size:.82rem}
.tbl th{text-align:left;padding:10px 8px;border-bottom:1px solid var(--border);color:var(--muted);font-weight:600;font-size:.75rem;text-transform:uppercase;letter-spacing:.04em}
.tbl td{padding:10px 8px;border-bottom:1px solid rgba(255,255,255,.04);color:var(--text)}
.tbl tr:hover td{background:rgba(255,255,255,.02)}
.tag{display:inline-block;font-size:.68rem;padding:3px 8px;border-radius:6px;font-weight:600;margin:1px}
.tag-cat{background:rgba(53,110,181,.15);color:#6ba3e8}
.tag-reg{background:rgba(34,197,94,.12);color:#4ade80}
.tag-sea{background:rgba(245,158,11,.12);color:#fbbf24}
.tag-sty{background:rgba(168,85,247,.12);color:#c084fc}
.tag-dev{background:rgba(255,255,255,.06);color:var(--muted)}
#loading{text-align:center;padding:60px;color:var(--muted);font-size:.95rem}
</style>
</head>
<body>
<div class="top">
  <div class="top-left">
    <img src="/static/widget/letizia.png" alt="">
    <h1>Letizia <span>Analytics</span></h1>
  </div>
  <div class="live"><span class="live-dot"></span> Live</div>
</div>
<div class="wrap" id="content"><div id="loading">Chargement des donnees...</div></div>

<script>
var KEY=new URLSearchParams(location.search).get('key')||'';
var charts={};
var PAL=['#356eb5','#22c55e','#f59e0b','#ec4899','#06b6d4','#a855f7','#ef4444','#f97316','#14b8a6','#8b5cf6','#64748b','#e11d48'];

function fetchStats(){
  fetch('/api/admin/stats?key='+encodeURIComponent(KEY))
    .then(function(r){return r.json()})
    .then(render)
    .catch(function(e){document.getElementById('content').innerHTML='<p style="color:#ef4444;padding:60px;text-align:center">Erreur: '+e.message+'</p>'});
}
function pct(n,t){return t?Math.round(n/t*100):0}

function makeDonut(id,data,title){
  var ctx=document.getElementById(id);
  if(!ctx)return;
  var entries=Object.entries(data).sort(function(a,b){return b[1]-a[1]});
  if(charts[id])charts[id].destroy();
  charts[id]=new Chart(ctx,{
    type:'doughnut',
    data:{labels:entries.map(function(e){return e[0]}),datasets:[{data:entries.map(function(e){return e[1]}),backgroundColor:PAL.slice(0,entries.length),borderWidth:0,hoverOffset:6}]},
    options:{responsive:true,maintainAspectRatio:false,cutout:'62%',plugins:{legend:{position:'right',labels:{color:'#8b8fa3',font:{size:11},padding:10,usePointStyle:true,pointStyleWidth:8}}}}
  });
}
function makeBar(id,data,color){
  var ctx=document.getElementById(id);
  if(!ctx)return;
  var entries=Object.entries(data).sort(function(a,b){return b[1]-a[1]});
  if(charts[id])charts[id].destroy();
  charts[id]=new Chart(ctx,{
    type:'bar',
    data:{labels:entries.map(function(e){return e[0]}),datasets:[{data:entries.map(function(e){return e[1]}),backgroundColor:color||'#356eb5',borderRadius:6,barPercentage:.7}]},
    options:{responsive:true,maintainAspectRatio:false,indexAxis:'y',plugins:{legend:{display:false}},scales:{x:{grid:{color:'rgba(255,255,255,.05)'},ticks:{color:'#8b8fa3',font:{size:11}}},y:{grid:{display:false},ticks:{color:'#e1e4ed',font:{size:11}}}}}
  });
}

function render(d){
  var mob=d.by_device.mobile||0;
  var desk=d.by_device.desktop||0;
  var h='';
  h+='<div class="kpis">';
  h+='<div class="kpi"><div class="val">'+d.total+'</div><div class="lbl">Total conversations</div></div>';
  h+='<div class="kpi"><div class="val">'+d.today+'</div><div class="lbl">Aujourd\'hui</div></div>';
  h+='<div class="kpi"><div class="val">'+d.this_week+'</div><div class="lbl">Cette semaine</div></div>';
  h+='<div class="kpi"><div class="val">'+pct(mob,d.total)+'%</div><div class="lbl">Mobile</div></div>';
  h+='<div class="kpi"><div class="val">'+pct(desk,d.total)+'%</div><div class="lbl">Desktop</div></div>';
  h+='</div>';

  h+='<div class="grid2">';
  h+='<div class="card"><h3>Categories</h3><div style="height:260px"><canvas id="chCat"></canvas></div></div>';
  h+='<div class="card"><h3>Regions</h3><div style="height:260px"><canvas id="chReg"></canvas></div></div>';
  h+='</div>';

  h+='<div class="grid3">';
  h+='<div class="card"><h3>Saisons</h3><div style="height:220px"><canvas id="chSea"></canvas></div></div>';
  h+='<div class="card"><h3>Style de voyage</h3><div style="height:220px"><canvas id="chSty"></canvas></div></div>';
  h+='<div class="card"><h3>Devices</h3><div style="height:220px"><canvas id="chDev"></canvas></div></div>';
  h+='</div>';

  h+='<div class="tbl"><h3>Dernieres conversations</h3>';
  h+='<table><thead><tr><th>Question</th><th>Categorie</th><th>Region</th><th>Saison</th><th>Style</th><th>Device</th><th>Date</th></tr></thead><tbody>';
  d.recent.slice(0,50).forEach(function(r){
    var ts=r.ts?new Date(r.ts).toLocaleString('fr-FR',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'}):'';
    h+='<tr>';
    h+='<td>'+r.question.substring(0,100)+'</td>';
    h+='<td><span class="tag tag-cat">'+r.category+'</span></td>';
    h+='<td><span class="tag tag-reg">'+r.region+'</span></td>';
    h+='<td><span class="tag tag-sea">'+r.season+'</span></td>';
    h+='<td><span class="tag tag-sty">'+r.style+'</span></td>';
    h+='<td><span class="tag tag-dev">'+r.device+'</span></td>';
    h+='<td style="white-space:nowrap;color:#8b8fa3">'+ts+'</td>';
    h+='</tr>';
  });
  h+='</tbody></table></div>';

  document.getElementById('content').innerHTML=h;

  setTimeout(function(){
    makeDonut('chCat',d.by_category);
    makeBar('chReg',d.by_region,'#22c55e');
    makeDonut('chSea',d.by_season);
    makeDonut('chSty',d.by_style);
    makeDonut('chDev',d.by_device);
  },50);
}
fetchStats();
setInterval(fetchStats,60000);
</script>
</body>
</html>"""
        return HTMLResponse(content=html)

    return app


settings = get_settings()
app = build_app(settings)
