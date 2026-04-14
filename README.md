# Chatbot RAG — Visit Corsica

Projet autonome : crawl du contenu français de [visit-corsica.com](https://www.visit-corsica.com/fr/), index sémantique local (Chroma + embeddings multilingues) et API FastAPI consommée par un widget JavaScript intégrable sur un autre site.

## Prérequis

- Python 3.11+
- Compte OpenAI (ou [Ollama](https://ollama.com) en local avec API compatible OpenAI)

## Installation

```bash
cd visit-corsica-chatbot
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Éditer .env : ANTHROPIC_API_KEY (Claude) et/ou OPENAI_API_KEY, etc.
```

## 1. Scraper tout le site (FR)

Le fichier [robots.txt](https://www.visit-corsica.com/robots.txt) impose un **Crawl-delay de 10 secondes**. Pour **~528 URLs** du [sitemap français](https://www.visit-corsica.com/sitemap_fr.xml), le crawl complet prend **environ 1,5 h**.

```bash
# Crawl complet (respect du délai robots.txt)
python -m scraper.scrape --out data/raw/pages_fr.jsonl

# Test rapide (ex. 30 pages, délai réduit — réservé au développement)
python -m scraper.scrape --max-pages 30 --delay 1 --out data/raw/pages_fr.jsonl
```

Les pages sont enregistrées en **JSONL** : une ligne par URL (`url`, `text`, `error`).

## 2. Construire l’index vectoriel

```bash
python -m rag.build_index --pages data/raw/pages_fr.jsonl --chroma data/chroma --reset
```

## 3. Lancer l’API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8765
```

- Santé : `GET /health`
- Chat : `POST /api/chat` avec corps JSON `{"message":"..."}`

Une clé LLM est **obligatoire** : avec **`ANTHROPIC_API_KEY`**, la génération utilise **Claude** (modèle configurable via `ANTHROPIC_MODEL`). Sinon, avec **`OPENAI_API_KEY`** (ou Ollama + `OPENAI_BASE_URL`), c’est OpenAI / compatible. Sans aucune de ces clés, **`POST /api/chat`** renvoie **503** (le serveur doit être configuré avant usage).

### Ollama (exemple)

```env
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_CHAT_MODEL=llama3.2
```

## 4. Intégrer le widget sur un autre site

1. Héberger `widget/visit-corsica-chat.js` (même domaine ou CDN).
2. Votre API doit être accessible en **HTTPS** en production et autoriser le **CORS** (voir `CORS_ORIGINS` dans `.env`).
3. Dans la page :

```html
<script>
  window.VISIT_CORSA_CHAT_API = 'https://votre-serveur.com';
</script>
<script src="/chemin/vers/visit-corsica-chat.js" defer></script>
```

## Propriété intellectuelle et usage

Les contenus du site officiel relèvent de l’**Agence du tourisme de la Corse** / éditeurs du site. Ce dépôt fournit un **cadre technique** (crawl poli, RAG, API) : vérifiez les **conditions d’utilisation**, les **droits de réutilisation** et obtenez les **accords nécessaires** avant toute mise en production ou republication massive des textes.
