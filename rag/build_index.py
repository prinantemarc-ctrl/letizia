"""
Construit l'index vectoriel Chroma Cloud à partir de data/raw/pages_fr.jsonl.

Usage (depuis la racine visit-corsica-chatbot):
  python -m rag.build_index

Requiert les variables d'environnement :
  OPENAI_API_KEY, CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import chromadb
from openai import OpenAI

from rag.chunking import chunk_text

COLLECTION = "visit_corsica_fr"
EMBEDDING_MODEL = "text-embedding-3-small"


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=Path, default=Path("data/raw/pages_fr.jsonl"))
    parser.add_argument("--reset", action="store_true", help="Supprime la collection existante")
    args = parser.parse_args()

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    chroma_host = os.environ.get("CHROMA_HOST", "api.trychroma.com")
    chroma_api_key = os.environ.get("CHROMA_API_KEY", "")
    chroma_tenant = os.environ.get("CHROMA_TENANT", "")
    chroma_database = os.environ.get("CHROMA_DATABASE", "")

    if not openai_key:
        print("OPENAI_API_KEY requis pour les embeddings.", file=sys.stderr)
        sys.exit(1)
    if not chroma_api_key or not chroma_tenant or not chroma_database:
        print("CHROMA_API_KEY, CHROMA_TENANT et CHROMA_DATABASE requis.", file=sys.stderr)
        sys.exit(1)

    if not args.pages.is_file():
        print(f"Fichier introuvable: {args.pages}", file=sys.stderr)
        sys.exit(1)

    oa = OpenAI(api_key=openai_key)

    print("Connexion à Chroma Cloud...", file=sys.stderr)
    client = chromadb.HttpClient(
        host=chroma_host,
        port=443,
        ssl=True,
        headers={"X-Chroma-Token": chroma_api_key},
        tenant=chroma_tenant,
        database=chroma_database,
    )

    if args.reset:
        try:
            client.delete_collection(COLLECTION)
            print(f"Collection '{COLLECTION}' supprimée.", file=sys.stderr)
        except Exception:  # noqa: BLE001
            pass

    collection = client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    with args.pages.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            url = row.get("url") or ""
            text = row.get("text") or ""
            if not text:
                continue
            parts = chunk_text(text)
            for i, part in enumerate(parts):
                ids.append(str(uuid.uuid4()))
                documents.append(part)
                metadatas.append({"url": url, "chunk_index": i})

    if not documents:
        print("Aucun document à indexer.", file=sys.stderr)
        sys.exit(1)

    batch = 64
    total = len(documents)
    print(f"Indexation de {total} chunks (batch={batch})...", file=sys.stderr)

    for start in range(0, total, batch):
        end = min(start + batch, total)
        batch_docs = documents[start:end]

        emb = embed_batch(oa, batch_docs)
        collection.add(
            ids=ids[start:end],
            documents=batch_docs,
            metadatas=metadatas[start:end],
            embeddings=emb,
        )

        done = min(end, total)
        print(f"  {done}/{total} chunks indexés", file=sys.stderr)
        if end < total:
            time.sleep(0.2)

    print(f"OK -> Chroma Cloud collection '{COLLECTION}' ({total} chunks)", file=sys.stderr)


if __name__ == "__main__":
    main()
