"""
Construit l'index vectoriel Chroma à partir de data/raw/pages_fr.jsonl.

Usage (depuis la racine visit-corsica-chatbot):
  python -m rag.build_index
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from rag.chunking import chunk_text

COLLECTION = "visit_corsica_fr"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=Path, default=Path("data/raw/pages_fr.jsonl"))
    parser.add_argument("--chroma", type=Path, default=Path("data/chroma"))
    parser.add_argument("--reset", action="store_true", help="Supprime la collection existante")
    args = parser.parse_args()

    if not args.pages.is_file():
        print(f"Fichier introuvable: {args.pages}", file=sys.stderr)
        sys.exit(1)

    args.chroma.mkdir(parents=True, exist_ok=True)

    print("Chargement du modèle d'embeddings (premier lancement : téléchargement)...", file=sys.stderr)
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(args.chroma), settings=Settings(anonymized_telemetry=False))

    if args.reset:
        try:
            client.delete_collection(COLLECTION)
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
    print(f"Indexation de {len(documents)} chunks...", file=sys.stderr)
    for start in range(0, len(documents), batch):
        end = min(start + batch, len(documents))
        emb = model.encode(documents[start:end], normalize_embeddings=True).tolist()
        collection.add(ids=ids[start:end], documents=documents[start:end], metadatas=metadatas[start:end], embeddings=emb)

    print(f"OK -> Chroma: {args.chroma} (collection {COLLECTION})", file=sys.stderr)


if __name__ == "__main__":
    main()
