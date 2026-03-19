import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "/app/data/chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

os.makedirs(CHROMA_DB_PATH, exist_ok=True)

_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_collection():
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name="academic_memories",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed_and_store(document_id: str, text: str, metadata: Dict[str, Any]) -> None:
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True).tolist()
    collection = get_collection()

    clean_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            clean_metadata[k] = v
        else:
            clean_metadata[k] = str(v)

    existing = collection.get(ids=[document_id])
    if existing and existing["ids"]:
        collection.update(
            ids=[document_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[clean_metadata],
        )
    else:
        collection.add(
            ids=[document_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[clean_metadata],
        )


def semantic_search(
    query_text: str, user_id: str, top_k: int = 5
) -> List[Dict[str, Any]]:
    model = get_embedding_model()
    query_embedding = model.encode(query_text, normalize_embeddings=True).tolist()
    collection = get_collection()

    count = collection.count()
    if count == 0:
        return []

    effective_k = min(top_k, count)

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_k,
            where={"user_id": user_id} if user_id else None,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        # When n_results exceeds the number of documents matching the user_id
        # filter, ChromaDB raises a ValueError. Retry with n_results=1 to
        # guarantee the closest result is returned.
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
                where={"user_id": user_id} if user_id else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

    output = []
    if not results["ids"] or not results["ids"][0]:
        return output

    for i, doc_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i]
        score = round(1 - distance, 4)
        output.append(
            {
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": score,
            }
        )

    return output


def get_vector_count() -> int:
    collection = get_collection()
    return collection.count()
