import re
from dataclasses import dataclass
from typing import Any

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.core.config import Settings

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


@dataclass
class RetrievalContext:
    embedding_model: SentenceTransformer
    collection: Any
    bm25_index: BM25Okapi
    chunk_ids: list[str]
    chunk_id_to_text: dict[str, str]
    chunk_id_to_metadata: dict[str, dict]
    chunk_id_to_embedding: dict[str, list[float]]


def tokenize_for_bm25(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def load_retrieval_context(settings: Settings) -> RetrievalContext:
    embedding_model = SentenceTransformer(settings.embedding_model_name)

    chroma_client = chromadb.PersistentClient(path=str(settings.vector_store_path))
    collection = chroma_client.get_collection(name=settings.collection_name)

    stored = collection.get(include=["documents", "metadatas", "embeddings"])
    chunk_ids = stored["ids"]
    documents = stored["documents"]
    metadatas = stored["metadatas"]
    embeddings = stored["embeddings"]

    chunk_id_to_text = dict(zip(chunk_ids, documents))
    chunk_id_to_metadata = dict(zip(chunk_ids, metadatas))
    chunk_id_to_embedding = dict(zip(chunk_ids, embeddings))

    bm25_corpus_tokens = [tokenize_for_bm25(text) for text in documents]
    bm25_index = BM25Okapi(bm25_corpus_tokens)

    return RetrievalContext(
        embedding_model=embedding_model,
        collection=collection,
        bm25_index=bm25_index,
        chunk_ids=chunk_ids,
        chunk_id_to_text=chunk_id_to_text,
        chunk_id_to_metadata=chunk_id_to_metadata,
        chunk_id_to_embedding=chunk_id_to_embedding,
    )


def dense_search_ids(context: RetrievalContext, query: str, n: int) -> list[str]:
    query_embedding = context.embedding_model.encode([query]).tolist()
    results = context.collection.query(query_embeddings=query_embedding, n_results=n)
    return results["ids"][0]


def bm25_search_ids(context: RetrievalContext, query: str, n: int) -> list[str]:
    query_tokens = tokenize_for_bm25(query)
    scores = context.bm25_index.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [context.chunk_ids[i] for i in ranked_indices]


def reciprocal_rank_fusion(ranked_id_lists: list[list[str]], k: int) -> list[tuple[str, float]]:
    fused_scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, doc_id in enumerate(ranked_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    a = np.array(vector_a)
    b = np.array(vector_b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def rerank_by_cosine_similarity(
    context: RetrievalContext, query: str, candidate_ids: list[str], top_k: int
) -> list[str]:
    query_embedding = context.embedding_model.encode([query])[0]
    scored = []
    for doc_id in candidate_ids:
        similarity = cosine_similarity(query_embedding, context.chunk_id_to_embedding[doc_id])
        scored.append((doc_id, similarity))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, similarity in scored[:top_k]]


def hybrid_retrieve(context: RetrievalContext, settings: Settings, query: str) -> list[dict]:
    dense_ids = dense_search_ids(context, query, n=settings.source_pool_size)
    sparse_ids = bm25_search_ids(context, query, n=settings.source_pool_size)
    fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=settings.rrf_k)
    candidate_ids = [doc_id for doc_id, score in fused[: settings.fused_candidate_pool]]
    reranked_ids = rerank_by_cosine_similarity(context, query, candidate_ids, settings.final_top_k)

    retrieved = []
    for doc_id in reranked_ids:
        metadata = context.chunk_id_to_metadata[doc_id]
        retrieved.append({
            "chunk_id": doc_id,
            "text": context.chunk_id_to_text[doc_id],
            "metadata": {
                "source_id": metadata.get("source_id", ""),
                "title": metadata.get("title", ""),
                "authors": metadata.get("authors", ""),
                "page": metadata.get("page", ""),
                "section": metadata.get("section", ""),
                "source_url": metadata.get("source_url", ""),
                "topic": metadata.get("topic", ""),
            },
        })
    return retrieved
