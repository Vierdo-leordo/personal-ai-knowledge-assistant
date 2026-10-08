"""Retriever: semantic search with hybrid (vector + keyword) re-ranking and multi-document diversity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .embeddings import Embedder, _stem, query_terms, tokenize
from .vector_store import SearchHit, VectorStore


@dataclass
class RetrievedChunk:
    text: str
    metadata: Dict
    score: float  # final (hybrid) relevance score
    vector_score: float
    keyword_score: float

    @property
    def file_name(self) -> str:
        return self.metadata.get("file_name", "unknown")

    @property
    def page(self) -> int:
        return int(self.metadata.get("page", 0) or 0)

    @property
    def chunk_id(self) -> str:
        return self.metadata.get("chunk_id", "")

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "file_name": self.file_name,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "doc_id": self.metadata.get("doc_id"),
            "category": self.metadata.get("category"),
            "heading": self.metadata.get("heading", ""),
            "score": round(self.score, 4),
            "vector_score": round(self.vector_score, 4),
            "keyword_score": round(self.keyword_score, 4),
        }


def keyword_overlap(query: str, text: str) -> float:
    q = {_stem(t) for t in query_terms(query)}
    if not q:
        return 0.0
    t = {_stem(x) for x in tokenize(text)}
    return len(q & t) / len(q)


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, keyword_weight: float = 0.25):
        self.embedder = embedder
        self.store = store
        self.keyword_weight = keyword_weight

    @staticmethod
    def build_filter(doc_ids: Optional[Sequence[str]] = None, category: Optional[str] = None) -> Optional[Dict]:
        where: Dict = {}
        if doc_ids:
            doc_ids = list(doc_ids)
            where["doc_id"] = doc_ids[0] if len(doc_ids) == 1 else {"$in": doc_ids}
        if category and category != "All":
            where["category"] = category
        return where or None

    def retrieve(
        self,
        query: str,
        k: int = 5,
        doc_ids: Optional[Sequence[str]] = None,
        category: Optional[str] = None,
        diversify: bool = False,
        min_score: float = 0.0,
    ) -> List[RetrievedChunk]:
        """Return the top-k chunks for ``query``.

        ``diversify=True`` (multi-document research) guarantees results are spread across files
        instead of all coming from the single most relevant document.
        """
        if not query.strip() or self.store.count() == 0:
            return []
        where = self.build_filter(doc_ids, category)
        q_vec = self.embedder.embed_query(query)
        candidates: List[SearchHit] = self.store.query(q_vec, k=max(k * 4, 20), where=where)

        ranked: List[RetrievedChunk] = []
        for hit in candidates:
            kw = keyword_overlap(query, hit.text)
            final = (1 - self.keyword_weight) * hit.score + self.keyword_weight * kw
            ranked.append(RetrievedChunk(hit.text, hit.metadata, final, hit.score, kw))
        ranked.sort(key=lambda r: r.score, reverse=True)
        ranked = [r for r in ranked if r.score >= min_score]

        if diversify:
            ranked = self._round_robin_by_document(ranked)
        return ranked[:k]

    @staticmethod
    def _round_robin_by_document(ranked: List[RetrievedChunk]) -> List[RetrievedChunk]:
        buckets: Dict[str, List[RetrievedChunk]] = {}
        for r in ranked:
            buckets.setdefault(r.metadata.get("doc_id", r.file_name), []).append(r)
        order = sorted(buckets.values(), key=lambda b: b[0].score, reverse=True)
        out: List[RetrievedChunk] = []
        while any(order):
            for b in order:
                if b:
                    out.append(b.pop(0))
        return out

    def get_document_chunks(self, doc_id: str) -> List[RetrievedChunk]:
        hits = self.store.get(where={"doc_id": doc_id})
        return [RetrievedChunk(h.text, h.metadata, 1.0, 1.0, 0.0) for h in hits]
