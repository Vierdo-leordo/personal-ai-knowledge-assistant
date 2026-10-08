"""Vector Store: persist embeddings and run similarity search.

``ChromaVectorStore`` is used when ``chromadb`` is installed; ``NumpyVectorStore`` is a small,
dependency-free fallback with the same interface (handy for tests and tiny deployments).
Each embedding model gets its own collection so vectors of different dimensions never mix.
"""
from __future__ import annotations

import json
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import Settings


@dataclass
class SearchHit:
    chunk_id: str
    text: str
    metadata: Dict
    score: float  # cosine similarity in [-1, 1]


def _collection_name(embedder_name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"kb_{embedder_name}")[:60]
    return name.strip("_-") or "kb_default"


def _matches(meta: Dict, where: Optional[Dict]) -> bool:
    if not where:
        return True
    for key, cond in where.items():
        val = meta.get(key)
        if isinstance(cond, dict) and "$in" in cond:
            if val not in cond["$in"]:
                return False
        elif val != cond:
            return False
    return True


class VectorStore(ABC):
    backend: str = "base"

    @abstractmethod
    def add(self, ids: Sequence[str], embeddings: np.ndarray, documents: Sequence[str], metadatas: Sequence[Dict]): ...

    @abstractmethod
    def query(self, embedding: np.ndarray, k: int = 5, where: Optional[Dict] = None) -> List[SearchHit]: ...

    @abstractmethod
    def get(self, where: Optional[Dict] = None, limit: Optional[int] = None) -> List[SearchHit]: ...

    @abstractmethod
    def delete(self, where: Dict) -> None: ...

    @abstractmethod
    def update_metadata(self, where: Dict, new_values: Dict) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    def reset(self) -> None:
        self.delete({})


class NumpyVectorStore(VectorStore):
    backend = "numpy"

    def __init__(self, directory: Path, collection: str):
        self.dir = Path(directory) / collection
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.ids: List[str] = []
        self.docs: List[str] = []
        self.metas: List[Dict] = []
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self._load()

    # persistence ---------------------------------------------------------
    def _load(self):
        meta_file, vec_file = self.dir / "records.json", self.dir / "vectors.npy"
        if meta_file.exists() and vec_file.exists():
            records = json.loads(meta_file.read_text())
            self.ids = [r["id"] for r in records]
            self.docs = [r["text"] for r in records]
            self.metas = [r["metadata"] for r in records]
            self.matrix = np.load(vec_file)

    def _save(self):
        records = [{"id": i, "text": d, "metadata": m} for i, d, m in zip(self.ids, self.docs, self.metas)]
        (self.dir / "records.json").write_text(json.dumps(records))
        np.save(self.dir / "vectors.npy", self.matrix)

    # api -------------------------------------------------------------------
    def add(self, ids, embeddings, documents, metadatas):
        emb = np.asarray(embeddings, dtype=np.float32)
        with self._lock:
            existing = {cid: i for i, cid in enumerate(self.ids)}
            new_rows = []
            for cid, vec, doc, meta in zip(ids, emb, documents, metadatas):
                if cid in existing:  # upsert
                    j = existing[cid]
                    self.matrix[j], self.docs[j], self.metas[j] = vec, doc, dict(meta)
                else:
                    self.ids.append(cid)
                    self.docs.append(doc)
                    self.metas.append(dict(meta))
                    new_rows.append(vec)
            if new_rows:
                block = np.vstack(new_rows)
                self.matrix = block if self.matrix.size == 0 else np.vstack([self.matrix, block])
            self._save()

    def query(self, embedding, k=5, where=None):
        if not self.ids:
            return []
        q = np.asarray(embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        mask = np.array([_matches(m, where) for m in self.metas])
        if not mask.any():
            return []
        scores = self.matrix @ q
        scores[~mask] = -np.inf
        top = np.argsort(-scores)[: min(k, int(mask.sum()))]
        return [SearchHit(self.ids[i], self.docs[i], dict(self.metas[i]), float(scores[i])) for i in top]

    def get(self, where=None, limit=None):
        hits = [
            SearchHit(i, d, dict(m), 0.0) for i, d, m in zip(self.ids, self.docs, self.metas) if _matches(m, where)
        ]
        hits.sort(key=lambda h: (h.metadata.get("doc_id", ""), h.metadata.get("chunk_index", 0)))
        return hits[:limit] if limit else hits

    def delete(self, where):
        with self._lock:
            keep = [i for i, m in enumerate(self.metas) if not (where and _matches(m, where))] if where else []
            self.ids = [self.ids[i] for i in keep]
            self.docs = [self.docs[i] for i in keep]
            self.metas = [self.metas[i] for i in keep]
            self.matrix = self.matrix[keep] if keep else np.zeros((0, 0), dtype=np.float32)
            self._save()

    def update_metadata(self, where, new_values):
        with self._lock:
            for m in self.metas:
                if _matches(m, where):
                    m.update(new_values)
            self._save()

    def count(self):
        return len(self.ids)


class ChromaVectorStore(VectorStore):
    backend = "chromadb"

    def __init__(self, directory: Path, collection: str):
        import chromadb

        self.client = chromadb.PersistentClient(path=str(directory))
        self.collection = self.client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})

    @staticmethod
    def _where(where: Optional[Dict]):
        if not where:
            return None
        clauses = [{k: v} if isinstance(v, dict) else {k: {"$eq": v}} for k, v in where.items()]
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def add(self, ids, embeddings, documents, metadatas):
        emb = np.asarray(embeddings, dtype=np.float32).tolist()
        for s in range(0, len(ids), 1000):
            self.collection.upsert(
                ids=list(ids[s : s + 1000]),
                embeddings=emb[s : s + 1000],
                documents=list(documents[s : s + 1000]),
                metadatas=[dict(m) for m in metadatas[s : s + 1000]],
            )

    def query(self, embedding, k=5, where=None):
        n = self.collection.count()
        if n == 0:
            return []
        res = self.collection.query(
            query_embeddings=[np.asarray(embedding, dtype=np.float32).tolist()],
            n_results=min(k, n),
            where=self._where(where),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(SearchHit(cid, doc, dict(meta), 1.0 - float(dist)))
        return hits

    def get(self, where=None, limit=None):
        res = self.collection.get(where=self._where(where), limit=limit, include=["documents", "metadatas"])
        hits = [SearchHit(i, d, dict(m), 0.0) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
        hits.sort(key=lambda h: (h.metadata.get("doc_id", ""), h.metadata.get("chunk_index", 0)))
        return hits

    def delete(self, where):
        if not where:
            ids = self.collection.get(include=[])["ids"]
            if ids:
                self.collection.delete(ids=ids)
            return
        self.collection.delete(where=self._where(where))

    def update_metadata(self, where, new_values):
        res = self.collection.get(where=self._where(where), include=["metadatas"])
        if res["ids"]:
            metas = [{**m, **new_values} for m in res["metadatas"]]
            self.collection.update(ids=res["ids"], metadatas=metas)

    def count(self):
        return self.collection.count()


def get_vector_store(settings: Settings, embedder_name: str) -> VectorStore:
    collection = _collection_name(embedder_name)
    choice = settings.vector_store
    if choice in ("chroma", "chromadb", "auto"):
        try:
            return ChromaVectorStore(settings.vector_dir / "chroma", collection)
        except ImportError:
            if choice != "auto":
                raise
    return NumpyVectorStore(settings.vector_dir / "numpy", collection)
