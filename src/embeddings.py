"""Embedding Generator: convert text into vectors.

Providers (chosen by ``EMBEDDING_PROVIDER``):
  * ``sentence-transformers`` – local model, free, recommended (all-MiniLM-L6-v2)
  * ``gemini``                – Gemini Embeddings API
  * ``openai``                – OpenAI Embeddings API
  * ``hashing``               – dependency-free lexical hashing embedder (offline fallback)
"""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np
import requests

from .config import Settings

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = set(
    """a an the and or but if of to in on for with at by from as is are was were be been being this that these
    those it its into than then so such not no can could should would will shall may might do does did done have has
    had having i you he she we they them his her our your their what which who whom whose how when where why also
    about over under between within via per each any all more most other some very just only own same too s t""".split()
)


# words that describe the *request* rather than the topic ("explain X using my notes")
QUERY_FILLER = set(
    """explain describe tell give show using use uses used material materials document documents notes note
    uploaded upload file files please help understand information according mentioned based want know""".split()
)


def query_terms(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in QUERY_FILLER]


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    toks = _TOKEN.findall(text.lower())
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]
    return toks


def _stem(tok: str) -> str:
    """Very light suffix stripping so 'learning'/'learned'/'learns' collide."""
    for suf in ("ings", "ing", "edly", "ed", "ies", "es", "s", "ly"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)] + ("y" if suf == "ies" else "")
    return tok


class Embedder(ABC):
    name: str = "base"
    dimension: int = 0

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        mat = np.asarray(mat, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class HashingEmbedder(Embedder):
    """Offline embedder: signed feature hashing of stemmed unigrams, bigrams and character trigrams.

    Not semantic like a neural model, but deterministic, fast, and good enough for keyword-heavy
    study material. Used automatically when no other provider is available.
    """

    name = "hashing"

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension

    def _features(self, text: str):
        toks = [_stem(t) for t in tokenize(text)]
        for t in toks:
            yield t, 1.0
            padded = f"#{t}#"
            for i in range(len(padded) - 2):
                yield "c:" + padded[i : i + 3], 0.25
        for a, b in zip(toks, toks[1:]):
            yield f"{a}_{b}", 0.7

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for feat, w in self._features(text):
                h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
                idx = h % self.dimension
                sign = 1.0 if (h >> 64) & 1 else -1.0
                counts[idx] = counts.get(idx, 0.0) + sign * w
            for idx, val in counts.items():
                mat[row, idx] = math.copysign(math.log1p(abs(val)), val)
        return self._normalize(mat)


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.name = f"st-{model_name.split('/')[-1]}"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode(list(texts), batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


class GeminiEmbedder(Embedder):
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"

    def __init__(self, api_key: str, model: str = "gemini-embedding-001", dimension: int = 768):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.api_key, self.model, self.dimension = api_key, model, dimension
        self.name = f"gemini-{model}"

    def _embed(self, texts: Sequence[str], task: str) -> np.ndarray:
        out: List[List[float]] = []
        for start in range(0, len(texts), 100):  # API batch limit
            batch = texts[start : start + 100]
            body = {
                "requests": [
                    {
                        "model": f"models/{self.model}",
                        "content": {"parts": [{"text": t[:8000]}]},
                        "taskType": task,
                        "outputDimensionality": self.dimension,
                    }
                    for t in batch
                ]
            }
            r = requests.post(
                self.URL.format(model=self.model),
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json=body,
                timeout=60,
            )
            r.raise_for_status()
            out.extend(e["values"] for e in r.json()["embeddings"])
        return self._normalize(np.array(out))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(list(texts), "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class OpenAIEmbedder(Embedder):
    URL = "https://api.openai.com/v1/embeddings"

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.api_key, self.model = api_key, model
        self.dimension = 3072 if "large" in model else 1536
        self.name = f"openai-{model}"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        out: List[List[float]] = []
        texts = list(texts)
        for start in range(0, len(texts), 256):
            r = requests.post(
                self.URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": [t[:8000] for t in texts[start : start + 256]]},
                timeout=60,
            )
            r.raise_for_status()
            out.extend(d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"]))
        return self._normalize(np.array(out))


def get_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider
    if provider in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder(settings.sentence_transformer_model)
    if provider == "gemini":
        return GeminiEmbedder(settings.gemini_api_key, settings.gemini_embedding_model)
    if provider == "openai":
        return OpenAIEmbedder(settings.openai_api_key, settings.openai_embedding_model)
    if provider == "hashing":
        return HashingEmbedder()
    # auto
    try:
        return SentenceTransformerEmbedder(settings.sentence_transformer_model)
    except Exception:
        pass
    if settings.gemini_api_key:
        return GeminiEmbedder(settings.gemini_api_key, settings.gemini_embedding_model)
    if settings.openai_api_key:
        return OpenAIEmbedder(settings.openai_api_key, settings.openai_embedding_model)
    return HashingEmbedder()
