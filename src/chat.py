"""KnowledgeAssistant: the orchestrator that wires the whole RAG pipeline together.

Upload -> Document Processing -> Chunking -> Embeddings -> Vector DB
Question -> Semantic Search -> Context -> LLM -> Answer + Citations  (+ conversation memory)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

from . import prompts
from .chunker import chunk_document, make_doc_id
from .citation import (
    Source,
    build_context,
    build_sources,
    citation_coverage,
    cited_sources,
    hallucination_rate,
    strip_invalid_citations,
)
from .config import Settings, get_settings
from .document_loader import load_document
from .embeddings import Embedder, get_embedder
from .llm_engine import LLMEngine, LLMError
from .memory import MemoryStore
from .offline import extractive_answer
from .retriever import RetrievedChunk, Retriever
from .vector_store import VectorStore, get_vector_store


@dataclass
class IngestResult:
    doc_id: str
    file_name: str
    num_pages: int
    num_chunks: int
    duplicate: bool = False
    message: str = ""


@dataclass
class AssistantResponse:
    answer: str
    sources: List[Source]
    cited: List[Source]
    retrieved: List[RetrievedChunk]
    intent: str = "qa"
    search_query: str = ""
    metrics: Dict = field(default_factory=dict)
    extra: Dict = field(default_factory=dict)  # structured payloads (flashcards, quiz, …)

    def to_dict(self) -> Dict:
        return {
            "answer": self.answer,
            "intent": self.intent,
            "search_query": self.search_query,
            "sources": [s.to_dict() for s in self.cited],
            "retrieved": [r.to_dict() for r in self.retrieved],
            "metrics": self.metrics,
            "extra": self.extra,
        }


# keyword router for the agent-based workflow (no extra LLM call needed)
BUSY_NOTE = ("> The AI service is busy or unavailable right now, so these are the most relevant passages from "
             "your documents. Ask again in a minute for a full answer.\n\n")

_INTENT_RULES = [
    ("exam_prep", re.compile(r"\b(prepare|prep|get ready)\b.*\b(exam|test|interview|quiz)\b|\brevision plan\b", re.I)),
    ("flashcards", re.compile(r"\bflash ?cards?\b", re.I)),
    ("quiz", re.compile(r"\b(quiz|mcqs?|practice (test|questions)|test me)\b", re.I)),
    ("notes", re.compile(r"\b(study notes|chapter notes|revision guide|exam summary|make notes|notes on)\b", re.I)),
    ("summarize", re.compile(r"\b(summari[sz]e|summary of|tl;?dr|key findings)\b", re.I)),
    ("compare", re.compile(r"\b(compare|comparison|contrast|difference(s)? between|vs\.?|versus)\b", re.I)),
]


def detect_intent(question: str) -> str:
    for label, pat in _INTENT_RULES:
        if pat.search(question):
            return label
    return "qa"


class KnowledgeAssistant:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
        llm: Optional[LLMEngine] = None,
    ):
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.embedder = embedder or get_embedder(self.settings)
        self.store = store or get_vector_store(self.settings, self.embedder.name)
        self.llm = llm or LLMEngine(self.settings)
        self.memory = MemoryStore(self.settings.db_path)
        self.retriever = Retriever(self.embedder, self.store)

    # ================================================================ status
    def status(self) -> Dict:
        return {
            "llm_provider": self.llm.provider,
            "llm_model": self.llm.model_name,
            "embedder": self.embedder.name,
            "vector_store": self.store.backend,
            "documents": len(self.memory.list_documents()),
            "chunks": self.store.count(),
        }

    def stale_documents(self) -> List[Dict]:
        """Documents indexed with a different embedding model than the current one (need re-indexing)."""
        return [d for d in self.memory.list_documents() if d.get("embedder") and d["embedder"] != self.embedder.name]

    # ================================================================ ingestion
    def ingest_file(
        self,
        source: Union[str, Path, bytes],
        file_name: Optional[str] = None,
        category: str = "General",
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> IngestResult:
        def report(p: float, msg: str):
            if progress:
                progress(p, msg)

        report(0.05, "Extracting text…")
        doc = load_document(source, file_name)
        doc_id = make_doc_id(doc.content_hash)

        existing = self.memory.get_document_by_hash(doc.content_hash)
        if existing and self.store.get(where={"doc_id": doc_id}, limit=1):
            report(1.0, "Already in knowledge base")
            return IngestResult(doc_id, existing["file_name"], existing["num_pages"], existing["num_chunks"], True,
                                "This file was already uploaded; skipped re-processing.")
        if not doc.pages:
            raise ValueError(
                f"No extractable text found in {doc.file_name}. If it is a scanned PDF, run OCR on it first."
            )

        report(0.2, "Chunking…")
        upload_date = datetime.now(timezone.utc).isoformat(timespec="seconds")
        chunks = chunk_document(doc, self.settings.chunk_size, self.settings.chunk_overlap, category, upload_date)

        report(0.4, f"Embedding {len(chunks)} chunks…")
        vectors = []
        batch = 64
        for i in range(0, len(chunks), batch):
            vectors.extend(self.embedder.embed_documents([c.text for c in chunks[i : i + batch]]))
            report(0.4 + 0.5 * min(1.0, (i + batch) / max(1, len(chunks))), "Embedding…")

        report(0.92, "Saving to vector database…")
        import numpy as np

        self.store.add([c.chunk_id for c in chunks], np.vstack(vectors), [c.text for c in chunks],
                       [c.metadata for c in chunks])

        stored_path = self.settings.uploads_dir / f"{doc_id}_{Path(doc.file_name).name}"
        if isinstance(source, (bytes, bytearray)):
            stored_path.write_bytes(source)
        elif Path(source).resolve() != stored_path.resolve():
            stored_path.write_bytes(Path(source).read_bytes())

        self.memory.add_document(
            doc_id=doc_id,
            file_name=doc.file_name,
            file_type=doc.file_type,
            category=category,
            content_hash=doc.content_hash,
            num_pages=doc.num_pages,
            num_chunks=len(chunks),
            word_count=doc.word_count,
            size_bytes=doc.metadata.get("size_bytes", 0),
            stored_path=str(stored_path),
            upload_date=upload_date,
            embedder=self.embedder.name,
        )
        report(1.0, "Done")
        return IngestResult(doc_id, doc.file_name, doc.num_pages, len(chunks))

    def delete_document(self, doc_id: str) -> None:
        self.store.delete({"doc_id": doc_id})
        doc = self.memory.get_document(doc_id)
        if doc and doc.get("stored_path"):
            Path(doc["stored_path"]).unlink(missing_ok=True)
        self.memory.delete_document(doc_id)

    def set_category(self, doc_id: str, category: str) -> None:
        self.store.update_metadata({"doc_id": doc_id}, {"category": category})
        self.memory.update_document_category(doc_id, category)

    def reindex_all(self, progress: Optional[Callable[[float, str], None]] = None) -> int:
        """Re-embed every stored upload (use after switching embedding model)."""
        docs = self.memory.list_documents()
        self.store.reset()
        for i, d in enumerate(docs):
            path = Path(d["stored_path"] or "")
            if path.exists():
                self.memory.delete_document(d["doc_id"])
                self.ingest_file(path.read_bytes(), d["file_name"], d["category"])
            if progress:
                progress((i + 1) / max(1, len(docs)), d["file_name"])
        return len(docs)

    # ================================================================ search
    def search(self, query: str, k: Optional[int] = None, doc_ids: Optional[Sequence[str]] = None,
               category: Optional[str] = None, diversify: bool = False) -> List[RetrievedChunk]:
        return self.retriever.retrieve(query, k or self.settings.top_k, doc_ids, category, diversify)

    # ================================================================ Q&A
    def _history_text(self, cid: Optional[str]) -> str:
        if not cid:
            return "(none)"
        msgs = self.memory.get_messages(cid, limit=self.settings.history_turns * 2)
        if not msgs:
            return "(none)"
        lines = []
        for m in msgs:
            content = re.sub(r"\s+", " ", m["content"])[:600]
            lines.append(f"{m['role'].title()}: {content}")
        return "\n".join(lines)

    def _condense(self, question: str, history: str) -> str:
        if history == "(none)" or self.llm.is_offline:
            return question
        try:
            q = self.llm.generate(prompts.CONDENSE_PROMPT.format(history=history, question=question),
                                  temperature=0, max_tokens=100)
            return q.strip().strip('"') or question
        except LLMError:
            return question

    def answer(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        k: Optional[int] = None,
        doc_ids: Optional[Sequence[str]] = None,
        category: Optional[str] = None,
        diversify: Optional[bool] = None,
        intent: str = "qa",
    ) -> AssistantResponse:
        """Core RAG question answering with citations (qa / compare intents)."""
        t0 = time.time()
        history = self._history_text(conversation_id)
        search_query = self._condense(question, history)
        if diversify is None:
            diversify = intent == "compare"
        chunks = self.search(search_query, k or self.settings.top_k, doc_ids, category, diversify)
        llm_error = ""
        sources = build_sources(chunks)
        context = build_context(chunks)

        if not chunks:
            text = ("I couldn't find anything about that in your library. "
                    "Add a document that covers it in **My Library**, or remove any document filters.")
        elif self.llm.is_offline:
            text = extractive_answer(question, chunks)
        else:
            system = prompts.SYSTEM_BASE.format(preferences=self.memory.preferences_prompt())
            template = prompts.COMPARE_PROMPT if intent == "compare" else prompts.QA_PROMPT
            prompt = template.format(history=history, context=context, question=question)
            try:
                text = self.llm.generate(prompt, system=system)
            except LLMError as e:
                llm_error = str(e)
                text = extractive_answer(question, chunks, note=BUSY_NOTE)
            text = strip_invalid_citations(text, len(sources))

        cited = cited_sources(text, sources) if chunks else []
        metrics = {
            "latency_s": round(time.time() - t0, 2),
            "n_retrieved": len(chunks),
            "citation_coverage": round(citation_coverage(text), 3) if chunks else 0.0,
            "hallucination_rate": round(hallucination_rate(text, context), 3) if chunks else 0.0,
            "top_score": round(chunks[0].score, 3) if chunks else 0.0,
        }
        if llm_error:
            metrics["llm_error"] = llm_error[:600]
        return AssistantResponse(text, sources, cited, chunks, intent, search_query, metrics)

    def ask(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        mode: str = "auto",
        **kwargs,
    ) -> AssistantResponse:
        """Chat entry point: routes to the right agent, then stores the turn in conversation memory."""
        from .agents import AgentOrchestrator  # local import avoids a cycle

        intent = detect_intent(question) if mode == "auto" else mode
        orchestrator = AgentOrchestrator(self)
        resp = orchestrator.handle(question, intent, conversation_id, **kwargs)

        if conversation_id:
            if not self.memory.get_messages(conversation_id, limit=1):
                self.memory.rename_conversation(conversation_id, question[:60])
            self.memory.add_message(conversation_id, "user", question)
            self.memory.add_message(
                conversation_id, "assistant", resp.answer, [s.to_dict() for s in resp.cited],
                {"intent": resp.intent, "metrics": resp.metrics, "extra": resp.extra,
                 "retrieved": [r.to_dict() for r in resp.retrieved]},
            )
        self.memory.record_topics(question)
        return resp
