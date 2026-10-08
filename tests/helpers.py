"""Shared test helpers: an isolated assistant, a fake LLM and a tiny PDF generator."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from src.chat import KnowledgeAssistant
from src.config import get_settings
from src.embeddings import HashingEmbedder
from src.llm_engine import LLMEngine
from src.vector_store import NumpyVectorStore

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "sample_data"


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal, valid multi-page PDF containing the given text (one string per page)."""
    objs: list[bytes] = []
    n = len(pages)
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(n))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    font_id = 3 + n * 2
    for i, text in enumerate(pages):
        lines = [text[j : j + 80] for j in range(0, len(text), 80)]
        ops = ["BT", "/F1 11 Tf", "14 TL", "50 780 Td"]
        for ln in lines:
            safe = ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({safe}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode()
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Contents {4 + i * 2} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


class FakeLLM(LLMEngine):
    """Deterministic stand-in for a real LLM: exercises the 'online' code paths without network access."""

    def __init__(self, settings):
        super().__init__(settings, provider="gemini")
        self.calls: list[str] = []

    def generate(self, prompt, system="", temperature=None, max_tokens=2048, json_mode=False, retries=2):
        self.calls.append(prompt)
        if "flashcards" in prompt and "Return ONLY valid JSON" in prompt:
            return json.dumps([{"question": "What is overfitting?", "answer": "Learning noise instead of patterns.",
                                "source": 1}])
        if '"mcq"' in prompt:
            return "```json\n" + json.dumps({"mcq": [{"question": "Overfitting means?", "options": ["noise", "bias",
                                              "speed", "none"], "answer_index": "0", "explanation": "x", "source": 1}],
                                             "short": [{"question": "Define PCA", "model_answer": "dimensionality "
                                                        "reduction", "source": 2}]}) + "\n```"
        if "most important distinct topics" in prompt:
            return '["Processes", "Scheduling", "Deadlocks"]'
        if "Grade the student's answer" in prompt:
            return '{"score": 0.8, "feedback": "good"}'
        if "evaluating a RAG system" in prompt:
            return '{"score": 4, "reason": "ok"}'
        if "standalone search query" in prompt:
            return "overfitting in machine learning"
        return "Overfitting is when a model learns noise in the training data [1]. It is reduced by regularization [1][9]."


class TempAssistantMixin:
    """unittest mixin: creates an isolated assistant (temp dirs, hashing embedder, NumPy store)."""

    fake_llm = False

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        settings = get_settings(
            data_dir=self.tmp / "data", vector_dir=self.tmp / "vs", outputs_dir=self.tmp / "out",
            llm_provider="offline", embedding_provider="hashing", vector_store="numpy",
        )
        emb = HashingEmbedder()
        store = NumpyVectorStore(settings.vector_dir, "kb_test")
        llm = FakeLLM(settings) if self.fake_llm else LLMEngine(settings, "offline")
        self.a = KnowledgeAssistant(settings, emb, store, llm)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def load_samples(self):
        for f in ("machine_learning_notes.md", "operating_systems.txt", "data_structures_notes.docx"):
            self.a.ingest_file(SAMPLES / f, category="Education")
