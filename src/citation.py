"""Citation Engine: number retrieved passages, build the LLM context, and map [n] markers back to sources."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .embeddings import _stem, tokenize
from .retriever import RetrievedChunk

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Source:
    number: int
    file_name: str
    page: int
    chunk_id: str
    score: float
    excerpt: str
    heading: str = ""

    def label(self) -> str:
        loc = f"Page {self.page}" if self.page else ""
        if self.heading:
            loc = f"{loc} · {self.heading}" if loc else self.heading
        return f"{self.file_name}" + (f", {loc}" if loc else "")

    def to_dict(self) -> Dict:
        return {
            "number": self.number,
            "file_name": self.file_name,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "excerpt": self.excerpt,
            "heading": self.heading,
        }


def build_sources(chunks: Sequence[RetrievedChunk]) -> List[Source]:
    return [
        Source(
            number=i,
            file_name=c.file_name,
            page=c.page,
            chunk_id=c.chunk_id,
            score=c.score,
            excerpt=c.text[:400] + ("…" if len(c.text) > 400 else ""),
            heading=c.metadata.get("heading", ""),
        )
        for i, c in enumerate(chunks, start=1)
    ]


def build_context(chunks: Sequence[RetrievedChunk], max_chars: int = 12000) -> str:
    """Render chunks as numbered passages the LLM can cite as [1], [2], …"""
    parts, used = [], 0
    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] Source: {c.file_name}" + (f", page {c.page}" if c.page else "")
        block = f"{header}\n{c.text}"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts)


def extract_citation_numbers(answer: str) -> List[int]:
    nums: List[int] = []
    for m in _CITE.finditer(answer):
        for n in m.group(1).split(","):
            n = int(n.strip())
            if n not in nums:
                nums.append(n)
    return nums


def cited_sources(answer: str, sources: Sequence[Source]) -> List[Source]:
    """Sources actually referenced by the answer (falls back to all sources if none were marked)."""
    nums = extract_citation_numbers(answer)
    by_num = {s.number: s for s in sources}
    picked = [by_num[n] for n in nums if n in by_num]
    return picked or list(sources)


def format_sources_markdown(sources: Sequence[Source]) -> str:
    if not sources:
        return ""
    lines = ["**Sources:**"]
    seen = set()
    for s in sources:
        key = (s.file_name, s.page)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- [{s.number}] {s.label()}")
    return "\n".join(lines)


def strip_invalid_citations(answer: str, n_sources: int) -> str:
    """Remove citation markers that point to non-existent sources (a common hallucination)."""

    def fix(m):
        nums = [n.strip() for n in m.group(1).split(",") if 1 <= int(n.strip()) <= n_sources]
        return f"[{', '.join(nums)}]" if nums else ""

    return _CITE.sub(fix, answer)


# ---------------------------------------------------------------- evaluation helpers
def _sentences(text: str) -> List[str]:
    text = re.sub(r"(?m)^\s*>.*$", "", text)  # quoted notes from the app are not claims
    text = re.sub(r"(?m)^\s*[-*#]+\s*", "", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip().split()) >= 5]


def citation_coverage(answer: str) -> float:
    """Share of substantive sentences that carry at least one [n] citation."""
    sents = _sentences(answer)
    if not sents:
        return 0.0
    return sum(1 for s in sents if _CITE.search(s)) / len(sents)


def unsupported_sentences(answer: str, context: str, threshold: float = 0.5) -> List[str]:
    """Sentences whose content words are mostly absent from the retrieved context (hallucination proxy)."""
    ctx = {_stem(t) for t in tokenize(context)}
    bad = []
    for s in _sentences(_CITE.sub("", answer)):
        toks = {_stem(t) for t in tokenize(s)}
        if len(toks) < 3:
            continue
        support = len(toks & ctx) / len(toks)
        if support < threshold:
            bad.append(s)
    return bad


def hallucination_rate(answer: str, context: str) -> float:
    sents = _sentences(_CITE.sub("", answer))
    if not sents:
        return 0.0
    return len(unsupported_sentences(answer, context)) / len(sents)
