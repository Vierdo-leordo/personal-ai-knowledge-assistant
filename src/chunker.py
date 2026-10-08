"""Chunking Engine: split documents into overlapping, sentence-aware chunks with metadata."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

from .document_loader import LoadedDocument

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict = field(default_factory=dict)


def is_heading_line(line: str, first_in_para: bool = True, next_line: str = "") -> bool:
    """Markdown headings, or a short title-like first line (e.g. "Chapter 2: Process Scheduling").

    Kept conservative so wrapped PDF lines are not mistaken for headings.
    """
    line = line.strip()
    if line.startswith("#"):
        return True
    if "|" in line or not first_in_para or not line or re.search(r"[.!?,;:]$", line):
        return False
    if len(line) > 60 or len(line.split()) > 8 or not (line[0].isupper() or line[0].isdigit()):
        return False
    return not next_line or next_line.strip()[:1].isupper()


def split_sentences(text: str) -> List[str]:
    """Sentence units; "\n" marks a paragraph break. Heading lines become their own unit."""
    sentences: List[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        lines = [ln for ln in para.split("\n") if ln.strip()]
        body: List[str] = []
        for i, line in enumerate(lines):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if is_heading_line(line, first_in_para=(i == 0 or line.lstrip().startswith("#")), next_line=nxt):
                sentences.append("\n")
                sentences.append(line.strip().lstrip("#").strip())
                sentences.append("\n")
            else:
                body.append(line.strip())
        if body:
            sentences.extend(p.strip() for p in _SENTENCE_SPLIT.split(" ".join(body)) if p.strip())
        sentences.append("\n")  # paragraph marker – preferred split point
    return sentences


def _hard_wrap(sentence: str, size: int) -> List[str]:
    """Split a single overly long sentence on word boundaries."""
    words, out, buf = sentence.split(), [], ""
    for w in words:
        if len(buf) + len(w) + 1 > size and buf:
            out.append(buf)
            buf = w
        else:
            buf = f"{buf} {w}".strip()
    if buf:
        out.append(buf)
    return out


def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> List[str]:
    """Greedy sentence packing up to ``chunk_size`` characters, carrying ~``chunk_overlap`` chars of context."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    units: List[str] = []
    for s in split_sentences(text):
        units.extend(_hard_wrap(s, chunk_size) if len(s) > chunk_size else [s])

    chunks: List[str] = []
    current: List[str] = []
    length = 0

    def flush():
        body = ""
        for u in current:
            if u == "\n":
                body = body.rstrip(" ") + "\n"
            else:
                body += ("" if not body or body.endswith("\n") else " ") + u
        body = re.sub(r"[ \t]+", " ", body)
        body = re.sub(r"\n\s*\n+", "\n", body).strip()
        if body:
            chunks.append(body)

    for unit in units:
        if unit == "\n":
            current.append(unit)
            continue
        if is_heading_line(unit) and length > chunk_size * 0.4:
            flush()  # start a fresh chunk at a section heading (no overlap across sections)
            current, length = [], 0
        elif length + len(unit) + 1 > chunk_size and length > 0:
            flush()
            # overlap: keep trailing sentences up to chunk_overlap characters
            overlap: List[str] = []
            olen = 0
            for u in reversed(current):
                if u == "\n":
                    continue
                if olen + len(u) > chunk_overlap:
                    break
                overlap.insert(0, u)
                olen += len(u) + 1
            current, length = overlap, olen
        current.append(unit)
        length += len(unit) + 1
    flush()

    # merge a tiny trailing chunk into its predecessor
    if len(chunks) > 1 and len(chunks[-1]) < chunk_size * 0.15:
        chunks[-2] = f"{chunks[-2]}\n{chunks[-1]}"
        chunks.pop()
    return chunks


def make_doc_id(content_hash: str) -> str:
    return content_hash[:16]


def chunk_document(
    doc: LoadedDocument,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    category: str = "General",
    upload_date: str | None = None,
) -> List[Chunk]:
    """Chunk every page of a document. Metadata: file name, page number, chunk id, upload date, category."""
    upload_date = upload_date or datetime.now(timezone.utc).isoformat(timespec="seconds")
    doc_id = make_doc_id(doc.content_hash)
    chunks: List[Chunk] = []
    seen: set[str] = set()
    idx = 0
    section = ""
    for page in doc.pages:
        section = page.heading or section
        for piece in chunk_text(page.text, chunk_size, chunk_overlap):
            # label = first heading inside the chunk (else the section carried over from earlier chunks)
            heads = [ln.strip().lstrip("#").strip() for ln in piece.split("\n")[:-1] if is_heading_line(ln)]
            label = heads[0] if heads else section
            section = heads[-1] if heads else section
            key = hashlib.md5(piece.lower().encode()).hexdigest()
            if key in seen:  # duplicate chunk inside the same document
                continue
            seen.add(key)
            chunk_id = f"{doc_id}-{idx:05d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=piece,
                    metadata={
                        "doc_id": doc_id,
                        "file_name": doc.file_name,
                        "file_type": doc.file_type,
                        "page": int(page.page_number),
                        "heading": label,
                        "chunk_id": chunk_id,
                        "chunk_index": idx,
                        "upload_date": upload_date,
                        "category": category,
                    },
                )
            )
            idx += 1
    return chunks
