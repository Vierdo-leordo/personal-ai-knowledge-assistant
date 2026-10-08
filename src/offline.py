"""Offline (no-API-key) generation: extractive answers, summaries, notes, flashcards and quizzes.

These heuristics keep the whole app working without an LLM, which is useful for demos, tests and
for showing what the retrieval layer alone achieves. With an API key the LLM versions are used instead.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from .chunker import is_heading_line
from .embeddings import _stem, query_terms, tokenize
from .retriever import RetrievedChunk

_DEF_PATTERNS = [
    # "A stack is a linear data structure…", "Overfitting is when…"
    re.compile(
        r"^(?:An?\s+|The\s+)?(?P<term>[A-Za-z][\w\s\-()/']{1,60}?)\s+"
        r"(?:is|are|refers to|means|is defined as|can be defined as)\s+"
        r"(?P<defn>(?:a|an|the|when|one|any|called)\b.+)$"
    ),
    # "Paging: a memory-management scheme…"
    re.compile(r"^(?P<term>[A-Z][\w\-()/]*(?:\s[\w\-()/]+){0,3})\s*[:–—]\s+(?P<defn>.{20,})$"),
]


_FORM = re.compile(r"\b(name|registration|reg\.? ?no|roll ?no|branch|section|faculty|date|course code|semester|"
                   r"marks|signature|page no|academic year|lecture \d+|unit \d+|week)\b\s*[:.\-]?", re.I)
_BAD_TERMS = {"name", "product name", "what", "which", "who", "lecture", "page", "date", "title", "note", "example",
              "output", "input", "result", "aim", "program", "code", "answer", "question", "marks"}


def looks_like_form(text: str) -> bool:
    """Administrative / form-field text that makes no sense as study material."""
    return text.count(":") >= 2 or len(_FORM.findall(text)) >= 2 or text.rstrip().endswith((":", "?"))


def sentences_with_source(chunks: Sequence[RetrievedChunk]) -> List[Tuple[str, int]]:
    """(sentence, passage number) pairs; headings dropped, duplicates (from chunk overlap) removed."""
    out, seen = [], set()
    for i, c in enumerate(chunks, start=1):
        for line in c.text.split("\n"):
            if is_heading_line(line):
                continue
            for s in re.split(r"(?<=[.!?])\s+", line):
                s = s.strip()
                key = s.lower()
                if 6 <= len(s.split()) <= 60 and key not in seen and not looks_like_form(s):
                    seen.add(key)
                    out.append((s, i))
    return out


def _score(sentence: str, terms: set) -> float:
    toks = {_stem(t) for t in tokenize(sentence)}
    if not toks:
        return 0.0
    overlap = len(toks & terms)
    return overlap + 0.5 * overlap / len(toks)


def extractive_answer(question: str, chunks: Sequence[RetrievedChunk], max_sentences: int = 5,
                      note: str = "> No AI key is set up, so these are the most relevant passages from your documents.\n\n") -> str:
    if not chunks:
        return "I couldn't find anything relevant to that in your documents. Try uploading related material or rephrasing."
    q = {_stem(t) for t in query_terms(question)}
    scored = sorted(sentences_with_source(chunks), key=lambda p: _score(p[0], q), reverse=True)
    best = _score(scored[0][0], q) if scored else 0
    picked = [p for p in scored if best > 0 and _score(p[0], q) >= 0.5 * best][:max_sentences] or scored[:3]
    # keep original reading order within each source
    order = {s: i for i, s in enumerate(sentences_with_source(chunks))}
    picked.sort(key=lambda p: order.get(p, 0))
    lines = [f"- {s} [{n}]" for s, n in picked]
    return note + "\n".join(lines)


def key_terms(text: str, n: int = 10) -> List[str]:
    words = [t for t in tokenize(text) if len(t) > 3 and not t.isdigit()]
    return [w for w, _ in Counter(words).most_common(n)]


def extractive_summary(chunks: Sequence[RetrievedChunk], file_name: str, n: int = 6) -> str:
    text = " ".join(c.text for c in chunks)
    terms = {_stem(t) for t in key_terms(text, 25)}
    sents = sentences_with_source(chunks)
    ranked = sorted(sents, key=lambda p: _score(p[0], terms), reverse=True)
    top = ranked[:n]
    order = {s: i for i, s in enumerate(sents)}
    top.sort(key=lambda p: order.get(p, 0))
    findings = ranked[n : n + 4]
    body = "\n".join(f"{s} [{k}]" for s, k in top)
    kf = "\n".join(f"- {s} [{k}]" for s, k in findings) or "- (not enough content)"
    kt = ", ".join(key_terms(text, 8))
    return (
        f"## Summary\n{body}\n\n## Key Findings\n{kf}\n\n## Recommendations\n"
        f"- Review the key terms: {kt}.\n- Generate flashcards for this document to test recall.\n"
        f"- Ask follow-up questions about sections that are unclear.\n\n"
        f"_Built from the key sentences of {file_name}. Set up an AI key in Settings for a written summary._"
    )


def find_definitions(chunks: Sequence[RetrievedChunk]) -> List[Dict]:
    defs, seen = [], set()
    for s, src in sentences_with_source(chunks):
        for pat in _DEF_PATTERNS:
            m = pat.match(s)
            if m:
                term = re.sub(r"^(?:an?|the)\s+", "", m.group("term").strip(), flags=re.I)
                term = term[0].upper() + term[1:] if term else term
                defn_text = m.group("defn").strip()
                if (term.lower() in _BAD_TERMS or len(term) < 3 or re.search(r"\d", term)
                        or len(defn_text.split()) < 5 or looks_like_form(defn_text)):
                    continue
                if not term or len(term.split()) > 6 or term.isupper() or term.lower() in {"it", "this", "that", "there", "they", "these"}:
                    continue
                if term.lower() in seen:
                    break
                seen.add(term.lower())
                defn = m.group("defn").strip().rstrip(".")
                defs.append({"term": term, "definition": defn[0].upper() + defn[1:], "source": src, "sentence": s})
                break
    return defs


def heuristic_flashcards(chunks: Sequence[RetrievedChunk], n: int = 10) -> List[Dict]:
    cards = [
        {"question": f"What is {d['term']}?", "answer": d["definition"] + ".", "source": d["source"]}
        for d in find_definitions(chunks)
    ]
    if len(cards) < n:  # cloze-deletion cards from key sentences
        text = " ".join(c.text for c in chunks)
        terms = key_terms(text, 30)
        used = {c["question"] for c in cards}
        for s, src in sentences_with_source(chunks):
            if len(cards) >= n:
                break
            for t in terms:
                if re.search(rf"\b{re.escape(t)}\b", s, re.I):
                    q = re.sub(rf"\b{re.escape(t)}\b", "_____", s, count=1, flags=re.I)
                    if q not in used:
                        cards.append({"question": f"Fill in the blank: {q}", "answer": t, "source": src})
                        used.add(q)
                    break
    return cards[:n]


def heuristic_quiz(chunks: Sequence[RetrievedChunk], n_mcq: int = 5, n_short: int = 3, seed: int = 7) -> Dict:
    rng = random.Random(seed)
    defs = find_definitions(chunks)
    mcq = []
    for d in defs[:n_mcq]:
        distractors = [x["definition"] for x in defs if x is not d]
        rng.shuffle(distractors)
        if len(distractors) < 3:
            continue
        options = [d["definition"]] + distractors[:3]
        rng.shuffle(options)
        mcq.append(
            {
                "question": f"Which statement best describes {d['term']}?",
                "options": [o[:220] for o in options],
                "answer_index": options.index(d["definition"]),
                "explanation": d["sentence"],
                "source": d["source"],
            }
        )
    # term-matching fallback when there are too few definitions
    if len(mcq) < n_mcq:
        text = " ".join(c.text for c in chunks)
        terms = key_terms(text, 40)
        for s, src in sentences_with_source(chunks):
            if len(mcq) >= n_mcq:
                break
            hit = next((t for t in terms if re.search(rf"\b{re.escape(t)}\b", s, re.I)), None)
            if not hit:
                continue
            wrong = [t for t in terms if t != hit and not re.search(rf"\b{re.escape(t)}\b", s, re.I)]
            if len(wrong) < 3:
                continue
            options = [hit] + rng.sample(wrong[:12], 3)
            rng.shuffle(options)
            mcq.append(
                {
                    "question": "Which word completes the statement? "
                    + re.sub(rf"\b{re.escape(hit)}\b", "_____", s, count=1, flags=re.I),
                    "options": options,
                    "answer_index": options.index(hit),
                    "explanation": s,
                    "source": src,
                }
            )
    short = [
        {"question": f"Explain {d['term']} in your own words.", "model_answer": d["definition"] + ".", "source": d["source"]}
        for d in defs[n_mcq : n_mcq + n_short]
    ]
    if len(short) < n_short:
        for t in key_terms(" ".join(c.text for c in chunks), n_short * 2):
            if len(short) >= n_short:
                break
            src_sent = next(((s, k) for s, k in sentences_with_source(chunks) if t in s.lower()), None)
            if src_sent:
                short.append({"question": f"What role does '{t}' play in this material?", "model_answer": src_sent[0], "source": src_sent[1]})
    return {"mcq": mcq[:n_mcq], "short": short[:n_short]}


def heuristic_notes(chunks: Sequence[RetrievedChunk], style: str, topic: str) -> str:
    by_file: Dict[str, List[Tuple[int, RetrievedChunk]]] = {}
    for i, c in enumerate(chunks, start=1):
        by_file.setdefault(c.file_name, []).append((i, c))
    out = [f"# {style}: {topic or 'All material'}", "_Built from key sentences in your documents. Set up an AI key in Settings for fully written notes._", ""]
    defs = find_definitions(chunks)
    if defs:
        out.append("## Key Definitions")
        out += [f"- **{d['term']}**: {d['definition']} [{d['source']}]" for d in defs[:12]]
        out.append("")
    for fname, items in by_file.items():
        out.append(f"## From {fname}")
        for i, c in items[:8]:
            sents = [x for x, _ in sentences_with_source([c])][:2]
            if not sents:
                continue
            label = f"**{c.metadata.get('heading')}** — " if c.metadata.get("heading") else ""
            out.append(f"- {label}{' '.join(sents)} [{i}]")
        out.append("")
    out.append("## Key Terms")
    out.append(", ".join(key_terms(" ".join(c.text for c in chunks), 15)))
    return "\n".join(out)


def heuristic_topics(chunks: Sequence[RetrievedChunk], n: int = 6) -> List[str]:
    headings = [c.metadata.get("heading") for c in chunks if c.metadata.get("heading")]
    topics = list(dict.fromkeys(h for h in headings if h))
    topics += [d["term"] for d in find_definitions(chunks)]
    topics += key_terms(" ".join(c.text for c in chunks), n)
    return list(dict.fromkeys(topics))[:n]


def heuristic_revision_plan(topics: List[str], days: int) -> str:
    days = max(1, days)
    lines = [f"# {days}-Day Revision Plan", ""]
    per_day = max(1, -(-len(topics) // max(1, days - 1))) if days > 1 else len(topics)
    idx = 0
    for d in range(1, days + 1):
        if d == days and days > 1:
            lines += [f"## Day {d}: Full review", "- Retake the practice quiz and review every flashcard you missed.",
                      "- Skim the exam summary; sleep early.", ""]
            break
        todays = topics[idx : idx + per_day] or topics[:1]
        idx += per_day
        lines += [f"## Day {d}: {', '.join(todays)}", "- Read the summary and chapter notes for these topics.",
                  "- Do the flashcards for these topics twice (morning/evening).",
                  "- Answer 3 practice questions without looking at notes.", "- Checkpoint: explain each topic aloud in 2 minutes.", ""]
    lines += ["## Exam-day tips", "- Read all questions first; answer the ones you know best.", "- Leave 10 minutes to review."]
    return "\n".join(lines)
