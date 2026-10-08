"""Agent-based workflow.

* RetrievalAgent       – finds the right content (topic search, whole-document, multi-document)
* SummarizationAgent   – document / topic summaries (map-reduce for long documents)
* LearningAgent        – study notes, flashcards, quizzes, answer grading
* ExamPrepAgent        – the Autonomous Learning Assistant ("Prepare me for my DBMS exam")
* AgentOrchestrator    – routes a chat request to the right agent
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from . import offline, prompts
from .embeddings import query_terms, tokenize
from .citation import build_context, build_sources, cited_sources, strip_invalid_citations
from .llm_engine import LLMError
from .retriever import RetrievedChunk

if TYPE_CHECKING:  # pragma: no cover
    from .chat import AssistantResponse, KnowledgeAssistant


def _even_sample(items: List, n: int) -> List:
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


class RetrievalAgent:
    def __init__(self, assistant: "KnowledgeAssistant"):
        self.a = assistant

    def gather(self, topic: str = "", doc_ids: Optional[Sequence[str]] = None, k: int = 12,
               category: Optional[str] = None) -> List[RetrievedChunk]:
        """Topic given -> semantic search (diversified across files); else an even sample of the documents."""
        if topic.strip():
            if not doc_ids:
                named = self.find_document_in_text(topic)
                doc_ids = [named] if named else None
            hits = self.a.search(topic, k=k, doc_ids=doc_ids, category=category)
            if hits:  # drop weak matches so unrelated files don't leak into study material
                top = hits[0].score
                hits = [h for h in hits if h.score >= 0.45 * top]
            if len(hits) < 4 and doc_ids:  # topic is the whole document -> use it all
                hits = hits + [c for d in doc_ids for c in _even_sample(self.a.retriever.get_document_chunks(d), k)
                               if c.chunk_id not in {h.chunk_id for h in hits}]
            return hits[:k]
        chunks: List[RetrievedChunk] = []
        ids = list(doc_ids) if doc_ids else [d["doc_id"] for d in self.a.memory.list_documents(category)]
        if not ids:
            return []
        per_doc = max(2, k // len(ids))
        for d in ids:
            chunks.extend(_even_sample(self.a.retriever.get_document_chunks(d), per_doc))
        return chunks[: max(k, per_doc)]

    def find_document_in_text(self, text: str) -> Optional[str]:
        """Resolve a document mentioned by name in a request ("summarize ResearchPaper.pdf")."""
        low = text.lower()
        best, best_len = None, 0
        for d in self.a.memory.list_documents():
            name = d["file_name"].lower()
            stem = re.sub(r"\.[a-z0-9]+$", "", name)
            for cand in (name, stem, stem.replace("_", " "), stem.replace("-", " ")):
                if len(cand) > 3 and cand in low and len(cand) > best_len:
                    best, best_len = d["doc_id"], len(cand)
            # "data structures" -> data_structures_notes.docx (all topic words appear in the file name)
            words = set(query_terms(text))
            name_words = set(tokenize(re.sub(r"[_\-.]", " ", stem)))
            if len(words) >= 2 and words <= name_words and len(" ".join(words)) > best_len:
                best, best_len = d["doc_id"], len(" ".join(words))
        return best


class SummarizationAgent:
    MAX_CONTEXT = 24000

    def __init__(self, assistant: "KnowledgeAssistant"):
        self.a = assistant
        self.retrieval = RetrievalAgent(assistant)

    def summarize_document(self, doc_id: str, length: str = "concise") -> Dict:
        doc = self.a.memory.get_document(doc_id) or {"file_name": doc_id}
        chunks = self.a.retriever.get_document_chunks(doc_id)
        return self._summarize(chunks, doc["file_name"], length)

    def summarize_topic(self, topic: str, doc_ids: Optional[Sequence[str]] = None, length: str = "concise") -> Dict:
        chunks = self.retrieval.gather(topic, doc_ids, k=12)
        return self._summarize(chunks, f"your documents (topic: {topic})", length)

    def _summarize(self, chunks: List[RetrievedChunk], file_name: str, length: str) -> Dict:
        if not chunks:
            return {"markdown": "Nothing to summarise – no content found.", "sources": []}
        if self.a.llm.is_offline:
            sample = _even_sample(chunks, 20)
            md = offline.extractive_summary(sample, file_name)
            return {"markdown": md, "sources": build_sources(sample)}

        total = sum(len(c.text) for c in chunks)
        if total <= self.MAX_CONTEXT:
            used = chunks
            context = build_context(used, self.MAX_CONTEXT)
        else:  # map-reduce: summarise groups, then summarise the summaries
            groups, buf, size = [], [], 0
            for c in chunks:
                if size + len(c.text) > self.MAX_CONTEXT // 2 and buf:
                    groups.append(buf)
                    buf, size = [], 0
                buf.append(c)
                size += len(c.text)
            if buf:
                groups.append(buf)
            partials = []
            for g in groups[:12]:
                pages = sorted({c.page for c in g if c.page})
                span = f"pages {pages[0]}-{pages[-1]}" if pages else "section"
                try:
                    partials.append(f"({span}) " + self.a.llm.generate(
                        "Summarise the key points of this text in 5-8 bullet points:\n\n"
                        + "\n\n".join(c.text for c in g), max_tokens=700))
                except LLMError as e:
                    partials.append(f"({span}) [error: {e}]")
            used = _even_sample(chunks, 12)
            context = "Partial summaries of the whole document:\n" + "\n\n".join(partials) + \
                      "\n\nRepresentative passages:\n" + build_context(used, 8000)
        try:
            md = self.a.llm.generate(prompts.SUMMARY_PROMPT.format(file_name=file_name, context=context, length=length),
                                     system=prompts.SYSTEM_BASE.format(preferences=""))
        except LLMError as e:
            md = "> The AI service is busy right now, so this summary is built from key sentences.\n\n" + offline.extractive_summary(_even_sample(chunks, 20), file_name)
        sources = build_sources(used)
        md = strip_invalid_citations(md, len(sources))
        return {"markdown": md, "sources": sources}


class LearningAgent:
    def __init__(self, assistant: "KnowledgeAssistant"):
        self.a = assistant
        self.retrieval = RetrievalAgent(assistant)

    # ---------------------------------------------------------------- notes
    def study_notes(self, style: str = "Chapter Notes", topic: str = "",
                    doc_ids: Optional[Sequence[str]] = None) -> Dict:
        chunks = self.retrieval.gather(topic, doc_ids, k=14)
        if not chunks:
            return {"markdown": "No content found for these notes.", "sources": []}
        sources = build_sources(chunks)
        if self.a.llm.is_offline:
            return {"markdown": offline.heuristic_notes(chunks, style, topic), "sources": sources}
        instruction = prompts.NOTES_PROMPTS.get(style, prompts.NOTES_PROMPTS["Chapter Notes"])
        try:
            md = self.a.llm.generate(
                prompts.NOTES_WRAPPER.format(instruction=instruction, topic=topic or "all key topics",
                                             context=build_context(chunks, 20000)),
                system=prompts.SYSTEM_BASE.format(preferences=self.a.memory.preferences_prompt()),
                max_tokens=3000,
            )
        except LLMError as e:
            md = "> The AI service is busy right now, so these notes are built from key sentences.\n\n" + offline.heuristic_notes(chunks, style, topic)
        return {"markdown": strip_invalid_citations(md, len(sources)), "sources": sources}

    # ---------------------------------------------------------------- flashcards
    def flashcards(self, topic: str = "", n: int = 10, doc_ids: Optional[Sequence[str]] = None) -> Dict:
        chunks = self.retrieval.gather(topic, doc_ids, k=12)
        sources = build_sources(chunks)
        cards: List[Dict] = []
        error = ""
        if chunks and not self.a.llm.is_offline:
            prompt = prompts.FLASHCARD_PROMPT.format(n=n, topic=topic or "the key concepts",
                                                     context=build_context(chunks, 14000))
            for attempt in range(2):  # one retry if the reply can't be used
                try:
                    cards = normalize_flashcards(self.a.llm.generate_json(
                        prompt, max_tokens=3500, temperature=0.3 if attempt == 0 else 0.0))
                except LLMError as e:
                    error = str(e)
                if cards:
                    break
        elif chunks:
            cards = offline.heuristic_flashcards(chunks, n)
        if chunks and not cards and not self.a.llm.is_offline:
            error = error or "The AI reply could not be read."
        for c in cards:
            s = next((x for x in sources if x.number == _as_int(c.get("source"))), None)
            c["source_label"] = s.label() if s else ""
        return {"cards": cards[:n], "sources": sources, "error": error if not cards else ""}

    # ---------------------------------------------------------------- quiz
    def quiz(self, topic: str = "", n_mcq: int = 5, n_short: int = 3, difficulty: str = "Medium",
             doc_ids: Optional[Sequence[str]] = None) -> Dict:
        chunks = self.retrieval.gather(topic, doc_ids, k=12)
        sources = build_sources(chunks)
        mcq: List[Dict] = []
        short: List[Dict] = []
        error = ""
        if chunks and not self.a.llm.is_offline:
            prompt = prompts.QUIZ_PROMPT.format(topic=topic or "the key concepts", n_mcq=n_mcq, n_short=n_short,
                                                difficulty=difficulty, context=build_context(chunks, 14000))
            for attempt in range(2):  # one retry if the reply can't be used
                try:
                    mcq, short = normalize_quiz(self.a.llm.generate_json(
                        prompt, max_tokens=4000, temperature=0.3 if attempt == 0 else 0.0))
                except LLMError as e:
                    error = str(e)
                if mcq or short:
                    break
        elif chunks:
            fallback = offline.heuristic_quiz(chunks, n_mcq, n_short)
            mcq, short = fallback["mcq"], fallback["short"]
        if chunks and not (mcq or short) and not self.a.llm.is_offline:
            error = error or "The AI reply could not be read."
        for q in mcq + short:
            s = next((x for x in sources if x.number == _as_int(q.get("source"))), None)
            q["source_label"] = s.label() if s else ""
        return {"mcq": mcq[:n_mcq], "short": short[:n_short], "sources": sources,
                "error": error if not (mcq or short) else ""}

    def grade_short_answer(self, question: str, model_answer: str, student_answer: str) -> Dict:
        if not student_answer.strip():
            return {"score": 0.0, "feedback": "No answer given."}
        if not self.a.llm.is_offline:
            try:
                res = self.a.llm.generate_json(prompts.GRADE_PROMPT.format(
                    question=question, model_answer=model_answer, student_answer=student_answer), temperature=0)
                return {"score": float(res.get("score", 0)), "feedback": str(res.get("feedback", ""))}
            except (LLMError, ValueError, AttributeError):
                pass
        from .retriever import keyword_overlap

        score = round(keyword_overlap(model_answer, student_answer), 2)
        fb = "Good coverage of the key points." if score >= 0.6 else "Some key points are missing – compare with the model answer."
        return {"score": score, "feedback": fb + " (keyword-based grading)"}


class ExamPrepAgent:
    """Autonomous Learning Assistant: goal -> topics -> summary, quiz, flashcards, revision plan."""

    def __init__(self, assistant: "KnowledgeAssistant"):
        self.a = assistant
        self.retrieval = RetrievalAgent(assistant)
        self.summarizer = SummarizationAgent(assistant)
        self.learner = LearningAgent(assistant)

    @staticmethod
    def subject_from_goal(goal: str) -> str:
        m = re.search(r"\bfor (?:my |the |an? )?(.+?)(?: exam| test| interview| quiz|$)", goal, re.I)
        subject = m.group(1) if m else goal
        return re.sub(r"\b(prepare|me|help|please)\b", "", subject, flags=re.I).strip(" .?!") or goal

    def topics(self, chunks: List[RetrievedChunk], n: int = 6) -> List[str]:
        if not self.a.llm.is_offline and chunks:
            try:
                data = self.a.llm.generate_json(prompts.TOPICS_PROMPT.format(n=n, context=build_context(chunks, 12000)))
                if isinstance(data, list) and data:
                    return [str(t) for t in data][:n]
            except LLMError:
                pass
        return offline.heuristic_topics(chunks, n)

    def prepare(self, goal: str, days: int = 5, doc_ids: Optional[Sequence[str]] = None,
                progress=None) -> Dict:
        def step(p, msg):
            if progress:
                progress(p, msg)

        subject = self.subject_from_goal(goal)
        step(0.1, "Retrieval agent: finding relevant material…")
        chunks = self.retrieval.gather(subject, doc_ids, k=12)
        if not chunks:
            return {"subject": subject, "error": "No relevant material found in your knowledge base."}
        step(0.25, "Identifying key topics…")
        topics = self.topics(chunks)
        step(0.4, "Summarization agent: writing the summary…")
        summary = self.summarizer._summarize(chunks, f"your documents (topic: {subject})", "concise")
        step(0.6, "Learning agent: building the quiz…")
        quiz = self.learner.quiz(subject, 5, 3, "Medium", doc_ids)
        step(0.75, "Learning agent: making flashcards…")
        cards = self.learner.flashcards(subject, 10, doc_ids)
        step(0.9, "Planning the revision schedule…")
        sources = build_sources(chunks)
        if self.a.llm.is_offline:
            plan = offline.heuristic_revision_plan(topics, days)
        else:
            try:
                plan = self.a.llm.generate(prompts.REVISION_PLAN_PROMPT.format(
                    goal=goal, days=days, topics="\n".join(f"- {t}" for t in topics),
                    context=build_context(chunks, 10000)), max_tokens=2500)
                plan = strip_invalid_citations(plan, len(sources))
            except LLMError:
                plan = offline.heuristic_revision_plan(topics, days)
        step(1.0, "Done")
        return {"subject": subject, "topics": topics, "summary": summary, "quiz": quiz, "flashcards": cards,
                "plan": plan, "sources": sources}


class AgentOrchestrator:
    """Routes a chat message to the agent that should handle it."""

    def __init__(self, assistant: "KnowledgeAssistant"):
        self.a = assistant
        self.retrieval = RetrievalAgent(assistant)

    def handle(self, question: str, intent: str, conversation_id: Optional[str] = None, **kw) -> "AssistantResponse":
        from .chat import AssistantResponse

        doc_ids = kw.get("doc_ids")
        mentioned = self.retrieval.find_document_in_text(question)
        if mentioned and not doc_ids and intent in {"summarize", "notes", "flashcards", "quiz", "exam_prep"}:
            doc_ids = [mentioned]
        topic = _strip_command(question)

        if intent in ("qa", "compare"):
            return self.a.answer(question, conversation_id, kw.get("k"), doc_ids, kw.get("category"),
                                 kw.get("diversify"), intent)

        if intent == "summarize":
            agent = SummarizationAgent(self.a)
            res = agent.summarize_document(doc_ids[0]) if doc_ids and len(doc_ids) == 1 and not topic \
                else (agent.summarize_document(mentioned) if mentioned else agent.summarize_topic(topic or question, doc_ids))
            return AssistantResponse(res["markdown"], res["sources"], cited_sources(res["markdown"], res["sources"]),
                                     [], intent)

        learner = LearningAgent(self.a)
        if intent == "notes":
            style = "Revision Guide" if "revision" in question.lower() else (
                "Exam Summary" if "exam" in question.lower() else "Chapter Notes")
            res = learner.study_notes(style, topic, doc_ids)
            return AssistantResponse(res["markdown"], res["sources"], cited_sources(res["markdown"], res["sources"]),
                                     [], intent)

        if intent == "flashcards":
            res = learner.flashcards(topic, 10, doc_ids)
            md = "### 🗂️ Flashcards\n\n" + "\n\n".join(
                f"**Q{i}.** {c['question']}\n\n> {c['answer']}" for i, c in enumerate(res["cards"], 1)) \
                if res["cards"] else "I couldn't make flashcards this time. Ask again, or pick a document in **Study → Flashcards**."
            return AssistantResponse(md, res["sources"], res["sources"][:5], [], intent, extra={"flashcards": res["cards"]})

        if intent == "quiz":
            res = learner.quiz(topic, 5, 3, "Medium", doc_ids)
            md = _quiz_markdown(res)
            return AssistantResponse(md, res["sources"], res["sources"][:5], [], intent,
                                     extra={"quiz": {"mcq": res["mcq"], "short": res["short"]}})

        if intent == "exam_prep":
            days_m = re.search(r"(\d+)\s*days?", question)
            res = ExamPrepAgent(self.a).prepare(question, int(days_m.group(1)) if days_m else 5, doc_ids)
            if "error" in res:
                return AssistantResponse(res["error"], [], [], [], intent)
            md = exam_prep_markdown(res)
            return AssistantResponse(md, res["sources"], res["sources"][:6], [], intent, extra={
                "flashcards": res["flashcards"]["cards"],
                "quiz": {"mcq": res["quiz"]["mcq"], "short": res["quiz"]["short"]}})

        return self.a.answer(question, conversation_id, kw.get("k"), doc_ids, kw.get("category"))


# ---------------------------------------------------------------- helpers
def _as_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


def _valid_mcq(q) -> bool:
    return (isinstance(q, dict) and q.get("question") and isinstance(q.get("options"), list)
            and len(q["options"]) >= 2 and 0 <= _as_int(q.get("answer_index")) < len(q["options"]))


def _strip_command(text: str) -> str:
    t = re.sub(r"(?i)\b(please|can you|could you|make|create|generate|give me|write|build|me|some|a|an|the|"
               r"summari[sz]e|summary of|flash ?cards?|quiz|mcqs?|practice (test|questions)|test me|"
               r"study notes|chapter notes|revision guide|exam summary|notes|on|about|for|of|from|my|documents?)\b",
               " ", text)
    return re.sub(r"\s+", " ", t).strip(" ?.!:,")


def _quiz_markdown(res: Dict) -> str:
    if not res["mcq"] and not res["short"]:
        return "I couldn't build a quiz this time. Ask again, or pick a document in **Study → Quiz**."
    out = ["### 📝 Practice Quiz"]
    for i, q in enumerate(res["mcq"], 1):
        opts = "\n".join(f"   {'ABCD'[j] if j < 4 else j}. {o}" for j, o in enumerate(q["options"]))
        out.append(f"**{i}. {q['question']}**\n{opts}")
    if res["short"]:
        out.append("**Short-answer questions**")
        out += [f"- {q['question']}" for q in res["short"]]
    out.append("\n<details><summary>Answer key</summary>\n\n" + "\n".join(
        f"{i}. {'ABCD'[q['answer_index']] if q['answer_index'] < 4 else q['answer_index']} – {q.get('explanation', '')}"
        for i, q in enumerate(res["mcq"], 1)) + "\n\n</details>")
    return "\n\n".join(out)


def exam_prep_markdown(res: Dict) -> str:
    parts = [f"# 🎓 Exam prep: {res['subject']}", "**Key topics:** " + ", ".join(res["topics"]),
             res["summary"]["markdown"], "---", res["plan"], "---", _quiz_markdown(res["quiz"]), "---",
             "### 🗂️ Flashcards\n\n" + "\n".join(f"- **{c['question']}** — {c['answer']}"
                                                  for c in res["flashcards"]["cards"])]
    return "\n\n".join(parts)


# ---------------------------------------------------------------- tolerant parsing of AI study output
_OPTION_PREFIX = re.compile(r"^\s*(?:\(?[A-Da-d1-4][).:\-]\s+)")


def _first(d: Dict, *keys):
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, ""):
            return d[k]
    return None


def _items(data, *keys) -> List:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in keys:
            if isinstance(data.get(k), list):
                return data[k]
    return []


def normalize_mcq(q) -> Optional[Dict]:
    """Accept the many shapes models use for an MCQ; return a clean dict or None."""
    if not isinstance(q, dict):
        return None
    question = _first(q, "question", "q", "prompt", "stem")
    options = _first(q, "options", "choices", "answers")
    if isinstance(options, dict):  # {"A": "...", "B": "..."}
        options = [options[k] for k in sorted(options)]
    if not question or not isinstance(options, list) or len(options) < 2:
        return None
    options = [_OPTION_PREFIX.sub("", str(o if not isinstance(o, dict) else _first(o, "text", "option", "value"))).strip()
               for o in options]
    raw = _first(q, "answer_index", "correct_index", "correct_option_index", "answer", "correct_answer",
                 "correct_option", "correct")
    idx = -1
    if isinstance(raw, bool):
        raw = None
    if isinstance(raw, (int, float)):
        idx = int(raw)
    elif isinstance(raw, str):
        r = raw.strip()
        if r.isdigit():
            idx = int(r)
        elif len(r) <= 3 and r[:1].upper() in "ABCD" and (len(r) == 1 or not r[1].isalpha()):
            idx = "ABCD".index(r[0].upper())
        else:
            text = _OPTION_PREFIX.sub("", r).strip().lower()
            idx = next((i for i, o in enumerate(options) if o.lower() == text), -1)
            if idx < 0:
                idx = next((i for i, o in enumerate(options) if text and (text in o.lower() or o.lower() in text)), -1)
    if not 0 <= idx < len(options):
        return None
    return {"question": str(question).strip(), "options": options, "answer_index": idx,
            "explanation": str(_first(q, "explanation", "reason", "rationale") or ""), "source": q.get("source")}


def normalize_quiz(data):
    mcq = [m for m in (normalize_mcq(q) for q in _items(data, "mcq", "mcqs", "multiple_choice", "questions",
                                                          "quiz")) if m]
    short = []
    for q in _items(data, "short", "short_answer", "short_answers", "short_questions"):
        question = _first(q, "question", "q") if isinstance(q, dict) else None
        if question:
            short.append({"question": str(question), "source": q.get("source"),
                          "model_answer": str(_first(q, "model_answer", "answer", "expected_answer") or "")})
    return mcq, short


def normalize_flashcards(data) -> List[Dict]:
    cards = []
    for c in _items(data, "flashcards", "cards", "items"):
        if not isinstance(c, dict):
            continue
        q = _first(c, "question", "front", "term", "q")
        a = _first(c, "answer", "back", "definition", "a")
        if q and a:
            cards.append({"question": str(q).strip(), "answer": str(a).strip(), "source": c.get("source")})
    return cards
