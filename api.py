"""FastAPI backend – the same assistant exposed as a REST API (for a React/Next.js frontend or other clients).

Run:   uvicorn api:app --reload
Docs:  http://localhost:8000/docs
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agents import ExamPrepAgent, LearningAgent, SummarizationAgent
from src.chat import KnowledgeAssistant
from src.config import CATEGORIES
from src.document_loader import SUPPORTED_EXTENSIONS, UnsupportedFileError

app = FastAPI(
    title="Personal AI Knowledge Assistant API",
    description="RAG over your own documents: upload, semantic search, cited Q&A, summaries, notes, flashcards, quizzes.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=1)
def assistant() -> KnowledgeAssistant:
    return KnowledgeAssistant()


# ------------------------------------------------------------------ schemas
class AskRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    mode: str = Field("auto", description="auto | qa | compare | summarize | notes | flashcards | quiz | exam_prep")
    k: Optional[int] = None
    doc_ids: Optional[List[str]] = None
    category: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    doc_ids: Optional[List[str]] = None
    category: Optional[str] = None
    diversify: bool = False


class StudyRequest(BaseModel):
    topic: str = ""
    doc_ids: Optional[List[str]] = None


class NotesRequest(StudyRequest):
    style: str = Field("Chapter Notes", description="Chapter Notes | Revision Guide | Exam Summary")


class FlashcardRequest(StudyRequest):
    n: int = 10


class QuizRequest(StudyRequest):
    n_mcq: int = 5
    n_short: int = 3
    difficulty: str = "Medium"


class SummaryRequest(BaseModel):
    doc_id: Optional[str] = None
    topic: Optional[str] = None
    length: str = "concise"


class ExamPrepRequest(BaseModel):
    goal: str
    days: int = 5
    doc_ids: Optional[List[str]] = None


class GradeRequest(BaseModel):
    question: str
    model_answer: str
    student_answer: str


class PreferenceUpdate(BaseModel):
    answer_style: Optional[str] = None
    expertise_level: Optional[str] = None
    language: Optional[str] = None
    custom_instructions: Optional[str] = None
    voice_replies: Optional[bool] = None


# ------------------------------------------------------------------ system
@app.get("/health")
def health():
    return {"status": "ok", **assistant().status()}


# ------------------------------------------------------------------ documents
@app.post("/documents")
async def upload_document(file: UploadFile = File(...), category: str = Form("General")):
    if category not in CATEGORIES:
        raise HTTPException(400, f"category must be one of {CATEGORIES}")
    ext = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type {ext}")
    data = await file.read()
    try:
        res = assistant().ingest_file(data, file.filename, category)
    except (UnsupportedFileError, ValueError) as e:
        raise HTTPException(422, str(e)) from e
    return res.__dict__


@app.get("/documents")
def list_documents(category: Optional[str] = None):
    return assistant().memory.list_documents(category)


@app.patch("/documents/{doc_id}")
def update_document(doc_id: str, category: str):
    if not assistant().memory.get_document(doc_id):
        raise HTTPException(404, "Document not found")
    assistant().set_category(doc_id, category)
    return {"doc_id": doc_id, "category": category}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if not assistant().memory.get_document(doc_id):
        raise HTTPException(404, "Document not found")
    assistant().delete_document(doc_id)
    return {"deleted": doc_id}


@app.get("/documents/{doc_id}/chunks")
def document_chunks(doc_id: str, limit: int = 50):
    return [c.to_dict() for c in assistant().retriever.get_document_chunks(doc_id)[:limit]]


# ------------------------------------------------------------------ search & chat
@app.post("/search")
def search(req: SearchRequest):
    return [r.to_dict() for r in assistant().search(req.query, req.k, req.doc_ids, req.category, req.diversify)]


@app.post("/ask")
def ask(req: AskRequest):
    a = assistant()
    cid = req.conversation_id or a.memory.create_conversation()
    resp = a.ask(req.question, cid, mode=req.mode, k=req.k, doc_ids=req.doc_ids, category=req.category)
    return {"conversation_id": cid, **resp.to_dict()}


@app.get("/conversations")
def conversations():
    return assistant().memory.list_conversations()


@app.get("/conversations/{cid}")
def conversation(cid: str):
    return assistant().memory.get_messages(cid)


@app.delete("/conversations/{cid}")
def delete_conversation(cid: str):
    assistant().memory.delete_conversation(cid)
    return {"deleted": cid}


# ------------------------------------------------------------------ study tools / agents
def _src(res: dict) -> dict:
    return {**res, "sources": [s.to_dict() for s in res.get("sources", [])]}


@app.post("/summarize")
def summarize(req: SummaryRequest):
    agent = SummarizationAgent(assistant())
    if req.doc_id:
        return _src(agent.summarize_document(req.doc_id, req.length))
    if req.topic:
        return _src(agent.summarize_topic(req.topic, length=req.length))
    raise HTTPException(400, "Provide doc_id or topic")


@app.post("/notes")
def notes(req: NotesRequest):
    return _src(LearningAgent(assistant()).study_notes(req.style, req.topic, req.doc_ids))


@app.post("/flashcards")
def flashcards(req: FlashcardRequest):
    return _src(LearningAgent(assistant()).flashcards(req.topic, req.n, req.doc_ids))


@app.post("/quiz")
def quiz(req: QuizRequest):
    return _src(LearningAgent(assistant()).quiz(req.topic, req.n_mcq, req.n_short, req.difficulty, req.doc_ids))


@app.post("/quiz/grade")
def grade(req: GradeRequest):
    return LearningAgent(assistant()).grade_short_answer(req.question, req.model_answer, req.student_answer)


@app.post("/exam-prep")
def exam_prep(req: ExamPrepRequest):
    res = ExamPrepAgent(assistant()).prepare(req.goal, req.days, req.doc_ids)
    if "error" in res:
        raise HTTPException(404, res["error"])
    return {
        "subject": res["subject"],
        "topics": res["topics"],
        "summary": _src(res["summary"]),
        "plan": res["plan"],
        "quiz": _src(res["quiz"]),
        "flashcards": _src(res["flashcards"]),
        "sources": [s.to_dict() for s in res["sources"]],
    }


# ------------------------------------------------------------------ memory
@app.get("/preferences")
def get_preferences():
    return assistant().memory.get_preferences()


@app.put("/preferences")
def set_preferences(update: PreferenceUpdate):
    for key, value in update.model_dump(exclude_none=True).items():
        assistant().memory.set_preference(key, str(value).lower() if isinstance(value, bool) else value)
    return assistant().memory.get_preferences()


@app.get("/topics")
def topics(n: int = 10):
    return assistant().memory.top_topics(n)
