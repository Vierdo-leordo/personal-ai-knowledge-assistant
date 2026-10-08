"""AI Evaluation: retrieval accuracy, citation coverage, response quality and hallucination rate.

Evaluation set format (JSON list):
[
  {"question": "What is overfitting?",
   "expected_file": "ml_notes.md",          # file that should be retrieved
   "expected_page": 2,                      # optional
   "expected_keywords": ["noise", "training data"]}   # optional, checked in the answer
]
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Union

from . import prompts
from .citation import build_context, citation_coverage, hallucination_rate
from .llm_engine import LLMError

if TYPE_CHECKING:  # pragma: no cover
    from .chat import KnowledgeAssistant


def load_eval_set(source: Union[str, Path, List[Dict]]) -> List[Dict]:
    if isinstance(source, list):
        return source
    return json.loads(Path(source).read_text())


def evaluate_retrieval(assistant: "KnowledgeAssistant", items: List[Dict], k: int = 5) -> Dict:
    hits, rr, rows = 0, [], []
    for it in items:
        results = assistant.search(it["question"], k=k)
        rank = None
        for i, r in enumerate(results, start=1):
            if r.file_name == it["expected_file"] and (
                "expected_page" not in it or not it["expected_page"] or r.page == it["expected_page"]
            ):
                rank = i
                break
        hits += rank is not None
        rr.append(1.0 / rank if rank else 0.0)
        rows.append({"question": it["question"], "expected": it["expected_file"], "rank": rank,
                     "top_result": f"{results[0].file_name} p.{results[0].page}" if results else "-"})
    n = max(1, len(items))
    return {f"hit_rate@{k}": round(hits / n, 3), "mrr": round(sum(rr) / n, 3), "rows": rows}


def judge_answer(assistant: "KnowledgeAssistant", question: str, context: str, answer: str) -> float:
    """LLM-as-judge quality score (1-5), or None when offline."""
    if assistant.llm.is_offline:
        return None
    try:
        res = assistant.llm.generate_json(prompts.JUDGE_PROMPT.format(question=question, context=context[:8000],
                                                                      answer=answer), temperature=0)
        return float(res.get("score"))
    except (LLMError, TypeError, ValueError, AttributeError):
        return None


def evaluate_answers(assistant: "KnowledgeAssistant", items: List[Dict], k: int = 5, judge: bool = True) -> Dict:
    rows = []
    for it in items:
        resp = assistant.answer(it["question"], k=k)
        context = build_context(resp.retrieved)
        kw = it.get("expected_keywords") or []
        kw_recall = sum(1 for w in kw if w.lower() in resp.answer.lower()) / len(kw) if kw else None
        cited_files = {s.file_name for s in resp.cited}
        rows.append({
            "question": it["question"],
            "citation_coverage": round(citation_coverage(resp.answer), 3),
            "hallucination_rate": round(hallucination_rate(resp.answer, context), 3),
            "cites_expected_file": it.get("expected_file") in cited_files if it.get("expected_file") else None,
            "keyword_recall": round(kw_recall, 3) if kw_recall is not None else None,
            "judge_score": judge_answer(assistant, it["question"], context, resp.answer) if judge else None,
            "answer": resp.answer,
        })

    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(statistics.mean([float(v) for v in vals]), 3) if vals else None

    return {
        "citation_coverage": mean("citation_coverage"),
        "hallucination_rate": mean("hallucination_rate"),
        "source_accuracy": mean("cites_expected_file"),
        "keyword_recall": mean("keyword_recall"),
        "response_quality_1to5": mean("judge_score"),
        "rows": rows,
    }


def run_full_evaluation(assistant: "KnowledgeAssistant", source, k: int = 5, judge: bool = True) -> Dict:
    items = load_eval_set(source)
    return {"retrieval": evaluate_retrieval(assistant, items, k), "answers": evaluate_answers(assistant, items, k, judge),
            "n_questions": len(items)}
