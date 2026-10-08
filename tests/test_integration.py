"""Integration tests: Document Upload -> Retrieval -> Response, agents, memory, evaluation, API."""
import importlib.util
import unittest

from src.agents import ExamPrepAgent, LearningAgent, RetrievalAgent, SummarizationAgent
from src.chat import detect_intent
from src.evaluation import run_full_evaluation
from src.llm_engine import LLMError, parse_json
from tests.helpers import SAMPLES, TempAssistantMixin, make_pdf


class TestOfflinePipeline(TempAssistantMixin, unittest.TestCase):
    def test_upload_retrieve_answer_with_citations(self):
        res = self.a.ingest_file(make_pdf(["Transfer learning reuses a pretrained model for a new task.",
                                           "Fine-tuning adapts the pretrained weights."]), "paper.pdf", "Research")
        self.assertEqual((res.num_pages, res.duplicate), (2, False))
        cid = self.a.memory.create_conversation()
        resp = self.a.ask("What is transfer learning?", cid)
        self.assertEqual(resp.intent, "qa")
        self.assertIn("[1]", resp.answer)
        self.assertEqual(resp.cited[0].file_name, "paper.pdf")
        self.assertEqual(resp.cited[0].page, 1)
        msgs = self.a.memory.get_messages(cid)
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[1]["sources"][0]["file_name"], "paper.pdf")
        self.assertEqual(self.a.memory.list_conversations()[0]["title"], "What is transfer learning?")

    def test_duplicate_upload_skipped_and_delete(self):
        first = self.a.ingest_file(SAMPLES / "operating_systems.txt")
        again = self.a.ingest_file(SAMPLES / "operating_systems.txt")
        self.assertTrue(again.duplicate)
        self.assertEqual(self.a.store.count(), first.num_chunks)
        self.a.delete_document(first.doc_id)
        self.assertEqual(self.a.store.count(), 0)
        self.assertEqual(self.a.memory.list_documents(), [])

    def test_category_change(self):
        res = self.a.ingest_file(SAMPLES / "operating_systems.txt", category="Notes")
        self.a.set_category(res.doc_id, "Projects")
        self.assertEqual(self.a.memory.get_document(res.doc_id)["category"], "Projects")
        self.assertTrue(self.a.search("deadlock", category="Projects"))

    def test_intent_routing(self):
        self.assertEqual(detect_intent("Compare supervised and unsupervised learning"), "compare")
        self.assertEqual(detect_intent("Summarize ResearchPaper.pdf"), "summarize")
        self.assertEqual(detect_intent("Make flashcards on hashing"), "flashcards")
        self.assertEqual(detect_intent("Give me MCQs on deadlocks"), "quiz")
        self.assertEqual(detect_intent("Prepare me for my DBMS exam"), "exam_prep")
        self.assertEqual(detect_intent("Create a revision guide for trees"), "notes")
        self.assertEqual(detect_intent("What is paging?"), "qa")

    def test_agents_offline(self):
        self.load_samples()
        ds = next(d for d in self.a.memory.list_documents() if d["file_name"].startswith("data_structures"))
        self.assertEqual(RetrievalAgent(self.a).find_document_in_text("data structures"), ds["doc_id"])
        summary = SummarizationAgent(self.a).summarize_document(ds["doc_id"])
        for section in ("## Summary", "## Key Findings", "## Recommendations"):
            self.assertIn(section, summary["markdown"])
        cards = LearningAgent(self.a).flashcards("data structures", 5)["cards"]
        self.assertEqual(len(cards), 5)
        self.assertTrue(any(c["question"] == "What is Stack?" for c in cards))
        quiz = LearningAgent(self.a).quiz("process scheduling", 3, 2)
        self.assertEqual(len(quiz["mcq"]), 3)
        for q in quiz["mcq"]:
            self.assertEqual(len(q["options"]), 4)
            self.assertTrue(0 <= q["answer_index"] < 4)
        notes = LearningAgent(self.a).study_notes("Revision Guide", "machine learning")
        self.assertIn("Key Definitions", notes["markdown"])
        grade = LearningAgent(self.a).grade_short_answer("q", "noise in training data", "it learns noise in data")
        self.assertGreater(grade["score"], 0.5)

    def test_exam_prep_offline(self):
        self.load_samples()
        res = ExamPrepAgent(self.a).prepare("Prepare me for my operating systems exam", days=3)
        self.assertEqual(res["subject"], "operating systems")
        self.assertTrue(res["topics"])
        self.assertIn("Day 1", res["plan"])
        self.assertTrue(res["quiz"]["mcq"])
        self.assertTrue(res["flashcards"]["cards"])
        # study material should come from the OS notes only
        self.assertEqual({s.file_name for s in res["sources"]}, {"operating_systems.txt"})
        chat = self.a.ask("Prepare me for my operating systems exam in 4 days")
        self.assertEqual(chat.intent, "exam_prep")
        self.assertIn("4-Day Revision Plan", chat.answer)

    def test_memory_preferences_and_topics(self):
        self.a.memory.set_preference("answer_style", "Concise")
        self.assertIn("short", self.a.memory.preferences_prompt())
        self.a.memory.record_topics("overfitting regularization")
        self.a.memory.record_topics("overfitting dropout")
        top = self.a.memory.top_topics(1)[0]
        self.assertEqual((top["topic"], top["count"]), ("overfitting", 2))

    def test_evaluation(self):
        self.load_samples()
        report = run_full_evaluation(self.a, str(SAMPLES / "eval_set.json"), k=5, judge=False)
        self.assertGreaterEqual(report["retrieval"]["hit_rate@5"], 0.8)
        self.assertGreaterEqual(report["answers"]["citation_coverage"], 0.5)
        self.assertLessEqual(report["answers"]["hallucination_rate"], 0.2)


class TestWithLLM(TempAssistantMixin, unittest.TestCase):
    fake_llm = True

    def test_llm_answer_path(self):
        self.load_samples()
        cid = self.a.memory.create_conversation()
        self.a.ask("What is overfitting?", cid)
        resp = self.a.ask("How can it be reduced?", cid)  # follow-up -> condensed query
        self.assertEqual(resp.search_query, "overfitting in machine learning")
        self.assertNotIn("[9]", resp.answer)  # invalid citation stripped
        self.assertEqual([s.number for s in resp.cited], [1])
        self.assertIn("User preferences", "".join(self.a.llm.calls) + self._system_used())

    def _system_used(self):
        from src import prompts
        return prompts.SYSTEM_BASE.format(preferences=self.a.memory.preferences_prompt())

    def test_llm_study_tools_parse_json(self):
        self.load_samples()
        cards = LearningAgent(self.a).flashcards("overfitting", 3)["cards"]
        self.assertEqual(cards[0]["question"], "What is overfitting?")
        quiz = LearningAgent(self.a).quiz("overfitting", 1, 1)
        self.assertEqual(quiz["mcq"][0]["answer_index"], 0)
        self.assertEqual(LearningAgent(self.a).grade_short_answer("q", "a", "b")["score"], 0.8)
        topics = ExamPrepAgent(self.a).topics(self.a.search("processes", k=3))
        self.assertEqual(topics, ["Processes", "Scheduling", "Deadlocks"])
        report = run_full_evaluation(self.a, str(SAMPLES / "eval_set.json"), judge=True)
        self.assertEqual(report["answers"]["response_quality_1to5"], 4.0)

    def test_parse_json(self):
        self.assertEqual(parse_json('Sure! ```json\n[1, 2]\n```'), [1, 2])
        self.assertEqual(parse_json('Here: {"a": 1} done'), {"a": 1})
        with self.assertRaises(LLMError):
            parse_json("no json here")


@unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"), "fastapi not installed")
class TestAPI(TempAssistantMixin, unittest.TestCase):
    def test_endpoints(self):
        from fastapi.testclient import TestClient

        import api

        api.assistant.cache_clear()
        api.assistant = lambda: self.a  # noqa: E731 - inject the isolated assistant
        client = TestClient(api.app)
        self.assertEqual(client.get("/health").json()["status"], "ok")
        r = client.post("/documents", files={"file": ("os.txt", (SAMPLES / "operating_systems.txt").read_bytes())},
                        data={"category": "Education"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(client.get("/documents").json()), 1)
        ans = client.post("/ask", json={"question": "What is a deadlock?"}).json()
        self.assertTrue(ans["sources"])
        self.assertTrue(client.post("/search", json={"query": "paging"}).json())
        self.assertIn("cards", client.post("/flashcards", json={"topic": "scheduling", "n": 3}).json())
        self.assertEqual(client.post("/documents", files={"file": ("x.png", b"123")}).status_code, 415)


if __name__ == "__main__":
    unittest.main()
