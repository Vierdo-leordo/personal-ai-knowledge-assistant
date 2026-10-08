"""Quiz/flashcard robustness: tolerant parsing of AI output, retry, and no junk fallback."""
import json
import unittest

from src.agents import LearningAgent, normalize_flashcards, normalize_mcq, normalize_quiz
from src.document_loader import Page, remove_repeated_lines
from src.offline import heuristic_quiz, looks_like_form
from src.retriever import RetrievedChunk
from tests.helpers import FakeLLM, TempAssistantMixin

OPTS = ["First-Come First-Served", "Round Robin", "Shortest Job First", "Priority"]


class TestNormalisation(unittest.TestCase):
    def test_answer_formats(self):
        for raw in (1, "1", "B", "b", "B)", "Round Robin", "B) Round Robin"):
            q = normalize_mcq({"question": "Q?", "options": OPTS, "answer_index" if raw in (1, "1") else "answer": raw})
            self.assertEqual(q["answer_index"], 1, raw)

    def test_option_shapes(self):
        q = normalize_mcq({"q": "Q?", "choices": {"A": "a) x", "B": "b) y", "C": "z", "D": "w"}, "correct_answer": "C"})
        self.assertEqual((q["options"], q["answer_index"]), (["x", "y", "z", "w"], 2))
        self.assertIsNone(normalize_mcq({"question": "Q?", "options": OPTS, "answer": "Nope"}))

    def test_quiz_and_cards_containers(self):
        mcq, short = normalize_quiz({"questions": [{"question": "Q", "options": OPTS, "answer": "A"}],
                                     "short_answer": [{"question": "Why?", "answer": "Because."}]})
        self.assertEqual((len(mcq), short[0]["model_answer"]), (1, "Because."))
        self.assertEqual(normalize_flashcards({"cards": [{"front": "F", "back": "B"}]})[0]["answer"], "B")


class TestCleanInput(unittest.TestCase):
    def test_repeated_header_lines_removed(self):
        header = "Registration no.: Branch & Section: Faculty Name:"
        topics = ["threads", "processes", "deadlocks", "paging", "semaphores"]
        pages = [Page(f"{header}\nExperiment {i}: {t}\nThis page explains {t} in detail.\nimport java.util.*;\n"
                      f"class Demo{i} {{}}\nMore notes about {t} here.\nPage {i} of 5", i)
                 for i, t in enumerate(topics, 1)]
        out = remove_repeated_lines(pages)
        self.assertTrue(all(header not in p.text and "of 5" not in p.text for p in out))
        self.assertTrue(all("import java.util.*;" in p.text for p in out))  # mid-page code is kept
        self.assertIn("explains threads", out[0].text)

    def test_form_text_not_used_for_offline_quiz(self):
        self.assertTrue(looks_like_form("Registration no.: Branch & Section: Faculty Name:"))
        chunk = RetrievedChunk("Name: Rahul. Registration no.: 1234. Product Name: Laptop is a good product name here.\n"
                               "A thread is the smallest unit of CPU execution within a process.\n"
                               "A process is a program in execution with its own memory.\n"
                               "A deadlock is a state where processes wait for each other forever.\n"
                               "A semaphore is a variable used to control access to a shared resource.",
                               {"file_name": "x.pdf", "page": 1}, 1, 1, 0)
        quiz = heuristic_quiz([chunk], 3, 0)
        for q in quiz["mcq"]:
            self.assertNotRegex(q["question"], r"Name|Registration")


class ScriptedLLM(FakeLLM):
    def __init__(self, settings, replies):
        super().__init__(settings)
        self.replies = list(replies)

    def generate(self, prompt, **kw):
        self.calls.append(prompt)
        return self.replies.pop(0) if self.replies else "not json"


class TestRetryAndNoJunk(TempAssistantMixin, unittest.TestCase):
    fake_llm = True

    def test_retry_then_success_with_letter_answers(self):
        self.load_samples()
        good = json.dumps({"mcq": [{"question": "Which uses a time quantum?", "options": OPTS, "answer": "B"}]})
        self.a.llm = ScriptedLLM(self.a.settings, ["garbage", good])
        res = LearningAgent(self.a).quiz("scheduling", 1, 0)
        self.assertEqual(res["mcq"][0]["answer_index"], 1)
        self.assertEqual(len(self.a.llm.calls), 2)
        self.assertNotIn("Name:", self.a.llm.calls[0].split("Content:")[0])  # prompt carries the study rules
        self.assertIn("IGNORE administrative text", self.a.llm.calls[0])

    def test_ai_failure_gives_error_not_nonsense(self):
        self.load_samples()
        self.a.llm = ScriptedLLM(self.a.settings, ["bad", "still bad"])
        res = LearningAgent(self.a).quiz("scheduling", 3, 1)
        self.assertEqual((res["mcq"], res["short"]), ([], []))
        self.assertTrue(res["error"])
        cards = LearningAgent(self.a).flashcards("scheduling", 3)
        self.assertEqual(cards["cards"], [])
        self.assertTrue(cards["error"])


if __name__ == "__main__":
    unittest.main()
