"""Unit tests: embedding pipeline, vector store, retriever, citations."""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.citation import (
    build_context,
    build_sources,
    citation_coverage,
    cited_sources,
    extract_citation_numbers,
    hallucination_rate,
    strip_invalid_citations,
)
from src.embeddings import HashingEmbedder
from src.retriever import Retriever
from src.vector_store import NumpyVectorStore
from tests.helpers import TempAssistantMixin


class TestEmbeddings(unittest.TestCase):
    def test_shape_normalised_deterministic(self):
        e = HashingEmbedder(512)
        v = e.embed_documents(["machine learning models", "operating system kernels"])
        self.assertEqual(v.shape, (2, 512))
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, rtol=1e-5)
        np.testing.assert_allclose(v[0], e.embed_query("machine learning models"))

    def test_similar_texts_closer(self):
        e = HashingEmbedder()
        q = e.embed_query("what is overfitting in machine learning")
        a = e.embed_query("overfitting happens when a machine learning model memorises noise")
        b = e.embed_query("deadlock requires mutual exclusion and circular wait")
        self.assertGreater(float(q @ a), float(q @ b))


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_add_query_filter_delete_persist(self):
        e = HashingEmbedder()
        s = NumpyVectorStore(self.tmp, "c")
        texts = ["cats are mammals", "stack is LIFO", "queue is FIFO"]
        metas = [{"doc_id": "a", "category": "Notes"}, {"doc_id": "b", "category": "Education"},
                 {"doc_id": "b", "category": "Education"}]
        s.add(["1", "2", "3"], e.embed_documents(texts), texts, metas)
        self.assertEqual(s.count(), 3)
        hits = s.query(e.embed_query("LIFO stack"), k=2)
        self.assertEqual(hits[0].chunk_id, "2")
        self.assertTrue(all(h.metadata["doc_id"] == "a" for h in s.query(e.embed_query("stack"), 3, {"doc_id": "a"})))
        self.assertEqual(len(s.query(e.embed_query("x"), 5, {"doc_id": {"$in": ["a", "b"]}})), 3)
        s.update_metadata({"doc_id": "a"}, {"category": "Research"})
        s.delete({"doc_id": "b"})
        reloaded = NumpyVectorStore(self.tmp, "c")
        self.assertEqual(reloaded.count(), 1)
        self.assertEqual(reloaded.get()[0].metadata["category"], "Research")
        # upsert does not duplicate
        reloaded.add(["1"], e.embed_documents(["cats"]), ["cats"], [{"doc_id": "a"}])
        self.assertEqual(reloaded.count(), 1)


class TestRetriever(TempAssistantMixin, unittest.TestCase):
    def test_retrieves_right_document(self):
        self.load_samples()
        cases = {
            "What is process scheduling?": "operating_systems.txt",
            "Explain transfer learning": "machine_learning_notes.md",
            "How does a hash table resolve collisions?": "data_structures_notes.docx",
        }
        for q, expected in cases.items():
            top = self.a.search(q, k=3)
            self.assertEqual(top[0].file_name, expected, q)
            self.assertGreater(top[0].score, 0)

    def test_filters_and_diversify(self):
        self.load_samples()
        doc = next(d for d in self.a.memory.list_documents() if d["file_name"] == "operating_systems.txt")
        hits = self.a.search("learning", k=5, doc_ids=[doc["doc_id"]])
        self.assertTrue(all(h.file_name == "operating_systems.txt" for h in hits))
        diverse = self.a.search("definition of a data structure or model", k=3, diversify=True)
        self.assertEqual(len({h.file_name for h in diverse}), 3)
        self.assertEqual(self.a.search("anything", category="Research"), [])

    def test_empty_store(self):
        self.assertEqual(Retriever(self.a.embedder, self.a.store).retrieve("hello"), [])


class TestCitations(TempAssistantMixin, unittest.TestCase):
    def test_context_sources_and_parsing(self):
        self.load_samples()
        chunks = self.a.search("overfitting", k=3)
        ctx = build_context(chunks)
        self.assertIn("[1] Source: machine_learning_notes.md, page", ctx)
        sources = build_sources(chunks)
        answer = "Overfitting learns noise [1]. More data helps [1, 3]."
        self.assertEqual(extract_citation_numbers(answer), [1, 3])
        self.assertEqual([s.number for s in cited_sources(answer, sources)], [1, 3])
        self.assertEqual(strip_invalid_citations("A [1]. B [7]. C [2, 9].", 3), "A [1]. B . C [2].")
        self.assertIn("Page", sources[0].label())

    def test_metrics(self):
        ctx = "Overfitting is when a model learns noise in the training data."
        good = "Overfitting is when a model learns noise in the training data [1]."
        bad = "Quantum chromodynamics describes gluon interactions inside protons."
        self.assertEqual(citation_coverage(good), 1.0)
        self.assertEqual(hallucination_rate(good, ctx), 0.0)
        self.assertEqual(hallucination_rate(bad, ctx), 1.0)


if __name__ == "__main__":
    unittest.main()
