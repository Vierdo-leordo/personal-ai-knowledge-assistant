"""Unit tests: document extraction, cleaning, chunking."""
import unittest

from src.chunker import chunk_document, chunk_text
from src.document_loader import (
    Page,
    UnsupportedFileError,
    clean_text,
    load_document,
    remove_duplicate_paragraphs,
)
from tests.helpers import SAMPLES, make_pdf


class TestDocumentLoader(unittest.TestCase):
    def test_pdf_extraction_keeps_page_numbers(self):
        pdf = make_pdf(["Operating systems manage hardware resources.", "Process scheduling allocates the CPU."])
        doc = load_document(pdf, "os.pdf")
        self.assertEqual(doc.file_type, "pdf")
        self.assertEqual(doc.num_pages, 2)
        self.assertIn("Process scheduling", doc.pages[1].text)
        self.assertEqual(doc.pages[1].page_number, 2)

    def test_docx_extraction_includes_headings_and_tables(self):
        doc = load_document(SAMPLES / "data_structures_notes.docx")
        text = doc.full_text
        self.assertIn("A stack is a linear data structure", text)
        self.assertIn("Linked list", text)  # table content
        self.assertGreaterEqual(doc.num_pages, 1)

    def test_txt_and_markdown(self):
        md = load_document(SAMPLES / "machine_learning_notes.md")
        self.assertTrue(any(p.heading == "Overfitting and Underfitting" for p in md.pages))
        txt = load_document(b"Hello world. This is a note.", "note.txt")
        self.assertEqual(txt.full_text, "Hello world. This is a note.")

    def test_unsupported_type(self):
        with self.assertRaises(UnsupportedFileError):
            load_document(b"x", "image.png")

    def test_clean_text(self):
        raw = "Machine lear-\nning is   great.\n\n\n\n12\nPage 3 of 10\nNext line"
        self.assertEqual(clean_text(raw), "Machine learning is great.\n\nNext line")

    def test_remove_duplicate_paragraphs(self):
        header = "University of Examples – Department of Computer Science"
        pages = [Page(f"{header}\n\nContent one is here.", 1), Page(f"{header}\n\nContent two is here.", 2)]
        out = remove_duplicate_paragraphs(pages)
        self.assertEqual(sum(header in p.text for p in out), 1)

    def test_same_file_same_hash(self):
        a = load_document(b"same content", "a.txt")
        b = load_document(b"same content", "b.txt")
        self.assertEqual(a.content_hash, b.content_hash)


class TestChunker(unittest.TestCase):
    def test_chunk_size_and_overlap(self):
        text = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(200))
        chunks = chunk_text(text, chunk_size=300, chunk_overlap=60)
        self.assertGreater(len(chunks), 5)
        self.assertTrue(all(len(c) <= 360 for c in chunks))
        # overlap: the last sentence of a chunk re-appears at the start of the next
        last_sentence = chunks[0].split(". ")[-1].rstrip(".")
        self.assertIn(last_sentence, chunks[1])

    def test_long_sentence_is_wrapped(self):
        chunks = chunk_text("word " * 1000, chunk_size=200, chunk_overlap=20)
        self.assertTrue(all(len(c) <= 220 for c in chunks))

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", 100, 100)

    def test_chunk_metadata(self):
        doc = load_document(SAMPLES / "operating_systems.txt")
        chunks = chunk_document(doc, 500, 80, category="Education")
        self.assertGreater(len(chunks), 3)
        meta = chunks[0].metadata
        for key in ("file_name", "page", "chunk_id", "upload_date", "category", "doc_id"):
            self.assertIn(key, meta)
        self.assertEqual(meta["category"], "Education")
        self.assertEqual(len({c.chunk_id for c in chunks}), len(chunks))
        # chunks start at section headings and carry them as metadata
        self.assertTrue(any(c.metadata["heading"] == "Chapter 3: Deadlocks" for c in chunks))


if __name__ == "__main__":
    unittest.main()
