"""Command-line interface.

Examples:
    python cli.py ingest sample_data/*.md sample_data/*.txt sample_data/*.docx --category Education
    python cli.py ask "Explain overfitting"
    python cli.py search "deadlock conditions" -k 3
    python cli.py evaluate sample_data/eval_set.json
    python cli.py status
"""
from __future__ import annotations

import argparse
import json

from src.chat import KnowledgeAssistant


def main() -> None:
    parser = argparse.ArgumentParser(description="Personal AI Knowledge Assistant")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest", help="add documents to the knowledge base")
    p.add_argument("files", nargs="+")
    p.add_argument("--category", default="General")
    p = sub.add_parser("ask", help="ask a question (agent routing applies)")
    p.add_argument("question")
    p = sub.add_parser("search", help="semantic search only")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p = sub.add_parser("evaluate", help="run the evaluation suite")
    p.add_argument("eval_file")
    p.add_argument("-k", type=int, default=5)
    sub.add_parser("status")
    args = parser.parse_args()

    a = KnowledgeAssistant()
    if args.cmd == "ingest":
        for f in args.files:
            r = a.ingest_file(f, category=args.category)
            print(f"{'skipped (duplicate)' if r.duplicate else 'indexed'}: {r.file_name} – {r.num_pages} pages, {r.num_chunks} chunks")
    elif args.cmd == "ask":
        r = a.ask(args.question)
        print(r.answer)
        print("\nSources:")
        for s in r.cited:
            print(f"  [{s.number}] {s.label()}  (score {s.score:.2f})")
    elif args.cmd == "search":
        for i, r in enumerate(a.search(args.query, args.k), 1):
            print(f"{i}. {r.file_name} p.{r.page}  score={r.score:.3f}\n   {r.text[:200]}…\n")
    elif args.cmd == "evaluate":
        from src.evaluation import run_full_evaluation

        rep = run_full_evaluation(a, args.eval_file, args.k)
        print(json.dumps({"retrieval": {k: v for k, v in rep["retrieval"].items() if k != "rows"},
                          "answers": {k: v for k, v in rep["answers"].items() if k != "rows"}}, indent=2))
    else:
        print(json.dumps(a.status(), indent=2))


if __name__ == "__main__":
    main()
