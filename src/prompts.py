"""Prompt templates (prompt engineering lives here so it is easy to iterate on)."""

SYSTEM_BASE = """You are a Personal AI Knowledge Assistant. You answer ONLY from the user's own documents,
which are provided to you as numbered context passages.

Rules:
1. Ground every factual statement in the context and cite it inline with the passage number, e.g. [1] or [2, 3].
2. If the context does not contain the answer, say so plainly ("I couldn't find this in your documents")
   and, if helpful, suggest what document might contain it. Never invent facts or citations.
3. When passages come from several documents, synthesise them and point out agreements or differences.
4. Use clear Markdown: short paragraphs, bullet points or tables when they help.
{preferences}"""

QA_PROMPT = """Conversation so far (for context only):
{history}

Context passages from the user's documents:
{context}

Question: {question}

Answer the question using the context passages and cite them inline like [1]."""

CONDENSE_PROMPT = """Rewrite the follow-up question as a standalone search query, resolving pronouns using the conversation.
Return ONLY the rewritten query.

Conversation:
{history}

Follow-up question: {question}
Standalone query:"""

COMPARE_PROMPT = """Context passages drawn from multiple documents:
{context}

Task: {question}

Write a structured comparison. Start with a short overview, then a Markdown table comparing the key
dimensions, then bullet-point takeaways. Cite passages inline like [1]. Mention which document each view comes from."""

SUMMARY_PROMPT = """Below is content from the document "{file_name}".

{context}

Produce a {length} summary in Markdown with exactly these sections:
## Summary
## Key Findings
(bullet points)
## Recommendations
(bullet points: what the reader should do / study next, or practical implications)

Cite passages inline like [1] where useful."""

NOTES_PROMPTS = {
    "Chapter Notes": """Create detailed, well-structured chapter notes from the content below.
Use headings for each major topic, bullet points for key ideas, **bold** for key terms, and include
definitions, examples and any formulas. Cite passages like [1].""",
    "Revision Guide": """Create a concise revision guide from the content below. For each topic give:
the core idea in one line, 3-5 must-remember points, common mistakes / confusions, and a quick self-check
question. Cite passages like [1].""",
    "Exam Summary": """Create a one-page exam cheat-sheet from the content below: the most testable definitions,
formulas, comparisons (as small tables), and a 'likely exam questions' list at the end. Be dense and
skimmable. Cite passages like [1].""",
}

NOTES_WRAPPER = """{instruction}

Topic focus: {topic}

Content:
{context}"""

STUDY_RULES = """Rules:
- Test real subject knowledge only: concepts, definitions, how things work, code behaviour, differences, examples.
- IGNORE administrative text such as names, registration numbers, dates, course codes, page numbers, marks,
  headings, table labels and form fields. Never ask about them.
- Every question must be answerable from the content and make sense on its own."""

FLASHCARD_PROMPT = """From the content below, create {n} high-quality flashcards for active recall on: {topic}.
Each card tests ONE concept. Answers should be 1-2 sentences.

""" + STUDY_RULES + """

Content:
{context}

Return ONLY valid JSON, exactly like this:
{{"flashcards": [{{"question": "What is a thread?", "answer": "The smallest unit of CPU execution within a process.", "source": 1}}]}}"""

QUIZ_PROMPT = """From the content below, create a practice test on: {topic}.
Include {n_mcq} multiple-choice questions (4 options each, exactly one correct) and {n_short} short-answer questions.
Difficulty: {difficulty}. Wrong options must be plausible but clearly wrong to someone who knows the material.

""" + STUDY_RULES + """

Content:
{context}

Return ONLY valid JSON, exactly like this ("answer_index" is the 0-based position of the correct option):
{{"mcq": [{{"question": "Which scheduling algorithm gives each process a fixed time slice?",
           "options": ["First-Come First-Served", "Round Robin", "Shortest Job First", "Priority"],
           "answer_index": 1, "explanation": "Round Robin cycles through processes with a fixed quantum.", "source": 2}}],
  "short": [{{"question": "Why can SJF cause starvation?", "model_answer": "Long jobs may wait forever if short jobs keep arriving.", "source": 3}}]}}"""

REVISION_PLAN_PROMPT = """The student says: "{goal}"
They have {days} days to prepare. Topics found in their materials:
{topics}

Relevant content:
{context}

Create a day-by-day revision plan in Markdown. For each day: focus topics, specific activities
(read / summarise / flashcards / practice questions), and a checkpoint. Finish with exam-day tips.
Base topics on the content and cite passages like [1]."""

TOPICS_PROMPT = """List the {n} most important distinct topics covered in the content below, as a JSON list of short strings.
Return ONLY the JSON list.

Content:
{context}"""

INTENT_PROMPT = """Classify the user's request into one of: qa, compare, summarize, notes, flashcards, quiz, exam_prep.
Return ONLY the label.

Request: {question}"""

GRADE_PROMPT = """Grade the student's answer against the model answer. Reply as JSON:
{{"score": number from 0 to 1, "feedback": short string}}

Question: {question}
Model answer: {model_answer}
Student answer: {student_answer}"""

JUDGE_PROMPT = """You are evaluating a RAG system answer. Rate from 1 to 5 how well the answer addresses the
question AND stays faithful to the context. Reply as JSON: {{"score": int, "reason": short string}}

Question: {question}
Context:
{context}
Answer:
{answer}"""
