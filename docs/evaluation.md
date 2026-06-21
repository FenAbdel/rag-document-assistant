# Evaluation approach

## Objective

The goal of evaluation is to verify that the RAG assistant retrieves relevant context and produces grounded answers.

For this first professional version, evaluation is intentionally simple and understandable.

## What is evaluated

The project evaluates two main aspects:

1. Retrieval quality
2. Core logic correctness

## Retrieval evaluation

Retrieval evaluation checks whether the retriever returns chunks from the expected source document.

The evaluation questions are stored in:

```text
evaluation/evaluation_questions.json
```

Each item contains:

```json
{
  "id": "q1",
  "question": "What are the main technical skills mentioned?",
  "source_filter": "document.pdf",
  "expected_source": "document.pdf",
  "expected_keywords": ["Python", "SQL", "Power BI"]
}
```

The evaluation script is:

```text
evaluation/run_retrieval_eval.py
```

Run it with:

```bash
python -m evaluation.run_retrieval_eval
```

Run it with multi-query retrieval:

```bash
python -m evaluation.run_retrieval_eval --multi-query
```

## Current retrieval metric

The current metric is simple:

```text
Did the expected source appear in the retrieved sources?
```

This produces a score:

```text
passed questions / total questions
```

Example:

```text
Total questions: 3
Passed: 3
Score: 100%
```

## Why this is useful

This evaluation helps detect whether retrieval breaks after changes to:

- chunk size
- chunk overlap
- embedding model
- retrieval logic
- multi-query retrieval
- source filtering

## Unit tests

Unit tests are placed in:

```text
tests/
```

They test core logic without calling OpenAI.

Tested areas include:

- document ID generation
- memory formatting and trimming
- retrieval query construction
- source formatting
- prompt construction
- result deduplication

Run tests with:

```bash
pytest
```

## Cost control

The unit tests are free because they do not call external APIs.

The retrieval evaluation script may call the OpenAI API because it uses the real RAG pipeline.

Recommended usage:

- run `pytest` often
- run retrieval evaluation only after retrieval-related changes
- avoid re-indexing unless documents, chunking, or embedding model change

## Limitations of current evaluation

The current evaluation is a baseline.

It does not yet fully measure:

- answer factuality
- hallucination rate
- citation correctness
- semantic relevance of each chunk
- answer completeness
- ranking quality beyond expected source presence

## Future evaluation improvements

Possible improvements:

- expected page evaluation
- expected chunk evaluation
- keyword coverage score
- manual answer quality checklist
- LLM-as-judge evaluation
- retrieval precision and recall
- comparison between single-query and multi-query retrieval
- evaluation dashboard
