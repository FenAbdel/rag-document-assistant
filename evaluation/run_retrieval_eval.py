import argparse
import json
from pathlib import Path
from typing import Dict, List

from app.pipelines.rag_pipeline import RAGPipeline


EVALUATION_FILE = Path("evaluation/evaluation_questions.json")


def load_evaluation_questions(file_path: Path) -> List[Dict]:
    """
    Load retrieval evaluation questions from a JSON file.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            "Evaluation file not found: {}".format(file_path)
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_retrieval(
    questions: List[Dict],
    use_multi_query: bool = False,
) -> Dict:
    """
    Evaluate whether retrieval returns the expected source document.

    This is a simple retrieval evaluation baseline.
    """
    pipeline = RAGPipeline()

    results = []
    passed = 0

    for item in questions:
        question = item["question"]
        expected_source = item.get("expected_source")
        source_filter = item.get("source_filter")

        result = pipeline.answer_question(
            question=question,
            source_filter=source_filter,
            use_multi_query=use_multi_query,
        )

        retrieved_sources = [
            source["source"]
            for source in result.get("sources", [])
        ]

        source_found = expected_source in retrieved_sources

        if source_found:
            passed += 1

        results.append(
            {
                "id": item.get("id"),
                "question": question,
                "expected_source": expected_source,
                "retrieved_sources": retrieved_sources,
                "passed": source_found,
                "retrieval_mode": result.get("retrieval_mode"),
            }
        )

    total = len(questions)
    score = passed / total if total else 0

    return {
        "total_questions": total,
        "passed": passed,
        "score": score,
        "results": results,
    }


def print_report(report: Dict) -> None:
    """
    Print a readable retrieval evaluation report.
    """
    print("\nRetrieval evaluation report")
    print("-" * 50)
    print("Total questions: {}".format(report["total_questions"]))
    print("Passed: {}".format(report["passed"]))
    print("Score: {:.2%}".format(report["score"]))

    print("\nDetails")
    print("-" * 50)

    for result in report["results"]:
        status = "PASS" if result["passed"] else "FAIL"

        print("{} — {}".format(result["id"], status))
        print("Question: {}".format(result["question"]))
        print("Expected source: {}".format(result["expected_source"]))
        print(
            "Retrieved sources: {}".format(
                ", ".join(result["retrieved_sources"])
            )
        )
        print("Retrieval mode: {}".format(result["retrieval_mode"]))
        print("-" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality for the RAG assistant."
    )

    parser.add_argument(
        "--multi-query",
        action="store_true",
        help="Evaluate retrieval using optional multi-query mode.",
    )

    args = parser.parse_args()

    questions = load_evaluation_questions(EVALUATION_FILE)

    report = evaluate_retrieval(
        questions=questions,
        use_multi_query=args.multi_query,
    )

    print_report(report)


if __name__ == "__main__":
    main()