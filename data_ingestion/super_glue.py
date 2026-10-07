from typing import Any

from data_ingestion.base import DatasetAdapter
from router.benchmark_schema import BenchmarkPrompt, EvaluationType

SUPPORTED_TASKS = (
    "axb",
    "axg",
    "boolq",
    "cb",
    "copa",
    "multirc",
    "record",
    "rte",
    "wic",
    "wsc",
    "wsc.fixed",
)

CB_LABELS = ("entailment", "contradiction", "neutral")
RTE_LABELS = ("entailment", "not_entailment")


def format_super_glue_prompt(task: str, record: dict[str, Any]) -> str:
    """Format a SuperGLUE task record into a model-facing prompt without leaking targets."""
    if task in ("cb", "rte", "axg"):
        premise = record.get("premise")
        hypothesis = record.get("hypothesis")
        if not isinstance(premise, str) or not premise.strip():
            raise ValueError(f"SuperGLUE '{task}' record is missing a valid 'premise'.")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError(f"SuperGLUE '{task}' record is missing a valid 'hypothesis'.")
        return f"Premise:\n{premise.strip()}\n\nHypothesis:\n{hypothesis.strip()}"

    if task == "axb":
        s1 = record.get("sentence1")
        s2 = record.get("sentence2")
        if not isinstance(s1, str) or not s1.strip():
            raise ValueError("SuperGLUE 'axb' record is missing a valid 'sentence1'.")
        if not isinstance(s2, str) or not s2.strip():
            raise ValueError("SuperGLUE 'axb' record is missing a valid 'sentence2'.")
        return f"Sentence 1:\n{s1.strip()}\n\nSentence 2:\n{s2.strip()}"

    if task == "copa":
        premise = record.get("premise")
        choice1 = record.get("choice1")
        choice2 = record.get("choice2")
        question = record.get("question")
        if not isinstance(premise, str) or not premise.strip():
            raise ValueError("SuperGLUE 'copa' record is missing a valid 'premise'.")
        if not isinstance(choice1, str) or not choice1.strip():
            raise ValueError("SuperGLUE 'copa' record is missing a valid 'choice1'.")
        if not isinstance(choice2, str) or not choice2.strip():
            raise ValueError("SuperGLUE 'copa' record is missing a valid 'choice2'.")
        q_text = question.strip() if isinstance(question, str) and question.strip() else "result"
        return (
            f"Premise:\n{premise.strip()}\n\n"
            f"Question: What was the {q_text}?\n"
            f"(A) {choice1.strip()}\n"
            f"(B) {choice2.strip()}"
        )

    if task == "wic":
        word = record.get("word")
        s1 = record.get("sentence1")
        s2 = record.get("sentence2")
        if not isinstance(word, str) or not word.strip():
            raise ValueError("SuperGLUE 'wic' record is missing a valid 'word'.")
        if not isinstance(s1, str) or not s1.strip():
            raise ValueError("SuperGLUE 'wic' record is missing a valid 'sentence1'.")
        if not isinstance(s2, str) or not s2.strip():
            raise ValueError("SuperGLUE 'wic' record is missing a valid 'sentence2'.")
        return (
            f"Word: {word.strip()}\n\n"
            f"Sentence 1:\n{s1.strip()}\n\n"
            f"Sentence 2:\n{s2.strip()}\n\n"
            f"Question: Does the word '{word.strip()}' have the same meaning in both sentences?"
        )

    if task in ("wsc", "wsc.fixed"):
        text = record.get("text")
        span1 = record.get("span1_text")
        span2 = record.get("span2_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"SuperGLUE '{task}' record is missing a valid 'text'.")
        if not isinstance(span1, str) or not span1.strip():
            raise ValueError(f"SuperGLUE '{task}' record is missing a valid 'span1_text'.")
        if not isinstance(span2, str) or not span2.strip():
            raise ValueError(f"SuperGLUE '{task}' record is missing a valid 'span2_text'.")
        return (
            f"Text:\n{text.strip()}\n\n"
            f"Question: In the text above, does '{span2.strip()}' refer to '{span1.strip()}'?"
        )

    if task == "multirc":
        paragraph = record.get("paragraph")
        question = record.get("question")
        answer = record.get("answer")
        if not isinstance(paragraph, str) or not paragraph.strip():
            raise ValueError("SuperGLUE 'multirc' record is missing a valid 'paragraph'.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("SuperGLUE 'multirc' record is missing a valid 'question'.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("SuperGLUE 'multirc' record is missing a valid 'answer'.")
        return (
            f"Paragraph:\n{paragraph.strip()}\n\n"
            f"Question:\n{question.strip()}\n\n"
            f"Candidate Answer:\n{answer.strip()}\n\n"
            f"Question: Is this candidate answer correct?"
        )

    if task == "record":
        passage = record.get("passage")
        query = record.get("query")
        if not isinstance(passage, str) or not passage.strip():
            raise ValueError("SuperGLUE 'record' is missing a valid 'passage'.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("SuperGLUE 'record' is missing a valid 'query'.")
        return f"Passage:\n{passage.strip()}\n\nQuery:\n{query.strip()}"

    if task == "boolq":
        passage = record.get("passage")
        question = record.get("question")
        if not isinstance(passage, str) or not passage.strip():
            raise ValueError("SuperGLUE 'boolq' record is missing a valid 'passage'.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("SuperGLUE 'boolq' record is missing a valid 'question'.")
        return f"Passage:\n{passage.strip()}\n\nQuestion:\n{question.strip()}"

    raise ValueError(f"Unsupported SuperGLUE task: '{task}'")


def extract_super_glue_label(task: str, record: dict[str, Any]) -> tuple[str | None, EvaluationType]:
    """Extract canonical target string and evaluation type for a task record."""
    if task == "record":
        answers = record.get("answers")
        if isinstance(answers, list) and len(answers) > 0 and str(answers[0]).strip():
            return str(answers[0]).strip(), "exact_match"
        return None, "exact_match"

    label = record.get("label")
    if label is None or label == -1:
        eval_type: EvaluationType = "multiple_choice" if task == "copa" else "exact_match"
        return None, eval_type

    if task == "cb":
        if isinstance(label, int) and 0 <= label < len(CB_LABELS):
            return CB_LABELS[label], "exact_match"
        if isinstance(label, str) and label in CB_LABELS:
            return label, "exact_match"
        raise ValueError(f"Invalid label {label!r} for SuperGLUE 'cb'.")

    if task in ("rte", "axb", "axg"):
        if isinstance(label, int) and 0 <= label < len(RTE_LABELS):
            return RTE_LABELS[label], "exact_match"
        if isinstance(label, str) and label in RTE_LABELS:
            return label, "exact_match"
        raise ValueError(f"Invalid label {label!r} for SuperGLUE '{task}'.")

    if task == "copa":
        if label in (0, "0", "choice1", "(A)", "A"):
            return "(A)", "multiple_choice"
        if label in (1, "1", "choice2", "(B)", "B"):
            return "(B)", "multiple_choice"
        raise ValueError(f"Invalid label {label!r} for SuperGLUE 'copa'.")

    if task in ("boolq", "wic", "wsc", "wsc.fixed", "multirc"):
        if label in (0, False, "0", "False", "false"):
            return "False", "exact_match"
        if label in (1, True, "1", "True", "true"):
            return "True", "exact_match"
        raise ValueError(f"Invalid label {label!r} for SuperGLUE '{task}'.")

    raise ValueError(f"Unsupported SuperGLUE task: '{task}'")


def resolve_super_glue_id(record: dict[str, Any], fallback_id: str) -> str:
    """Resolve native stable ID from 'idx' field or fallback to positional ID."""
    idx = record.get("idx")
    if isinstance(idx, int):
        return str(idx)
    if isinstance(idx, dict):
        parts = [str(v) for v in idx.values()]
        return "-".join(parts)
    return str(fallback_id).strip()


class SuperGlueAdapter(DatasetAdapter):
    """Adapter for the SuperGLUE benchmark suite."""

    source_dataset = "super_glue"

    def __init__(self, source_config: str) -> None:
        if not isinstance(source_config, str) or not source_config.strip():
            raise ValueError("source_config must be a non-empty string.")

        source_config = source_config.strip()
        if source_config not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported SuperGLUE task: '{source_config}'. Supported: {sorted(SUPPORTED_TASKS)}"
            )

        self.source_config = source_config

    def convert_record(
        self,
        record: dict[str, Any],
        *,
        source_split: str,
        source_id: str,
        benchmark_split: str,
    ) -> BenchmarkPrompt:
        if not isinstance(record, dict):
            raise ValueError("SuperGLUE record must be a dictionary.")

        prompt_text = format_super_glue_prompt(self.source_config, record)
        expected_output, evaluation_type = extract_super_glue_label(self.source_config, record)
        resolved_source_id = resolve_super_glue_id(record, source_id)

        prompt_id = (
            f"super_glue-{self.source_config}-"
            f"{source_split}-{resolved_source_id}"
        )

        return BenchmarkPrompt(
            prompt_id=prompt_id,
            prompt=prompt_text,
            task_type="classification",
            domain=f"super_glue_{self.source_config}",
            difficulty="medium",
            source_dataset=self.source_dataset,
            source_config=self.source_config,
            source_split=source_split,
            source_id=resolved_source_id,
            split=benchmark_split,
            expected_output=expected_output,
            evaluation_type=evaluation_type,
        )
