import pytest

from data_ingestion.super_glue import (
    SuperGlueAdapter,
    extract_super_glue_label,
    format_super_glue_prompt,
    resolve_super_glue_id,
)


def test_resolve_super_glue_id():
    assert resolve_super_glue_id({"idx": 123}, "fallback") == "123"
    assert resolve_super_glue_id({"idx": {"paragraph": 1, "question": 2, "answer": 3}}, "fallback") == "1-2-3"
    assert resolve_super_glue_id({}, "fallback") == "fallback"


def test_format_super_glue_prompt_cb():
    record = {"premise": "He said yes.", "hypothesis": "He agreed."}
    prompt = format_super_glue_prompt("cb", record)
    assert prompt == "Premise:\nHe said yes.\n\nHypothesis:\nHe agreed."


def test_format_super_glue_prompt_copa():
    record = {
        "premise": "The item was fragile.",
        "choice1": "It broke.",
        "choice2": "It survived.",
        "question": "effect",
    }
    prompt = format_super_glue_prompt("copa", record)
    assert "Premise:\nThe item was fragile." in prompt
    assert "(A) It broke." in prompt
    assert "(B) It survived." in prompt


def test_extract_super_glue_label():
    assert extract_super_glue_label("cb", {"label": 0}) == ("entailment", "exact_match")
    assert extract_super_glue_label("cb", {"label": 1}) == ("contradiction", "exact_match")
    assert extract_super_glue_label("cb", {"label": 2}) == ("neutral", "exact_match")
    assert extract_super_glue_label("cb", {"label": -1}) == (None, "exact_match")

    assert extract_super_glue_label("rte", {"label": 0}) == ("entailment", "exact_match")
    assert extract_super_glue_label("rte", {"label": 1}) == ("not_entailment", "exact_match")

    assert extract_super_glue_label("copa", {"label": 0}) == ("(A)", "multiple_choice")
    assert extract_super_glue_label("copa", {"label": 1}) == ("(B)", "multiple_choice")

    assert extract_super_glue_label("wic", {"label": 0}) == ("False", "exact_match")
    assert extract_super_glue_label("wic", {"label": 1}) == ("True", "exact_match")

    assert extract_super_glue_label("record", {"answers": ["Paris", "City of Light"]}) == ("Paris", "exact_match")
    assert extract_super_glue_label("record", {"answers": []}) == (None, "exact_match")


def test_super_glue_adapter_cb_conversion():
    adapter = SuperGlueAdapter(source_config="cb")
    record = {
        "premise": "Betti was in the kitchen.",
        "hypothesis": "Betti was inside.",
        "idx": 42,
        "label": 0,
    }
    item = adapter.convert_record(
        record,
        source_split="validation",
        source_id="0",
        benchmark_split="train",
    )

    assert item.prompt_id == "super_glue-cb-validation-42"
    assert "Premise:\nBetti was in the kitchen." in item.prompt
    assert item.task_type == "classification"
    assert item.domain == "super_glue_cb"
    assert item.difficulty == "medium"
    assert item.source_dataset == "super_glue"
    assert item.source_config == "cb"
    assert item.source_split == "validation"
    assert item.source_id == "42"
    assert item.split == "train"
    assert item.expected_output == "entailment"
    assert item.evaluation_type == "exact_match"


def test_super_glue_adapter_copa_conversion():
    adapter = SuperGlueAdapter(source_config="copa")
    record = {
        "premise": "The glass fell.",
        "choice1": "It shattered.",
        "choice2": "It flew.",
        "question": "result",
        "idx": 10,
        "label": 0,
    }
    item = adapter.convert_record(
        record,
        source_split="test",
        source_id="0",
        benchmark_split="test",
    )

    assert item.prompt_id == "super_glue-copa-test-10"
    assert item.evaluation_type == "multiple_choice"
    assert item.expected_output == "(A)"


def test_super_glue_adapter_multirc_conversion():
    adapter = SuperGlueAdapter(source_config="multirc")
    record = {
        "paragraph": "A story about a dog.",
        "question": "Was it a cat?",
        "answer": "No, it was a dog.",
        "idx": {"paragraph": 3, "question": 1, "answer": 4},
        "label": 1,
    }
    item = adapter.convert_record(
        record,
        source_split="train",
        source_id="0",
        benchmark_split="validation",
    )

    assert item.prompt_id == "super_glue-multirc-train-3-1-4"
    assert item.source_id == "3-1-4"
    assert item.expected_output == "True"
    assert item.evaluation_type == "exact_match"


def test_super_glue_adapter_record_conversion():
    adapter = SuperGlueAdapter(source_config="record")
    record = {
        "passage": "Marie Curie discovered radium.",
        "query": "@placeholder discovered radium.",
        "entities": ["Marie Curie"],
        "answers": ["Marie Curie"],
        "idx": {"passage": 5, "query": 2},
    }
    item = adapter.convert_record(
        record,
        source_split="validation",
        source_id="0",
        benchmark_split="test",
    )

    assert item.prompt_id == "super_glue-record-validation-5-2"
    assert item.source_id == "5-2"
    assert item.expected_output == "Marie Curie"
    assert item.evaluation_type == "exact_match"


def test_super_glue_adapter_missing_fields_rejected():
    adapter = SuperGlueAdapter(source_config="cb")

    with pytest.raises(ValueError, match="missing a valid 'premise'"):
        adapter.convert_record(
            {"hypothesis": "Something"},
            source_split="train",
            source_id="0",
            benchmark_split="train",
        )

    with pytest.raises(ValueError, match="missing a valid 'hypothesis'"):
        adapter.convert_record(
            {"premise": "Something"},
            source_split="train",
            source_id="0",
            benchmark_split="train",
        )


def test_super_glue_adapter_invalid_task_rejected():
    with pytest.raises(ValueError, match="Unsupported SuperGLUE task"):
        SuperGlueAdapter(source_config="invalid_task")

    with pytest.raises(ValueError, match="non-empty string"):
        SuperGlueAdapter(source_config="   ")


def test_super_glue_adapter_invalid_record_type_rejected():
    adapter = SuperGlueAdapter(source_config="cb")
    with pytest.raises(ValueError, match="must be a dictionary"):
        adapter.convert_record(
            "not a dict",  # type: ignore[arg-type]
            source_split="train",
            source_id="0",
            benchmark_split="train",
        )
