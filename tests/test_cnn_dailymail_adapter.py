import pytest

from data_ingestion.cnn_dailymail import (
    CNNDailyMailAdapter,
    format_cnn_dailymail_prompt,
)


def test_format_cnn_dailymail_prompt():
    article = "A major storm hit the coast on Tuesday, causing power outages."
    prompt = format_cnn_dailymail_prompt(article)
    assert prompt == f"Summarize the following article:\n\n{article}"


def test_format_cnn_dailymail_prompt_invalid():
    with pytest.raises(ValueError, match="missing a valid 'article'"):
        format_cnn_dailymail_prompt("")

    with pytest.raises(ValueError, match="missing a valid 'article'"):
        format_cnn_dailymail_prompt("   ")


def test_cnn_dailymail_adapter_record_conversion():
    adapter = CNNDailyMailAdapter(source_config="3.0.0")

    record = {
        "id": "f001ec5c4704938247d27a44948eebb37ae98d01",
        "article": "The Palestinian Authority officially became the 123rd member of the ICC.",
        "highlights": "Membership gives the ICC jurisdiction over alleged crimes.\nIsrael and the US opposed.",
    }

    item = adapter.convert_record(
        record,
        source_split="test",
        source_id="0",
        benchmark_split="validation",
    )

    assert item.prompt_id == "cnn_dailymail-3.0.0-test-f001ec5c4704938247d27a44948eebb37ae98d01"
    assert "Summarize the following article:" in item.prompt
    assert "The Palestinian Authority officially became" in item.prompt
    # Model-facing prompt must not contain the reference summary
    assert "Membership gives the ICC jurisdiction" not in item.prompt

    assert item.task_type == "summarization"
    assert item.domain == "news"
    assert item.difficulty == "medium"
    assert item.source_dataset == "cnn_dailymail"
    assert item.source_config == "3.0.0"
    assert item.source_split == "test"
    assert item.source_id == "f001ec5c4704938247d27a44948eebb37ae98d01"
    assert item.split == "validation"
    assert item.expected_output == "Membership gives the ICC jurisdiction over alleged crimes.\nIsrael and the US opposed."
    assert item.evaluation_type == "reference_based"


def test_cnn_dailymail_adapter_fallback_to_source_id_if_id_missing():
    adapter = CNNDailyMailAdapter(source_config="3.0.0")

    record = {
        "article": "News content.",
        "highlights": "News summary.",
    }

    item = adapter.convert_record(
        record,
        source_split="train",
        source_id="99",
        benchmark_split="train",
    )

    assert item.prompt_id == "cnn_dailymail-3.0.0-train-99"
    assert item.source_id == "99"


def test_cnn_dailymail_adapter_missing_article_rejected():
    adapter = CNNDailyMailAdapter()

    with pytest.raises(ValueError, match="missing a valid 'article'"):
        adapter.convert_record(
            {"highlights": "Some summary."},
            source_split="test",
            source_id="0",
            benchmark_split="train",
        )

    with pytest.raises(ValueError, match="missing a valid 'article'"):
        adapter.convert_record(
            {"article": "   ", "highlights": "Some summary."},
            source_split="test",
            source_id="0",
            benchmark_split="train",
        )


def test_cnn_dailymail_adapter_missing_highlights_rejected():
    adapter = CNNDailyMailAdapter()

    with pytest.raises(ValueError, match="missing valid 'highlights'"):
        adapter.convert_record(
            {"article": "Some news."},
            source_split="test",
            source_id="0",
            benchmark_split="train",
        )

    with pytest.raises(ValueError, match="missing valid 'highlights'"):
        adapter.convert_record(
            {"article": "Some news.", "highlights": "   "},
            source_split="test",
            source_id="0",
            benchmark_split="train",
        )


def test_cnn_dailymail_adapter_invalid_record_type_rejected():
    adapter = CNNDailyMailAdapter()

    with pytest.raises(ValueError, match="must be a dictionary"):
        adapter.convert_record(
            "not a dict",  # type: ignore[arg-type]
            source_split="test",
            source_id="0",
            benchmark_split="train",
        )


def test_cnn_dailymail_adapter_invalid_config_rejected():
    with pytest.raises(ValueError, match="source_config must be a non-empty string"):
        CNNDailyMailAdapter(source_config="   ")
