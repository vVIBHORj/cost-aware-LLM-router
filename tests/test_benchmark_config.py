from pathlib import Path

import pytest

from router.benchmark_config import load_benchmark_config


CONFIG_PATH = Path("configs/benchmark.yaml")


def test_benchmark_config_loads():
    config = load_benchmark_config(CONFIG_PATH)

    assert config.benchmark.name == "qwen_router_benchmark_v1"
    assert config.benchmark.version == 1
    assert config.benchmark.random_seed == 42


def test_benchmark_totals():
    config = load_benchmark_config(CONFIG_PATH)

    assert config.benchmark.total_prompts == 2300
    assert config.benchmark.core_prompts == 2000
    assert config.benchmark.stress_prompts == 300

    assert config.core.total == 2000
    assert config.stress.total == 300


def test_core_category_counts():
    config = load_benchmark_config(CONFIG_PATH)

    categories = {
        category.name: category.count
        for category in config.core.categories
    }

    assert categories == {
        "general_qa": 400,
        "reasoning": 300,
        "math": 300,
        "coding": 300,
        "instruction_following": 250,
        "summarization": 200,
        "classification_extraction": 150,
        "other": 100,
    }


def test_core_split_counts():
    config = load_benchmark_config(CONFIG_PATH)

    assert config.core.split.train == 1400
    assert config.core.split.validation == 300
    assert config.core.split.test == 300


def test_stress_counts():
    config = load_benchmark_config(CONFIG_PATH)

    stress_types = {
        stress_type.name: stress_type.count
        for stress_type in config.stress.types
    }

    assert stress_types == {
        "long_prompts": 60,
        "distractors": 50,
        "paraphrased": 50,
        "formatting_variations": 40,
        "harder_reasoning_math": 40,
        "code_variations": 30,
        "held_out_domain_task": 30,
    }


def test_stress_is_held_out():
    config = load_benchmark_config(CONFIG_PATH)

    assert config.stress.held_out is True
    assert config.stress.split["stress"] == 300


def test_prompt_level_leakage_guard():
    config = load_benchmark_config(CONFIG_PATH)

    assert config.splitting.unit == "prompt"
    assert config.splitting.leakage_guard is True

    assert set(config.splitting.allowed_splits) == {
        "train",
        "validation",
        "test",
        "stress",
    }


def test_candidate_models():
    config = load_benchmark_config(CONFIG_PATH)

    assert [model.name for model in config.benchmark.candidate_models] == [
        "qwen3-1.7b",
        "qwen3-4b",
        "qwen3-8b",
    ]


def test_source_datasets():
    config = load_benchmark_config(CONFIG_PATH)

    assert [source.name for source in config.source_datasets] == [
        "mmlu",
        "gsm8k",
        "math",
        "humaneval",
        "mbpp",
        "bbh",
        "ifeval",
        "cnn_dailymail",
        "boolq",
        "super_glue",
        "hellaswag",
    ]


def test_sampling_configuration():
    config = load_benchmark_config(CONFIG_PATH)

    assert config.sampling.strategy == "stratified"
    assert config.sampling.stratify_by == [
        "category",
        "difficulty",
    ]
    assert config.sampling.dedup is True
    assert config.sampling.shuffle is True


def test_invalid_core_total_is_rejected(tmp_path):
    path = tmp_path / "invalid.yaml"

    path.write_text(
        """
benchmark:
  name: test
  version: 1
  random_seed: 42
  total_prompts: 2300
  core_prompts: 1999
  stress_prompts: 300
  candidate_models:
    - name: qwen3-1.7b

core:
  total: 2000
  categories:
    - name: test
      label: Test
      count: 2000
      sources:
        - test
  difficulty_levels:
    - easy
  difficulty_strategy: stratified
  split:
    train: 1400
    validation: 300
    test: 300

stress:
  total: 300
  held_out: true
  types:
    - name: test
      label: Test
      count: 300
  split:
    stress: 300

splitting:
  unit: prompt
  leakage_guard: true
  allowed_splits:
    - train
    - validation
    - test
    - stress

source_datasets:
  - name: test
    version: null
    config: null
    covers:
      - test

reference_only: []

sampling:
  strategy: stratified
  stratify_by:
    - category
  dedup: true
  max_prompt_tokens: null
  shuffle: true

provenance:
  required_fields:
    - prompt_id

evaluation:
  allowed_types:
    - exact_match
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_benchmark_config(path)


def test_undefined_source_is_rejected(tmp_path):
    path = tmp_path / "invalid.yaml"

    path.write_text(
        """
benchmark:
  name: test
  version: 1
  random_seed: 42
  total_prompts: 101
  core_prompts: 100
  stress_prompts: 1
  candidate_models:
    - name: qwen3-1.7b

core:
  total: 100
  categories:
    - name: qa
      label: QA
      count: 100
      sources:
        - nonexistent_dataset
  difficulty_levels:
    - easy
  difficulty_strategy: stratified
  split:
    train: 70
    validation: 15
    test: 15

stress:
  total: 1
  held_out: true
  types:
    - name: test
      label: Test
      count: 1
  split:
    stress: 1

splitting:
  unit: prompt
  leakage_guard: true
  allowed_splits:
    - train
    - validation
    - test
    - stress

source_datasets:
  - name: mmlu
    version: null
    config: null
    covers:
      - general_qa

reference_only: []

sampling:
  strategy: stratified
  stratify_by:
    - category
  dedup: true
  max_prompt_tokens: null
  shuffle: true

provenance:
  required_fields:
    - prompt_id

evaluation:
  allowed_types:
    - exact_match
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="undefined datasets"):
        load_benchmark_config(path)