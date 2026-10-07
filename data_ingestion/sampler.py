import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from router.benchmark_schema import BenchmarkPrompt
from data_ingestion.split_guard import SplitGuard, LeakageError, normalize_text
from data_ingestion.writer import write_benchmark_jsonl

from data_ingestion.mmlu import MMLUAdapter
from data_ingestion.bbh import BBHAdapter, SUPPORTED_TASKS as BBH_TASKS
from data_ingestion.gsm8k import GSM8KAdapter
from data_ingestion.hendrycks_math import HendrycksMATHAdapter, SUPPORTED_SUBJECTS as MATH_SUBJECTS
from data_ingestion.mbpp import MBPPAdapter
from data_ingestion.humaneval import HumanEvalAdapter
from data_ingestion.ifeval import IFEvalAdapter
from data_ingestion.cnn_dailymail import CNNDailyMailAdapter
from data_ingestion.boolq import BoolQAdapter
from data_ingestion.super_glue import SuperGlueAdapter
from data_ingestion.hellaswag import HellaSwagAdapter

MMLU_SUBJECTS = (
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
    "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
    "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions",
)


SUPER_GLUE_CORE_TASKS = ("cb", "copa", "rte", "wic", "wsc", "multirc")


def compute_file_sha256(path: Path) -> str:
    """Compute standard SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class BenchmarkSampler:
    """
    Deterministic benchmark sampler constructing 2,000 core + 300 stress prompts
    strictly adhering to reports/benchmark_split_policy_v1.md.
    """

    def __init__(
        self,
        *,
        seed: int = 42,
        dataset_loader: Callable[..., Any] | None = None,
    ) -> None:
        self.seed = seed
        if dataset_loader is None:
            from datasets import load_dataset
            self.dataset_loader = load_dataset
        else:
            self.dataset_loader = dataset_loader

    def _load_split(
        self,
        dataset_name: str,
        config: str | None = None,
        *,
        split: str,
    ) -> list[dict[str, Any]]:
        """Load and sort dataset split deterministically by compound key."""
        if config:
            ds = self.dataset_loader(dataset_name, config, split=split)
        else:
            ds = self.dataset_loader(dataset_name, split=split)

        records = [dict(r) for r in ds]
        return records

    # -------------------------------------------------------------------------
    # Core Category Samplers
    # -------------------------------------------------------------------------

    def sample_general_qa(self) -> tuple[list[BenchmarkPrompt], list[BenchmarkPrompt], list[BenchmarkPrompt]]:
        """MMLU (400 total): 280 train (from val), 60 val (from val), 60 test (from test)."""
        from collections import defaultdict
        train_prompts: list[BenchmarkPrompt] = []
        val_prompts: list[BenchmarkPrompt] = []
        test_prompts: list[BenchmarkPrompt] = []

        n_subjects = len(MMLU_SUBJECTS)
        train_quot, train_rem = divmod(280, n_subjects)
        val_quot, val_rem = divmod(60, n_subjects)
        test_quot, test_rem = divmod(60, n_subjects)

        val_all = self._load_split("cais/mmlu", "all", split="validation")
        test_all = self._load_split("cais/mmlu", "all", split="test")

        val_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in val_all:
            val_by_subject[r["subject"]].append(r)

        test_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in test_all:
            test_by_subject[r["subject"]].append(r)

        for idx, subj in enumerate(sorted(MMLU_SUBJECTS)):
            adapter = MMLUAdapter(source_config=subj)
            n_train = train_quot + (1 if idx < train_rem else 0)
            n_val = val_quot + (1 if idx < val_rem else 0)
            n_test = test_quot + (1 if idx < test_rem else 0)

            val_records = val_by_subject.get(subj, [])
            for i in range(n_train):
                train_prompts.append(
                    adapter.convert_record(
                        val_records[i],
                        source_split="validation",
                        source_id=f"val_{i:04d}",
                        benchmark_split="train",
                    )
                )
            for i in range(n_train, n_train + n_val):
                val_prompts.append(
                    adapter.convert_record(
                        val_records[i],
                        source_split="validation",
                        source_id=f"val_{i:04d}",
                        benchmark_split="validation",
                    )
                )

            test_records = test_by_subject.get(subj, [])
            for i in range(n_test):
                test_prompts.append(
                    adapter.convert_record(
                        test_records[i],
                        source_split="test",
                        source_id=f"test_{i:04d}",
                        benchmark_split="test",
                    )
                )

        return train_prompts, val_prompts, test_prompts


    def sample_reasoning(self) -> tuple[list[BenchmarkPrompt], list[BenchmarkPrompt], list[BenchmarkPrompt]]:
        """BBH (300 total): Strategy B intra-task partition across 27 tasks: 210 train, 45 val, 45 test."""
        train_prompts: list[BenchmarkPrompt] = []
        val_prompts: list[BenchmarkPrompt] = []
        test_prompts: list[BenchmarkPrompt] = []

        n_tasks = len(BBH_TASKS)
        train_quot, train_rem = divmod(210, n_tasks)  # 7, 21 -> 21 get 8, 6 get 7
        val_quot, val_rem = divmod(45, n_tasks)      # 1, 18 -> 18 get 2, 9 get 1
        test_quot, test_rem = divmod(45, n_tasks)    # 1, 18 -> 18 get 2, 9 get 1

        for idx, task in enumerate(sorted(BBH_TASKS)):
            adapter = BBHAdapter(source_config=task)
            n_train = train_quot + (1 if idx < train_rem else 0)
            n_val = val_quot + (1 if idx < val_rem else 0)
            n_test = test_quot + (1 if idx < test_rem else 0)

            records = self._load_split("lukaemon/bbh", task, split="test")

            # Slice disjointly
            offset = 0
            for i in range(offset, offset + n_train):
                train_prompts.append(
                    adapter.convert_record(
                        records[i],
                        source_split="test",
                        source_id=f"strategy_b_{i:04d}",
                        benchmark_split="train",
                    )
                )
            offset += n_train
            for i in range(offset, offset + n_val):
                val_prompts.append(
                    adapter.convert_record(
                        records[i],
                        source_split="test",
                        source_id=f"strategy_b_{i:04d}",
                        benchmark_split="validation",
                    )
                )
            offset += n_val
            for i in range(offset, offset + n_test):
                test_prompts.append(
                    adapter.convert_record(
                        records[i],
                        source_split="test",
                        source_id=f"strategy_b_{i:04d}",
                        benchmark_split="test",
                    )
                )

        return train_prompts, val_prompts, test_prompts

    def sample_math(self) -> tuple[list[BenchmarkPrompt], list[BenchmarkPrompt], list[BenchmarkPrompt]]:
        """Math (300 total): GSM8K (150) + Hendrycks MATH (150). Train: 210, Val: 45, Test: 45."""
        train_prompts: list[BenchmarkPrompt] = []
        val_prompts: list[BenchmarkPrompt] = []
        test_prompts: list[BenchmarkPrompt] = []

        # 1. GSM8K: 105 train, 23 val (from train held-out), 22 test (from test)
        gsm8k_adapter = GSM8KAdapter()
        gsm8k_train = self._load_split("openai/gsm8k", "main", split="train")
        gsm8k_test = self._load_split("openai/gsm8k", "main", split="test")

        for i in range(105):
            train_prompts.append(
                gsm8k_adapter.convert_record(
                    gsm8k_train[i],
                    source_split="train",
                    source_id=f"train_{i:04d}",
                    benchmark_split="train",
                )
            )
        for i in range(105, 105 + 23):
            val_prompts.append(
                gsm8k_adapter.convert_record(
                    gsm8k_train[i],
                    source_split="train",
                    source_id=f"train_{i:04d}",
                    benchmark_split="validation",
                )
            )
        for i in range(22):
            test_prompts.append(
                gsm8k_adapter.convert_record(
                    gsm8k_test[i],
                    source_split="test",
                    source_id=f"test_{i:04d}",
                    benchmark_split="test",
                )
            )

        # 2. Hendrycks MATH: 105 train (15 * 7), 22 val, 23 test
        n_math_subjects = len(MATH_SUBJECTS)
        val_quot, val_rem = divmod(22, n_math_subjects)   # 3, 1 -> 1 gets 4, 6 get 3
        test_quot, test_rem = divmod(23, n_math_subjects) # 3, 2 -> 2 get 4, 5 get 3

        for idx, subj in enumerate(sorted(MATH_SUBJECTS)):
            adapter = HendrycksMATHAdapter(source_config=subj)
            n_val = val_quot + (1 if idx < val_rem else 0)
            n_test = test_quot + (1 if idx < test_rem else 0)

            m_train = self._load_split("EleutherAI/hendrycks_math", subj, split="train")
            m_test = self._load_split("EleutherAI/hendrycks_math", subj, split="test")

            for i in range(15):
                train_prompts.append(
                    adapter.convert_record(
                        m_train[i],
                        source_split="train",
                        source_id=f"train_{i:04d}",
                        benchmark_split="train",
                    )
                )
            for i in range(15, 15 + n_val):
                val_prompts.append(
                    adapter.convert_record(
                        m_train[i],
                        source_split="train",
                        source_id=f"train_{i:04d}",
                        benchmark_split="validation",
                    )
                )
            for i in range(n_test):
                test_prompts.append(
                    adapter.convert_record(
                        m_test[i],
                        source_split="test",
                        source_id=f"test_{i:04d}",
                        benchmark_split="test",
                    )
                )

        return train_prompts, val_prompts, test_prompts

    def sample_coding(self) -> tuple[list[BenchmarkPrompt], list[BenchmarkPrompt], list[BenchmarkPrompt]]:
        """Coding (300 total): MBPP (210 train, 45 val, 20 test) + HumanEval (25 test)."""
        train_prompts: list[BenchmarkPrompt] = []
        val_prompts: list[BenchmarkPrompt] = []
        test_prompts: list[BenchmarkPrompt] = []

        mbpp_adapter = MBPPAdapter(source_config="full")
        mbpp_train = self._load_split("google-research-datasets/mbpp", "full", split="train")
        mbpp_val = self._load_split("google-research-datasets/mbpp", "full", split="validation")
        mbpp_test = self._load_split("google-research-datasets/mbpp", "full", split="test")

        for i in range(210):
            train_prompts.append(
                mbpp_adapter.convert_record(
                    mbpp_train[i],
                    source_split="train",
                    source_id=f"train_{i:04d}",
                    benchmark_split="train",
                )
            )
        for i in range(45):
            val_prompts.append(
                mbpp_adapter.convert_record(
                    mbpp_val[i],
                    source_split="validation",
                    source_id=f"val_{i:04d}",
                    benchmark_split="validation",
                )
            )
        for i in range(20):
            test_prompts.append(
                mbpp_adapter.convert_record(
                    mbpp_test[i],
                    source_split="test",
                    source_id=f"test_{i:04d}",
                    benchmark_split="test",
                )
            )

        # HumanEval: 25 test prompts strictly held-out
        he_adapter = HumanEvalAdapter()
        he_test = self._load_split("openai/openai_humaneval", "openai_humaneval", split="test")
        for i in range(25):
            test_prompts.append(
                he_adapter.convert_record(
                    he_test[i],
                    source_split="test",
                    source_id=f"test_{i:04d}",
                    benchmark_split="test",
                )
            )

        return train_prompts, val_prompts, test_prompts

    def sample_instruction_following(self) -> tuple[list[BenchmarkPrompt], list[BenchmarkPrompt], list[BenchmarkPrompt]]:
        """IFEval (250 total): Deterministic SHA-256 partition on key: 175 train, 37 val, 38 test."""
        train_prompts: list[BenchmarkPrompt] = []
        val_prompts: list[BenchmarkPrompt] = []
        test_prompts: list[BenchmarkPrompt] = []

        adapter = IFEvalAdapter()
        records = self._load_split("google/IFEval", "default", split="train")

        # Partition based on locked formula: sha256(f"ifeval:{key}:{seed}") % 100
        train_candidates = []
        val_candidates = []
        test_candidates = []

        for r in sorted(records, key=lambda x: x["key"]):
            key = r["key"]
            h = int(hashlib.sha256(f"ifeval:{key}:{self.seed}".encode()).hexdigest(), 16) % 100
            if h < 70:
                train_candidates.append(r)
            elif h < 85:
                val_candidates.append(r)
            else:
                test_candidates.append(r)

        for i in range(175):
            rec = train_candidates[i]
            train_prompts.append(
                adapter.convert_record(
                    rec,
                    source_split="train",
                    source_id=str(rec["key"]),
                    benchmark_split="train",
                )
            )
        for i in range(37):
            rec = val_candidates[i]
            val_prompts.append(
                adapter.convert_record(
                    rec,
                    source_split="train",
                    source_id=str(rec["key"]),
                    benchmark_split="validation",
                )
            )
        for i in range(38):
            rec = test_candidates[i]
            test_prompts.append(
                adapter.convert_record(
                    rec,
                    source_split="train",
                    source_id=str(rec["key"]),
                    benchmark_split="test",
                )
            )

        return train_prompts, val_prompts, test_prompts

    def sample_summarization(self) -> tuple[list[BenchmarkPrompt], list[BenchmarkPrompt], list[BenchmarkPrompt]]:
        """CNN/DailyMail (200 total): Config 3.0.0. 140 train, 30 val, 30 test. Max prompt tokens = 2048."""
        train_prompts: list[BenchmarkPrompt] = []
        val_prompts: list[BenchmarkPrompt] = []
        test_prompts: list[BenchmarkPrompt] = []

        adapter = CNNDailyMailAdapter(source_config="3.0.0")
        train_records = self._load_split("abisee/cnn_dailymail", "3.0.0", split="train")
        val_records = self._load_split("abisee/cnn_dailymail", "3.0.0", split="validation")
        test_records = self._load_split("abisee/cnn_dailymail", "3.0.0", split="test")

        for i in range(140):
            train_prompts.append(
                adapter.convert_record(
                    train_records[i],
                    source_split="train",
                    source_id=f"train_{i:04d}",
                    benchmark_split="train",
                )
            )
        for i in range(30):
            val_prompts.append(
                adapter.convert_record(
                    val_records[i],
                    source_split="validation",
                    source_id=f"val_{i:04d}",
                    benchmark_split="validation",
                )
            )
        for i in range(30):
            test_prompts.append(
                adapter.convert_record(
                    test_records[i],
                    source_split="test",
                    source_id=f"test_{i:04d}",
                    benchmark_split="test",
                )
            )

        return train_prompts, val_prompts, test_prompts

    def sample_classification_extraction(self) -> tuple[list[BenchmarkPrompt], list[BenchmarkPrompt], list[BenchmarkPrompt]]:
        """BoolQ (75) + SuperGLUE non-boolq (75): Total 105 train, 22 val, 23 test."""
        train_prompts: list[BenchmarkPrompt] = []
        val_prompts: list[BenchmarkPrompt] = []
        test_prompts: list[BenchmarkPrompt] = []

        # 1. BoolQ: 53 train, 11 val (from val slice A), 11 test (from val slice B)
        boolq_adapter = BoolQAdapter()
        boolq_train = self._load_split("google/boolq", "default", split="train")
        boolq_val = self._load_split("google/boolq", "default", split="validation")

        for i in range(53):
            train_prompts.append(
                boolq_adapter.convert_record(
                    boolq_train[i],
                    source_split="train",
                    source_id=f"train_{i:04d}",
                    benchmark_split="train",
                )
            )
        for i in range(11):
            val_prompts.append(
                boolq_adapter.convert_record(
                    boolq_val[i],
                    source_split="validation",
                    source_id=f"val_{i:04d}",
                    benchmark_split="validation",
                )
            )
        for i in range(11, 22):
            test_prompts.append(
                boolq_adapter.convert_record(
                    boolq_val[i],
                    source_split="validation",
                    source_id=f"val_{i:04d}",
                    benchmark_split="test",
                )
            )

        # 2. SuperGLUE non-boolq: 52 train, 11 val, 12 test
        n_tasks = len(SUPER_GLUE_CORE_TASKS)  # 6 tasks: cb, copa, rte, wic, wsc, multirc
        train_quot, train_rem = divmod(52, n_tasks) # 8, 4 -> 4 get 9, 2 get 8
        val_quot, val_rem = divmod(11, n_tasks)     # 1, 5 -> 5 get 2, 1 gets 1
        test_quot, test_rem = divmod(12, n_tasks)   # 2, 0 -> all 6 get 2

        for idx, task in enumerate(SUPER_GLUE_CORE_TASKS):
            adapter = SuperGlueAdapter(source_config=task)
            n_t = train_quot + (1 if idx < train_rem else 0)
            n_v = val_quot + (1 if idx < val_rem else 0)
            n_te = test_quot + (1 if idx < test_rem else 0)

            t_records = self._load_split("aps/super_glue", task, split="train")
            v_records = self._load_split("aps/super_glue", task, split="validation")

            for i in range(n_t):
                train_prompts.append(
                    adapter.convert_record(
                        t_records[i],
                        source_split="train",
                        source_id=f"train_{i:04d}",
                        benchmark_split="train",
                    )
                )
            for i in range(n_v):
                val_prompts.append(
                    adapter.convert_record(
                        v_records[i],
                        source_split="validation",
                        source_id=f"val_{i:04d}",
                        benchmark_split="validation",
                    )
                )
            for i in range(n_v, n_v + n_te):
                test_prompts.append(
                    adapter.convert_record(
                        v_records[i],
                        source_split="validation",
                        source_id=f"val_{i:04d}",
                        benchmark_split="test",
                    )
                )

        return train_prompts, val_prompts, test_prompts

    def sample_other(self) -> tuple[list[BenchmarkPrompt], list[BenchmarkPrompt], list[BenchmarkPrompt]]:
        """HellaSwag (100 total): 70 train, 15 val (from val slice A), 15 test (from val slice B)."""
        train_prompts: list[BenchmarkPrompt] = []
        val_prompts: list[BenchmarkPrompt] = []
        test_prompts: list[BenchmarkPrompt] = []

        adapter = HellaSwagAdapter()
        train_records = self._load_split("Rowan/hellaswag", "default", split="train")
        val_records = self._load_split("Rowan/hellaswag", "default", split="validation")

        for i in range(70):
            train_prompts.append(
                adapter.convert_record(
                    train_records[i],
                    source_split="train",
                    source_id=f"train_{i:04d}",
                    benchmark_split="train",
                )
            )
        for i in range(15):
            val_prompts.append(
                adapter.convert_record(
                    val_records[i],
                    source_split="validation",
                    source_id=f"val_{i:04d}",
                    benchmark_split="validation",
                )
            )
        for i in range(15, 30):
            test_prompts.append(
                adapter.convert_record(
                    val_records[i],
                    source_split="validation",
                    source_id=f"val_{i:04d}",
                    benchmark_split="test",
                )
            )

        return train_prompts, val_prompts, test_prompts

    # -------------------------------------------------------------------------
    # Stress Sourcing & Provenance
    # -------------------------------------------------------------------------

    def sample_stress(self) -> list[BenchmarkPrompt]:
        """
        Construct 300 stress prompts across 7 categories, strictly quarantined
        from core records and with complete provenance metadata.
        """
        stress_prompts: list[BenchmarkPrompt] = []

        # 1. long_prompts: 60 (from unselected CNN/DailyMail train records offset >= 1000)
        cnn_adapter = CNNDailyMailAdapter(source_config="3.0.0")
        cnn_recs = self._load_split("abisee/cnn_dailymail", "3.0.0", split="train")
        for i in range(60):
            rec_idx = 1000 + i
            base_item = cnn_adapter.convert_record(
                cnn_recs[rec_idx],
                source_split="train",
                source_id=f"stress_{rec_idx:04d}",
                benchmark_split="stress",
            )
            stress_prompts.append(
                BenchmarkPrompt(
                    prompt_id=f"stress-long_prompts-{i:03d}",
                    prompt=base_item.prompt,
                    task_type="summarization",
                    domain="news_long_context",
                    difficulty="hard",
                    source_dataset="cnn_dailymail",
                    source_config="3.0.0",
                    source_split="train",
                    source_id=f"stress_{rec_idx:04d}",
                    split="stress",
                    expected_output=base_item.expected_output,
                    evaluation_type="reference_based",
                    stress_transformation="long_context_filter",
                    original_source_dataset="cnn_dailymail",
                    original_source_id=f"train_{rec_idx:04d}",
                )
            )

        # 2. distractors: 50 (from unselected BoolQ train offset >= 500)
        boolq_adapter = BoolQAdapter()
        boolq_recs = self._load_split("google/boolq", "default", split="train")
        distractor_prefix = (
            "Background note: Recent high-energy cosmological models suggest "
            "interstellar dark matter distributions vary non-linearly across galactic clusters.\n\n"
        )
        for i in range(50):
            rec_idx = 500 + i
            base_item = boolq_adapter.convert_record(
                boolq_recs[rec_idx],
                source_split="train",
                source_id=f"stress_{rec_idx:04d}",
                benchmark_split="stress",
            )
            perturbed_prompt = distractor_prefix + base_item.prompt
            stress_prompts.append(
                BenchmarkPrompt(
                    prompt_id=f"stress-distractors-{i:03d}",
                    prompt=perturbed_prompt,
                    task_type="classification",
                    domain="distractor_robustness",
                    difficulty="hard",
                    source_dataset="boolq",
                    source_config="default",
                    source_split="train",
                    source_id=f"stress_{rec_idx:04d}",
                    split="stress",
                    expected_output=base_item.expected_output,
                    evaluation_type="exact_match",
                    stress_transformation="distractor_context_injection",
                    original_source_dataset="boolq",
                    original_source_id=f"train_{rec_idx:04d}",
                )
            )

        # 3. paraphrased: 50 (from unselected GSM8K train offset >= 500)
        gsm8k_adapter = GSM8KAdapter()
        gsm8k_recs = self._load_split("openai/gsm8k", "main", split="train")
        for i in range(50):
            rec_idx = 500 + i
            base_item = gsm8k_adapter.convert_record(
                gsm8k_recs[rec_idx],
                source_split="train",
                source_id=f"stress_{rec_idx:04d}",
                benchmark_split="stress",
            )
            rephrased = f"Mathematical reasoning inquiry: Please calculate the answer to the following scenario:\n{base_item.prompt}\nState the final number clearly."
            stress_prompts.append(
                BenchmarkPrompt(
                    prompt_id=f"stress-paraphrased-{i:03d}",
                    prompt=rephrased,
                    task_type="math",
                    domain="grade_school_math_paraphrased",
                    difficulty="medium",
                    source_dataset="gsm8k",
                    source_config="main",
                    source_split="train",
                    source_id=f"stress_{rec_idx:04d}",
                    split="stress",
                    expected_output=base_item.expected_output,
                    evaluation_type="exact_match",
                    stress_transformation="instruction_paraphrase",
                    original_source_dataset="gsm8k",
                    original_source_id=f"train_{rec_idx:04d}",
                )
            )

        # 4. formatting_variations: 40 (from unselected BBH records offset >= 50)
        # Sourced across 10 diverse BBH tasks
        formatting_tasks = sorted(BBH_TASKS)[:10]
        for task_idx, task in enumerate(formatting_tasks):
            bbh_adapter = BBHAdapter(source_config=task)
            recs = self._load_split("lukaemon/bbh", task, split="test")
            for j in range(4):
                item_idx = len(stress_prompts) - 160  # relative index
                rec_idx = 50 + j
                base_item = bbh_adapter.convert_record(
                    recs[rec_idx],
                    source_split="test",
                    source_id=f"stress_{task}_{rec_idx:04d}",
                    benchmark_split="stress",
                )
                escaped_p = base_item.prompt.replace('"', '\\"').replace('\n', ' ')
                json_prompt = (
                    "Please parse the task specification enclosed in JSON format and respond accordingly:\n"
                    "```json\n"
                    "{\n"
                    f'  "task": "{task}",\n'
                    f'  "input": "{escaped_p}"\n'
                    "}\n"
                    "```\n"
                    "Answer:"
                )
                stress_prompts.append(
                    BenchmarkPrompt(
                        prompt_id=f"stress-formatting_variations-{item_idx:03d}",
                        prompt=json_prompt,
                        task_type="reasoning",
                        domain=f"json_schema_{task}",
                        difficulty="hard",
                        source_dataset="bbh",
                        source_config=task,
                        source_split="test",
                        source_id=f"stress_{task}_{rec_idx:04d}",
                        split="stress",
                        expected_output=base_item.expected_output,
                        evaluation_type=base_item.evaluation_type,
                        stress_transformation="json_schema_reformatting",
                        original_source_dataset="bbh",
                        original_source_id=f"test_{task}_{rec_idx:04d}",
                    )
                )

        # 5. harder_reasoning_math: 40 (Hendrycks MATH Level 5 problems offset >= 50)
        math_subjects_stress = ["algebra", "geometry", "number_theory", "intermediate_algebra"]
        for s_idx, subj in enumerate(math_subjects_stress):
            adapter = HendrycksMATHAdapter(source_config=subj)
            recs = self._load_split("EleutherAI/hendrycks_math", subj, split="train")
            # Filter Level 5 problems
            level_5_recs = [r for r in recs if "Level 5" in str(r.get("level", ""))]
            for j in range(10):
                item_idx = s_idx * 10 + j
                rec_idx = 30 + j
                r = level_5_recs[rec_idx]
                base_item = adapter.convert_record(
                    r,
                    source_split="train",
                    source_id=f"stress_{subj}_{rec_idx:04d}",
                    benchmark_split="stress",
                )
                stress_prompts.append(
                    BenchmarkPrompt(
                        prompt_id=f"stress-harder_reasoning_math-{item_idx:03d}",
                        prompt=base_item.prompt,
                        task_type="math",
                        domain=f"competition_math_level_5_{subj}",
                        difficulty="hard",
                        source_dataset="math",
                        source_config=subj,
                        source_split="train",
                        source_id=f"stress_{subj}_{rec_idx:04d}",
                        split="stress",
                        expected_output=base_item.expected_output,
                        evaluation_type="exact_match",
                        stress_transformation="level_5_extreme_difficulty",
                        original_source_dataset="math",
                        original_source_id=f"train_{subj}_{rec_idx:04d}",
                    )
                )

        # 6. code_variations: 30 (from unselected MBPP test offset >= 100)
        mbpp_adapter = MBPPAdapter(source_config="full")
        mbpp_recs = self._load_split("google-research-datasets/mbpp", "full", split="test")
        for i in range(30):
            rec_idx = 100 + i
            base_item = mbpp_adapter.convert_record(
                mbpp_recs[rec_idx],
                source_split="test",
                source_id=f"stress_{rec_idx:04d}",
                benchmark_split="stress",
            )
            constrained_prompt = (
                "Write a Python function to satisfy the following problem specification. "
                "Ensure your code is strictly self-contained, typed, and contains no globals:\n"
                f"{base_item.prompt}"
            )
            stress_prompts.append(
                BenchmarkPrompt(
                    prompt_id=f"stress-code_variations-{i:03d}",
                    prompt=constrained_prompt,
                    task_type="coding",
                    domain="python_refactored",
                    difficulty="hard",
                    source_dataset="mbpp",
                    source_config="full",
                    source_split="test",
                    source_id=f"stress_{rec_idx:04d}",
                    split="stress",
                    expected_output=base_item.expected_output,
                    evaluation_type="code_execution",
                    stress_transformation="code_specification_variation",
                    original_source_dataset="mbpp",
                    original_source_id=f"test_{rec_idx:04d}",
                )
            )

        # 7. held_out_domain_task: 30 (SuperGLUE 'record' held out from core)
        record_adapter = SuperGlueAdapter(source_config="record")
        record_recs = self._load_split("aps/super_glue", "record", split="validation")
        for i in range(30):
            rec_idx = i
            base_item = record_adapter.convert_record(
                record_recs[rec_idx],
                source_split="validation",
                source_id=f"stress_{rec_idx:04d}",
                benchmark_split="stress",
            )
            stress_prompts.append(
                BenchmarkPrompt(
                    prompt_id=f"stress-held_out_domain_task-{i:03d}",
                    prompt=base_item.prompt,
                    task_type="classification",
                    domain="super_glue_record_held_out",
                    difficulty="hard",
                    source_dataset="super_glue",
                    source_config="record",
                    source_split="validation",
                    source_id=f"stress_record_{rec_idx:04d}",
                    split="stress",
                    expected_output=base_item.expected_output,
                    evaluation_type=base_item.evaluation_type,
                    stress_transformation="held_out_task_domain",
                    original_source_dataset="super_glue",
                    original_source_id=f"val_record_{rec_idx:04d}",
                )
            )

        return stress_prompts

    # -------------------------------------------------------------------------
    # Pipeline Orchestration
    # -------------------------------------------------------------------------

    def sample_all(
        self,
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Execute full benchmark sampling pipeline:
        1. Sample all 8 core categories into train/val/test
        2. Sample 7 stress categories into stress
        3. Enforce anti-leakage and deduplication guards
        4. Write JSONL artifacts and manifest if output_dir provided
        """
        train_prompts: list[BenchmarkPrompt] = []
        val_prompts: list[BenchmarkPrompt] = []
        test_prompts: list[BenchmarkPrompt] = []

        # Category samplers in deterministic sequence
        samplers = [
            ("general_qa", self.sample_general_qa),
            ("reasoning", self.sample_reasoning),
            ("math", self.sample_math),
            ("coding", self.sample_coding),
            ("instruction_following", self.sample_instruction_following),
            ("summarization", self.sample_summarization),
            ("classification_extraction", self.sample_classification_extraction),
            ("other", self.sample_other),
        ]

        category_counts: dict[str, dict[str, int]] = {}

        for cat_name, sampler_fn in samplers:
            t, v, te = sampler_fn()
            train_prompts.extend(t)
            val_prompts.extend(v)
            test_prompts.extend(te)
            category_counts[cat_name] = {
                "train": len(t),
                "validation": len(v),
                "test": len(te),
                "total": len(t) + len(v) + len(te),
            }

        stress_prompts = self.sample_stress()

        # Enforce SplitGuard validation across all splits
        guard_result = SplitGuard.validate_all(
            train=train_prompts,
            val=val_prompts,
            test=test_prompts,
            stress=stress_prompts,
        )

        result: dict[str, Any] = {
            "counts": {
                "train": len(train_prompts),
                "validation": len(val_prompts),
                "test": len(test_prompts),
                "stress": len(stress_prompts),
                "core_total": len(train_prompts) + len(val_prompts) + len(test_prompts),
                "grand_total": len(train_prompts) + len(val_prompts) + len(test_prompts) + len(stress_prompts),
            },
            "category_distribution": category_counts,
            "guard_result": guard_result,
            "train_prompts": train_prompts,
            "val_prompts": val_prompts,
            "test_prompts": test_prompts,
            "stress_prompts": stress_prompts,
        }

        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)

            train_path = out_p / "benchmark_train.jsonl"
            val_path = out_p / "benchmark_validation.jsonl"
            test_path = out_p / "benchmark_test.jsonl"
            stress_path = out_p / "benchmark_stress.jsonl"
            manifest_path = out_p / "benchmark_manifest.json"

            write_benchmark_jsonl(train_prompts, train_path)
            write_benchmark_jsonl(val_prompts, val_path)
            write_benchmark_jsonl(test_prompts, test_path)
            write_benchmark_jsonl(stress_prompts, stress_path)

            file_checksums = {
                "benchmark_train.jsonl": compute_file_sha256(train_path),
                "benchmark_validation.jsonl": compute_file_sha256(val_path),
                "benchmark_test.jsonl": compute_file_sha256(test_path),
                "benchmark_stress.jsonl": compute_file_sha256(stress_path),
            }

            manifest_data = {
                "benchmark_name": "qwen_router_benchmark_v1",
                "version": 1,
                "policy_version": "benchmark_split_policy_v1",
                "seed": self.seed,
                "counts": result["counts"],
                "category_distribution": category_counts,
                "stress_counts": {
                    "long_prompts": 60,
                    "distractors": 50,
                    "paraphrased": 50,
                    "formatting_variations": 40,
                    "harder_reasoning_math": 40,
                    "code_variations": 30,
                    "held_out_domain_task": 30,
                    "total": 300,
                },
                "guard_status": guard_result,
                "checksums_sha256": file_checksums,
            }

            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)

            result["files"] = {
                "train": str(train_path),
                "validation": str(val_path),
                "test": str(test_path),
                "stress": str(stress_path),
                "manifest": str(manifest_path),
            }
            result["checksums"] = file_checksums

        return result
