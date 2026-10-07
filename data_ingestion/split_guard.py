import re
from typing import Any
from collections import defaultdict

from router.benchmark_schema import BenchmarkPrompt


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and collapsing whitespace."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


class LeakageError(ValueError):
    """Raised when leakage or split contamination is detected."""
    pass


class SplitGuard:
    """
    Enforces anti-leakage, source quarantine, context isolation,
    and prompt deduplication rules across benchmark splits.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def check_source_id_quarantine(
        train_items: list[BenchmarkPrompt],
        val_items: list[BenchmarkPrompt],
        test_items: list[BenchmarkPrompt],
        stress_items: list[BenchmarkPrompt] | None = None,
    ) -> None:
        """
        Verify that no source (source_dataset, source_id) is shared across splits:
        - train ∩ validation = empty
        - train ∩ test = empty
        - validation ∩ test = empty
        - core ∩ stress = empty (for untransformed or original source IDs)
        """
        def get_keys(items: list[BenchmarkPrompt], use_original: bool = False) -> set[tuple[str, str]]:
            keys = set()
            for item in items:
                ds = item.original_source_dataset if (use_original and item.original_source_dataset) else item.source_dataset
                sid = item.original_source_id if (use_original and item.original_source_id) else item.source_id
                keys.add((ds, sid))
            return keys

        train_keys = get_keys(train_items)
        val_keys = get_keys(val_items)
        test_keys = get_keys(test_items)

        # 1. train vs val
        tv_overlap = train_keys & val_keys
        if tv_overlap:
            sample = list(tv_overlap)[:5]
            raise LeakageError(
                f"Source ID quarantine violation: {len(tv_overlap)} records overlap between train and validation. "
                f"Sample: {sample}"
            )

        # 2. train vs test
        tt_overlap = train_keys & test_keys
        if tt_overlap:
            sample = list(tt_overlap)[:5]
            raise LeakageError(
                f"Source ID quarantine violation: {len(tt_overlap)} records overlap between train and test. "
                f"Sample: {sample}"
            )

        # 3. val vs test
        vt_overlap = val_keys & test_keys
        if vt_overlap:
            sample = list(vt_overlap)[:5]
            raise LeakageError(
                f"Source ID quarantine violation: {len(vt_overlap)} records overlap between validation and test. "
                f"Sample: {sample}"
            )

        # 4. core vs stress
        if stress_items:
            core_keys = train_keys | val_keys | test_keys
            stress_keys = get_keys(stress_items, use_original=True)
            cs_overlap = core_keys & stress_keys
            if cs_overlap:
                sample = list(cs_overlap)[:5]
                raise LeakageError(
                    f"Core/Stress quarantine violation: {len(cs_overlap)} records overlap between core and stress. "
                    f"Sample: {sample}"
                )

    @staticmethod
    def check_prompt_deduplication(
        all_items: list[BenchmarkPrompt],
    ) -> None:
        """
        Verify that no duplicate prompts exist in the benchmark pool
        using normalized (lowercase, whitespace-collapsed) comparison.
        """
        seen_prompts: dict[str, str] = {}
        duplicates: list[tuple[str, str, str]] = []

        for item in all_items:
            norm_p = normalize_text(item.prompt)
            if norm_p in seen_prompts:
                duplicates.append((item.prompt_id, seen_prompts[norm_p], norm_p[:60]))
            else:
                seen_prompts[norm_p] = item.prompt_id

        if duplicates:
            sample = duplicates[:5]
            raise LeakageError(
                f"Prompt deduplication violation: {len(duplicates)} duplicate prompts detected. "
                f"Sample: {sample}"
            )

    @staticmethod
    def check_context_leakage(
        items_by_split: dict[str, list[dict[str, Any]]],
    ) -> None:
        """
        For datasets with contexts/passages (e.g. BoolQ, CNN/DailyMail, MultiRC),
        ensure that records sharing the exact same context reside strictly in the same split.

        items_by_split maps split_name -> list of dicts with at least 'context' and 'prompt_id'.
        """
        context_to_splits: dict[str, set[str]] = defaultdict(set)
        context_to_ids: dict[str, list[str]] = defaultdict(list)

        for split_name, records in items_by_split.items():
            for rec in records:
                ctx = rec.get("context")
                if not ctx or not isinstance(ctx, str) or len(ctx.strip()) < 30:
                    continue
                norm_ctx = normalize_text(ctx)
                context_to_splits[norm_ctx].add(split_name)
                context_to_ids[norm_ctx].append(rec.get("prompt_id", "unknown"))

        violating_contexts = {
            ctx: splits for ctx, splits in context_to_splits.items() if len(splits) > 1
        }

        if violating_contexts:
            first_ctx = next(iter(violating_contexts))
            raise LeakageError(
                f"Context leakage violation: {len(violating_contexts)} contexts span multiple splits. "
                f"Example spans {violating_contexts[first_ctx]} on prompt IDs: {context_to_ids[first_ctx][:5]}"
            )

    @classmethod
    def validate_all(
        cls,
        *,
        train: list[BenchmarkPrompt],
        val: list[BenchmarkPrompt],
        test: list[BenchmarkPrompt],
        stress: list[BenchmarkPrompt],
        context_records_by_split: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """Run all split and leakage validation checks."""
        # 1. Source ID quarantine
        cls.check_source_id_quarantine(train, val, test, stress)

        # 2. Prompt deduplication
        all_items = train + val + test + stress
        cls.check_prompt_deduplication(all_items)

        # 3. Context leakage
        if context_records_by_split:
            cls.check_context_leakage(context_records_by_split)

        return {
            "source_id_quarantine": "passed",
            "prompt_deduplication": "passed",
            "context_leakage": "passed" if context_records_by_split else "skipped",
            "total_prompts_checked": len(all_items),
        }
