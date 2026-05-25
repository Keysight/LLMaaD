from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List


class ResultWriter:
    """
    Handles:
    1. Writing one result JSON per prompt
    2. Writing aggregate average scores across all results
    """

    def __init__(self, base_dir: str | Path = "all_results") -> None:
        self.base_dir = Path(base_dir)

    # ---------------------------------------------------------
    # Generic converters
    # ---------------------------------------------------------
    def _to_dict(self, obj: Any) -> Dict[str, Any]:
        """
        Convert dataclass / dict / object into serializable dict.
        """
        if obj is None:
            return {}

        if isinstance(obj, dict):
            return obj

        if is_dataclass(obj):
            return asdict(obj)

        if hasattr(obj, "__dict__"):
            return vars(obj)

        raise TypeError(f"Unsupported object type for serialization: {type(obj)}")

    def _safe_float(self, value: Any) -> float | None:
        """
        Convert score values safely to float where possible.
        Handles ints/floats and ignores tuples/None/non-numeric types.
        """
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        return None

    def _clean_dict(self, obj):
        """
        Recursively remove None, empty strings, empty lists and empty dicts.
        """
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                v = self._clean_dict(v)
                if v not in (None, "", [], {}, False):
                    cleaned[k] = v
            return cleaned

        elif isinstance(obj, list):
            cleaned_list = [self._clean_dict(v) for v in obj]
            return [v for v in cleaned_list if v not in (None, "", [], {})]

        else:
            return obj

    # ---------------------------------------------------------
    # Path helpers
    # ---------------------------------------------------------
    def _build_prompt_result_dir(self, inp: Any, algo_name: str) -> Path:
        dataset_name = getattr(inp, "dataset_name", "dataset")
        run_dir = getattr(inp, "run_dir", "run")
        out_dir = self.base_dir / algo_name / f"{run_dir}_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _build_summary_dir(self, algo_name: str) -> Path:
        out_dir = self.base_dir / algo_name / "summary"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # ---------------------------------------------------------
    # 1) Write one result per prompt
    # ---------------------------------------------------------
    def write_prompt_result(self, inp: Any, result_obj: Any, algo_name: str, file_stem: str = None) -> Path:
        """
        Writes one JSON per prompt/result.
        file_stem: if provided, used as the filename base (shared with the log file).
        """
        out_dir = self._build_prompt_result_dir(inp, algo_name)

        if file_stem is None:
            file_stem = f"final_{os.getpid()}_{time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())}"

        payload = {
            "input": self._to_dict(inp),
            "result": self._to_dict(result_obj),
        }
        payload = self._clean_dict(payload)
        file_path = out_dir / f"{file_stem}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

        return file_path

    # ---------------------------------------------------------
    # Score extraction helpers
    # ---------------------------------------------------------
    def _extract_scores_from_result(self, result_obj: Any) -> Dict[str, float | None]:
        """
        Tries to extract judge scores from a result object.

        Expected flexible shapes:
        - result.initial_scores / result.final_scores
        - dict["initial_scores"] / dict["final_scores"]

        Each score block may contain fields like:
            judge_llm
            pair_gpt_judge_score
            gcg_judge
            strong_reject   <- ignored for average unless numeric
        """
        result_dict = self._to_dict(result_obj)

        initial_scores = result_dict.get("initial_scores", {}) or {}
        final_scores = result_dict.get("final_scores", {}) or {}

        return {
            "initial_judge_llm": self._safe_float(initial_scores.get("judge_llm")),
            "initial_pair_gpt_judge": self._safe_float(initial_scores.get("pair_gpt_judge_score")),
        "initial_gcg_judge": self._safe_float(initial_scores.get("gcg_judge")),
            "final_judge_llm": self._safe_float(final_scores.get("judge_llm")),
            "final_pair_gpt_judge": self._safe_float(final_scores.get("pair_gpt_judge_score")),
            "final_gcg_judge": self._safe_float(final_scores.get("gcg_judge")),
        }

    def _average(self, values: List[float | None]) -> float | None:
        clean_vals = [v for v in values if v is not None]
        if not clean_vals:
            return None
        return sum(clean_vals) / len(clean_vals)

    # ---------------------------------------------------------
    # 2) Write average summary for all prompts
    # ---------------------------------------------------------
    def write_average_summary(
        self,
        results: List[Any],
        algo_name: str,
        summary_name: str = "average_scores",
    ) -> Path:

        extracted = [self._extract_scores_from_result(r) for r in results]

        # ----------------------------
        # normal judge scores
        # ----------------------------
        initial_judge_llm_vals = []
        initial_pair_gpt_judge_vals = []
        initial_gcg_judge_vals = []

        final_judge_llm_vals = []
        final_pair_gpt_judge_vals = []
        final_gcg_judge_vals = []

        # ----------------------------
        # strong reject lists
        # ----------------------------
        sr1_vals = []
        sr2_vals = []
        sr3_vals = []

        for r in results:

            r_dict = self._to_dict(r)

            initial_scores = r_dict.get("initial_scores", {}) or {}
            final_scores = r_dict.get("final_scores", {}) or {}

            # --------------------------------
            # standard judges
            # --------------------------------
            initial_judge_llm_vals.append(
                self._safe_float(initial_scores.get("judge_llm"))
            )
            initial_pair_gpt_judge_vals.append(
                self._safe_float(initial_scores.get("pair_gpt_judge_score"))
            )
            initial_gcg_judge_vals.append(
                self._safe_float(initial_scores.get("gcg_judge"))
            )

            final_judge_llm_vals.append(
                self._safe_float(final_scores.get("judge_llm"))
            )
            final_pair_gpt_judge_vals.append(
                self._safe_float(final_scores.get("pair_gpt_judge_score"))
            )
            final_gcg_judge_vals.append(
                self._safe_float(final_scores.get("gcg_judge"))
            )

            # --------------------------------
            # strong reject handling
            # --------------------------------
            sr = final_scores.get("strong_reject")

            if isinstance(sr, (list, tuple)) and len(sr) == 3:

                sr1_vals.append(self._safe_float(sr[0]))
                sr2_vals.append(self._safe_float(sr[1]))
                sr3_vals.append(self._safe_float(sr[2]))

        # ----------------------------
        # averages
        # ----------------------------
        avg_initial_judge_llm = self._average(initial_judge_llm_vals)
        avg_initial_pair_gpt_judge = self._average(initial_pair_gpt_judge_vals)
        avg_initial_gcg_judge = self._average(initial_gcg_judge_vals)

        avg_final_judge_llm = self._average(final_judge_llm_vals)
        avg_final_pair_gpt_judge = self._average(final_pair_gpt_judge_vals)
        avg_final_gcg_judge = self._average(final_gcg_judge_vals)

        avg_sr1 = self._average(sr1_vals)
        avg_sr2 = self._average(sr2_vals)
        avg_sr3 = self._average(sr3_vals)

        # overall averages of initial + final
        avg_judge_llm_overall = self._average(
            [avg_initial_judge_llm, avg_final_judge_llm]
        )
        avg_pair_gpt_judge_overall = self._average(
            [avg_initial_pair_gpt_judge, avg_final_pair_gpt_judge]
        )
        avg_gcg_judge_overall = self._average(
            [avg_initial_gcg_judge, avg_final_gcg_judge]
        )

        summary_payload = {
            "total_results": len(results),
            "averages": {
                "initial": {
                    "judge_llm": avg_initial_judge_llm,
                    "pair_gpt_judge": avg_initial_pair_gpt_judge,
                    "gcg_judge": avg_initial_gcg_judge,
                },
                "final": {
                    "judge_llm": avg_final_judge_llm,
                    "pair_gpt_judge": avg_final_pair_gpt_judge,
                    "gcg_judge": avg_final_gcg_judge,
                    "strong_reject": {
                        "sr1_avg": avg_sr1,
                        "sr2_avg": avg_sr2,
                        "sr3_avg": avg_sr3,
                    },
                },
                "overall_mean_of_initial_and_final": {
                    "judge_llm": avg_judge_llm_overall,
                    "pair_gpt_judge": avg_pair_gpt_judge_overall,
                    "gcg_judge": avg_gcg_judge_overall,
                },
            },
        }

        out_dir = self._build_summary_dir(algo_name)

        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())

        file_path = out_dir / f"{summary_name}_{now}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=4, ensure_ascii=False)

        return file_path