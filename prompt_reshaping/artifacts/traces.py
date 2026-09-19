import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ReshapingModelConfig:
    backend: str = "vllm"
    model_name: str = "mlabonne/NeuralDaredevil-8B-abliterated"
    ip: Optional[str] = os.getenv("ABLITERATED_IP", "localhost")
    port: Optional[int] = int(os.getenv("ABLITERATED_PORT", 8000))

    def as_selector_kwargs(self) -> Dict[str, Any]:
        if self.backend == "vllm":
            return {
                "model": self.model_name,
                "ip": self.ip,
                "port": self.port,
            }
        return {"model": self.model_name}

    def to_metadata(self) -> Dict[str, Any]:
        payload = {
            "backend": self.backend,
            "model": self.model_name,
        }
        if self.backend == "vllm":
            payload["ip"] = self.ip
            payload["port"] = self.port
        return payload

@dataclass
class PromptInput:
    input_prompt: str
    harmful_sentences: str | list[str]
    dataset_name: str = "dataset2"
    run_dir: str = "v5"
    normal_model: str = "gpt-oss"
    reshaping_model: ReshapingModelConfig = field(default_factory=ReshapingModelConfig)

@dataclass
class PromptArtifacts:
    initial_prompt: str
    initial_response: str = ""
    jumbled_prompt: str = ""
    expansion_prompt: str = ""
    expanded_prompt: str = ""
    inserted_context_prompt: str = ""
    jumbled_inserted_context_prompt: str = ""
    compress_input: str = ""
    summarized_prompt: str = ""
    harmful_injected_prompt: str = ""
    followup_resp: str = ""
    algo2_resp: str = ""

@dataclass
class JudgeScores:
    prompt : str = ""
    response: str = ""

    gcg_judge: bool = False
    pair_gpt_judge: str = ""
    strongreject_judge: str = ""
    nlp_classifiers: bool = False
    custom_judge: str = ""
    gptfuzz_judge: bool = False

    judge_llm_score: Optional[float] = None
    pair_gpt_judge_score: Optional[float] = None
    gcg_judge_score: Optional[float] = None
    strong_reject_score: Optional[Tuple[float, float, float]] = None
    classifiers_score: Optional[Tuple[float, float]] = None
    gptfuzz_judge_score: Optional[Dict[str, Any]] = None

    def has_judges(self) -> bool:
        return bool(self.gcg_judge or self.pair_gpt_judge or self.strongreject_judge
                    or self.nlp_classifiers or self.custom_judge or self.gptfuzz_judge)

    def __str__(self) -> str:
        lines = ["── Results ───────────────────────────────"]
        if self.prompt:
            lines.append(f"  prompt   : {self.prompt}")
        if self.response:
            lines.append(f"  response : {self.response}")

        score_fields = {
            "gcg_judge_score": self.gcg_judge_score,
            "pair_gpt_judge_score": self.pair_gpt_judge_score,
            "judge_llm_score": self.judge_llm_score,
            "strong_reject_score": self.strong_reject_score,
            "classifiers_score": self.classifiers_score,
            "gptfuzz_judge_score": self.gptfuzz_judge_score,
        }
        active = {k: v for k, v in score_fields.items() if v is not None}
        if active:
            lines.append("  scores   :")
            for name, value in active.items():
                lines.append(f"    {name:<25} {value}")
        else:
            lines.append("  scores   : (none)")
        lines.append("──────────────────────────────────────────")
        return "\n".join(lines)
    

@dataclass
class PromptRunResult:
    artifacts: PromptArtifacts
    initial_scores: JudgeScores = field(default_factory=JudgeScores)
    final_scores: JudgeScores = field(default_factory=JudgeScores)
    timings: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class VLLMProfile:
    ip: str
    default_model: str
    port: int = 8000
