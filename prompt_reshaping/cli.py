import argparse
from typing import List, Tuple
from prompt_reshaping.artifacts.traces import PromptInput, JudgeScores, ReshapingModelConfig
from prompt_reshaping.multi.runner import MultiTryRunner
from prompt_reshaping.datasets.loader import load_dataset
from prompt_reshaping.sentence_gen.generator import harmful_sentence_gen
from prompt_reshaping.llm_gen.clients import DEFAULT_OPENAI_MODEL
from prompt_reshaping.llm_gen.model_pick import LOCAL_VLLM_MODEL_PROFILES

_vllm = ReshapingModelConfig()  # single source of truth for vLLM defaults

SUPPORTED_API_MODELS = [
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]
LEGACY_VLLM_ALIASES = {
    "abliterated": _vllm.model_name,
}


def apply_judge_defaults(args) -> None:
    if not args.select_judge:
        return

    if args.all_judges:
        args.gcg = True
        args.pair_gpt = args.pair_gpt or DEFAULT_OPENAI_MODEL
        args.strong_reject = args.strong_reject or LOCAL_VLLM_MODEL_PROFILES["gpt-oss"]["model_name"]
        args.nlp = True
        args.gptfuzz = True


# ------------------------------------------------------------------
# Input builders
# ------------------------------------------------------------------
def build_reshaping_model_config(args) -> ReshapingModelConfig:
    if not args.model:
        raise ValueError("--model is required")

    if args.port <= 0:
        raise ValueError("--port must be a positive integer")

    if args.reshaping_backend == "api":
        if args.model not in SUPPORTED_API_MODELS:
            supported = ", ".join(SUPPORTED_API_MODELS)
            raise ValueError(f"Unsupported API model '{args.model}'. Choose one of: {supported}")
        return ReshapingModelConfig(
            backend="api",
            model_name=args.model,
            ip=None,
            port=None,
        )

    model_name = LEGACY_VLLM_ALIASES.get(args.model, args.model)
    return ReshapingModelConfig(
        backend="vllm",
        model_name=model_name,
        ip=args.ip,
        port=args.port,
    )


def build_judge_scores(args) -> JudgeScores:
    return JudgeScores(
        gcg_judge=args.gcg,
        pair_gpt_judge=args.pair_gpt,
        strongreject_judge=args.strong_reject,
        nlp_classifiers=args.nlp,
        custom_judge=args.custom,
        gptfuzz_judge=args.gptfuzz,
    )


def build_single_cli_input(args) -> Tuple[PromptInput, JudgeScores]:
    return (
        PromptInput(
            input_prompt=args.prompt,
            harmful_sentences=args.sentence if args.sentence else harmful_sentence_gen(args.prompt, args.reshaping_model.model_name, args.reshaping_model.ip),
            reshaping_model=args.reshaping_model,
            dataset_name=args.dataset,
            run_dir=args.run_dir,
            normal_model=args.model,
        ),
        build_judge_scores(args),
    )


def build_multi_range_inputs(args) -> List[Tuple[PromptInput, JudgeScores]]:
    prompts, sentences = load_dataset(args.dataset)
    inputs: List[Tuple[PromptInput, JudgeScores]] = []

    if len(prompts) != len(sentences):
        raise ValueError(
            f"Dataset mismatch: prompts={len(prompts)} sentences={len(sentences)}"
        )

    if args.range:
        start, end = args.range

        if start < 0 or end < 0:
            raise ValueError("Range values must be non-negative")

        if start >= len(prompts):
            raise IndexError(
                f"Range start {start} out of bounds for dataset size {len(prompts)}"
            )

        if end > len(prompts):
            raise IndexError(
                f"Range end {end} out of bounds for dataset size {len(prompts)}"
            )

        if start >= end:
            raise ValueError("Range must satisfy: start < end")

        prompt_slice = prompts[start:end]
        sentence_slice = sentences[start:end]
    else:
        prompt_slice = prompts
        sentence_slice = sentences

    for prompt, sentence in zip(prompt_slice, sentence_slice):
        prompt_input = PromptInput(
            input_prompt=prompt,
            harmful_sentences=sentence if sentence else harmful_sentence_gen(prompt),
            reshaping_model=args.reshaping_model,
            dataset_name=args.dataset,
            run_dir=args.run_dir,
            normal_model=args.model,
        )
        inputs.append((prompt_input, build_judge_scores(args)))

    return inputs


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI for prompt reshaping experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --------------------------------------------------
    # Mode selection
    # --------------------------------------------------
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--single",
        action="store_true",
        help="Run a single prompt/sentence directly from CLI",
    )
    mode_group.add_argument(
        "--multi",
        action="store_true",
        help="Run multiple prompt/sentence pairs from dataset",
    )

    # --------------------------------------------------
    # Common options
    # --------------------------------------------------
    parser.add_argument(
        "--algo",
        choices=["algo1", "algo1q", "algo2"],
        required=True,
        help="Reshaping algorithm to run",
    )

    reshaping_backend = parser.add_mutually_exclusive_group()
    reshaping_backend.add_argument(
        "--vllm",
        dest="reshaping_backend",
        action="store_const",
        const="vllm",
        help="Use a locally hosted OpenAI-compatible vLLM endpoint for reshaping",
    )
    reshaping_backend.add_argument(
        "--api",
        dest="reshaping_backend",
        action="store_const",
        const="api",
        help="Use a cloud API model for reshaping",
    )
    parser.set_defaults(reshaping_backend="vllm")

    parser.add_argument(
        "--model",
        required=True,
        help=(
            f"Model to use for reshaping. For --vllm: a HuggingFace model id or alias (e.g. abliterated). "
            f"For --api: one of {', '.join(SUPPORTED_API_MODELS)}"
        ),
    )
    parser.add_argument(
        "--reshaping_model",
        dest="model",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ip",
        default=_vllm.ip,
        help="Host IP for local vLLM reshaping model",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_vllm.port,
        help="Host port for local vLLM reshaping model",
    )
    parser.add_argument(
        "--run_dir",
        default="v5",
        help="Run/output directory tag",
    )
    parser.add_argument(
        "--expansion_words",
        type=int,
        default=400,
        help="Expansion word count",
    )
    parser.add_argument(
        "--compression_words",
        type=int,
        default=200,
        help="Compression word count",
    )
    parser.add_argument(
        "--print_test_commands",
        action="store_true",
        help="Record request commands and return dummy responses",
    )

    # --------------------------------------------------
    # Judge options
    # --------------------------------------------------
    judge_group = parser.add_argument_group(
        "judge options",
        "Use --select-judge to enable. Example: --select-judge --gcg --pair-gpt gpt-4.1",
    )
    judge_group.add_argument(
        "--select-judge",
        action="store_true",
        help="Enable explicit judge selection/configuration",
    )
    judge_group.add_argument(
        "--all-judges",
        action="store_true",
        help="Select all available judges (use with --select-judge)",
    )
    judge_group.add_argument(
        "--gcg",
        action="store_true",
        help="Enable GCG judge",
    )
    judge_group.add_argument(
        "--pair-gpt",
        dest="pair_gpt",
        type=str,
        default=None,
        help="Enable PAIR-GPT judge and specify model, e.g. --pair-gpt gpt-4.1",
    )
    judge_group.add_argument(
        "--strong-reject",
        dest="strong_reject",
        type=str,
        default=None,
        help=f"Enable StrongReject judge and specify model (default when --all-judges: {LOCAL_VLLM_MODEL_PROFILES['gpt-oss']['model_name']})",
    )
    judge_group.add_argument(
        "--nlp",
        action="store_true",
        default=None,
        help="Enable NLP toxicity classifier judges (s-nlp/roberta + martin-ha)",
    )
    judge_group.add_argument(
        "--custom",
        type=str,
        default=None,
        help="Enable custom LLM judge and specify model identifier",
    )
    judge_group.add_argument(
        "--gptfuzz",
        action="store_true",
        help="Enable the GPTFuzz local classifier judge",
    )

    # --------------------------------------------------
    # --single options
    # --------------------------------------------------
    single_group = parser.add_argument_group(
        "--single mode options",
        "Arguments used when running with --single",
    )
    single_group.add_argument(
        "--prompt",
        help="Input prompt",
    )
    single_group.add_argument(
        "--sentence",
        help="Harmful sentence corresponding to the prompt",
    )

    # --------------------------------------------------
    # --multi options
    # --------------------------------------------------
    multi_group = parser.add_argument_group(
        "--multi mode options",
        "Arguments used when running with --multi",
    )
    multi_group.add_argument(
        "--dataset",
        default="dataset2",
        help="Dataset name to load prompts and sentences from",
    )
    multi_group.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Dataset slice [START:END], e.g. --range 5 10",
    )
    multi_group.add_argument(
        "--all",
        action="store_true",
        help="Run all prompt/sentence pairs from dataset",
    )
    multi_group.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker threads for multi mode",
    )

    return parser


BANNER = r"""
██╗     ██╗     ███╗   ███╗ █████╗  █████╗ ██████╗
██║     ██║     ████╗ ████║██╔══██╗██╔══██╗██╔══██╗
██║     ██║     ██╔████╔██║███████║███████║██║  ██║
██║     ██║     ██║╚██╔╝██║██╔══██║██╔══██║██║  ██║
███████╗███████╗██║ ╚═╝ ██║██║  ██║██║  ██║██████╔╝
╚══════╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝
       Misdirection as a Defense Against Model-Guided Automated LLM Attacks (LLMaaD)
"""


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print(BANNER)
    parser = build_parser()
    args = parser.parse_args()
    apply_judge_defaults(args)

    if args.single and not args.prompt:
        parser.error("--single requires --prompt")

    if args.multi and not args.range and not args.all:
        parser.error("--multi requires either --range START END or --all")

    try:
        args.reshaping_model = build_reshaping_model_config(args)
    except ValueError as exc:
        parser.error(str(exc))

    runner = MultiTryRunner(
        algo_name=args.algo,
        expansion_words=args.expansion_words,
        compression_words=args.compression_words,
        print_test_commands=args.print_test_commands,
    )

    if args.single:
        prompt_input, judge_scores = build_single_cli_input(args)
        result = runner.run_single(prompt_input, judge_scores)
        print(result)
        return

    if args.multi:
        inputs = build_multi_range_inputs(args)
        runner.run_multiple(inputs, workers=args.workers)
        return


if __name__ == "__main__":
    main()
