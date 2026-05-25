import os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompt_reshaping.artifacts.logging import configure_logger
from prompt_reshaping.artifacts.writer import ResultWriter
from prompt_reshaping.algos.algo1 import Algo1Reshaper
from prompt_reshaping.algos.algo1q import Algo1qReshaper
from prompt_reshaping.algos.algo2 import Algo2Reshaper


def _make_stem() -> str:
    return f"final_{os.getpid()}_{time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())}"


class MultiTryRunner:
    def __init__(
        self,
        algo_name: str,
        expansion_words: int,
        compression_words: int,
        print_test_commands: bool = False,
    ):
        self.algo = self.get_algo(
            algo_name=algo_name,
            expansion_words=expansion_words,
            compression_words=compression_words,
        )
        self.print_test_commands = print_test_commands
        self.writer = ResultWriter()

    # ------------------------------------------------------------------
    # Algo factory
    # ------------------------------------------------------------------
    @staticmethod
    def get_algo(algo_name: str, expansion_words: int, compression_words: int):
        if algo_name == "algo1":
            return Algo1Reshaper(
                expansion_words=expansion_words,
                compression_words=compression_words,
            )
        elif algo_name == "algo1q":
            return Algo1qReshaper(
                expansion_words=expansion_words,
                compression_words=compression_words,
            )
        elif algo_name == "algo2":
            return Algo2Reshaper(
                expansion_words=expansion_words,
                compression_words=compression_words,
            )
        else:
            raise ValueError(f"Unknown algo: {algo_name}")

    def run_single(self, inp, judges):
        stem = _make_stem()
        configure_logger(Path("logs") / f"{stem}.log")

        result = self.algo.run(inp, judges)
        print(f"Running for prompt ==> {inp.input_prompt}")
        path = self.writer.write_prompt_result(
            inp=inp,
            result_obj=result,
            algo_name=self.algo.name,
            file_stem=stem,
        )
        print(f"Saved to: {path}")
        return result.final_scores

    def run_multiple(self, inputs, workers: int = 1):
        """
        inputs: List[Tuple[PromptInput, JudgeScores]]
        workers: number of parallel threads (>1 enables concurrent execution)
        """
        def _run_one(item):
            inp, judges = item
            stem = _make_stem()
            configure_logger(Path("logs") / f"{stem}.log")

            print(f"Running for {inp.input_prompt}")
            result = self.algo.run(inp, judges)
            path = self.writer.write_prompt_result(
                inp=inp,
                result_obj=result,
                algo_name=self.algo.name,
                file_stem=stem,
            )
            print(f"Saved to: {path}")
            return result

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_run_one, item) for item in inputs]
                all_results = [f.result() for f in as_completed(futures)]
        else:
            all_results = [_run_one(item) for item in inputs]

        return [r.final_scores for r in all_results]
