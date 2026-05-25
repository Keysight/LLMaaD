import time
from prompt_reshaping.artifacts import utils as ut
from prompt_reshaping.algos.algo1 import Algo1Reshaper
from prompt_reshaping.artifacts.traces import PromptInput, PromptArtifacts, PromptRunResult, JudgeScores
from prompt_reshaping.llm_gen.clients import ChatRequest
from prompt_reshaping.artifacts.logging import log_action
from dataclasses import replace

r'''
# -----------------------------------------------------
Algorithm 1q :
algo1 -> insert a follow up detailed questions at the end -> attacker Judge
# -----------------------------------------------------
'''

class Algo1qReshaper(Algo1Reshaper):
    name = "algo1q"

    def __init__(self, expansion_words: int = 400, compression_words: int = 200):
        self.followup_system_prompt = ut.followup_system_prompt()
        super().__init__(expansion_words=expansion_words, compression_words=compression_words, algo1q_flag=True)


    def run(self, inp:PromptInput, judges: JudgeScores):
        t0 = time.time()
        promptart = PromptArtifacts(initial_prompt=inp.input_prompt)
        reshaping_selector_kwargs = inp.reshaping_model.as_selector_kwargs()
        promptart, initial_scores, self.judge = super().run(inp, judges)

        # 1 -> 6 steps same as algo1

        # 7) Follow Up Response
        with log_action("followup generation"):
            promptart.followup_resp = self.model.model_selector(ChatRequest(user_prompt=promptart.harmful_injected_prompt,
                                                                       system_prompt=self.followup_system_prompt),
                                                                       **reshaping_selector_kwargs)[0]
    
        # 8) final scoring
        final_scores = replace(initial_scores, response=(promptart.harmful_injected_prompt + " " + promptart.followup_resp))
        if judges.has_judges():
            final_scores = self.judge.scoring(final_scores, **reshaping_selector_kwargs)

        t1 = time.time()
        return PromptRunResult(
            artifacts=promptart,
            initial_scores=initial_scores,
            final_scores=final_scores,
            timings={"total_sec": t1 - t0},
            metadata={"algo": self.name, "reshaping_model": inp.reshaping_model.to_metadata()},
        )