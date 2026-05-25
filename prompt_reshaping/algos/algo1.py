import time, re
from prompt_reshaping.artifacts import utils as ut
from prompt_reshaping.artifacts.traces import PromptInput, PromptArtifacts, PromptRunResult, JudgeScores
from prompt_reshaping.llm_gen.model_pick import ModelSelect
from prompt_reshaping.llm_gen.clients import ChatRequest
from prompt_reshaping.detectors.base import JudgeService
from prompt_reshaping.artifacts.logging import log_action
from dataclasses import replace


r'''
# -----------------------------------------------------
Algorithm 1 :
prompt -> token jumble -> expand (400 words) -> insert context-preserving tokens -> token jumble -> remove punctuations "?!,.!" -> summerize (200 words) -> insert malicious-intent sentences -> attacker Judge
# -----------------------------------------------------
'''

class Algo1Reshaper():
    name = "algo1"

    def __init__(
        self,
        expansion_words: int = 400,
        compression_words: int = 200,
        algo1q_flag = False,
    ):
        self.expansion_words = expansion_words
        self.expansion_sys_prompt = ut.expansion_system_prompt(expansion_words)
        self.summarization_sys_prompt = ut.summarization_system_prompt() % compression_words
        self.algo1q_flag = algo1q_flag
        self.model = ModelSelect()
        self.judge = JudgeService()


    def run(self, inp: PromptInput, judges: JudgeScores):
        t0 = time.time()
        judges.prompt = inp.input_prompt
        promptart = PromptArtifacts(initial_prompt=inp.input_prompt)
        reshaping_selector_kwargs = inp.reshaping_model.as_selector_kwargs()

        # 1) Initial Response
        with log_action("initial response"):
            judges.response = self.model.model_selector(ChatRequest(user_prompt=promptart.initial_prompt),
                                                        model=inp.normal_model,
                                                        ip=inp.reshaping_model.ip,
                                                        port=inp.reshaping_model.port)[0]

        # 2) Initial scoring
        init_scores = self.judge.scoring(judges, **reshaping_selector_kwargs) if judges.has_judges() else judges

        # 3) expansion
        promptart.jumbled_prompt = ut.jumble_words(promptart.initial_prompt)
        promptart.expansion_prompt = promptart.jumbled_prompt
        with log_action("prompt expansion"):
            promptart.expanded_prompt = self.model.model_selector(ChatRequest(user_prompt=promptart.expansion_prompt,
                                                                         system_prompt=self.expansion_sys_prompt),
                                                                         **reshaping_selector_kwargs)[0]

        # 4) context tokens + jumble + strip punct
        insert_count = ut.count_sentences(promptart.expanded_prompt) * 2
        promptart.inserted_context_prompt = ut.insert_context_preserving_words(promptart.expanded_prompt, insert_count)
        promptart.jumbled_inserted_context_prompt = ut.jumble_words(promptart.inserted_context_prompt)
        promptart.compress_input = re.sub(r'[?!,.!]', '', promptart.jumbled_inserted_context_prompt)

        # 5) summarization
        with log_action("prompt summarization"):
            summarized_raw = self.model.model_selector(ChatRequest(user_prompt=promptart.compress_input,
                                                              system_prompt=self.summarization_sys_prompt),
                                                              **reshaping_selector_kwargs)[0]
        promptart.summarized_prompt = ut.extract_after_double_newline(summarized_raw)

        # 6) harmful injection
        final_judges = replace(judges)
        final_judges.response = promptart.harmful_injected_prompt = ut.malicious_themed_sentence_injection(
            promptart.summarized_prompt, inp.harmful_sentences
        )

        if self.algo1q_flag:
            return promptart, init_scores, self.judge

        # 7) final scoring
        final_scores = self.judge.scoring(final_judges, **reshaping_selector_kwargs) if judges.has_judges() else final_judges

        t1 = time.time()
        return PromptRunResult(
            artifacts=promptart,
            initial_scores=init_scores,
            final_scores=final_scores,
            timings={"total_sec": t1 - t0},
            metadata={"algo": self.name, "reshaping_model": inp.reshaping_model.to_metadata()},
        )
