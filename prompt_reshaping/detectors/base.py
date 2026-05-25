from prompt_reshaping.detectors.detector_gen import GCG_JUDGE, PAIR_GPT_JUDGE, StrongReject_JUDGE, NLPClassifiers, CustomAttacker_JUDGE, GPTFuzz_JUDGE
from prompt_reshaping.artifacts.traces import JudgeScores

class JudgeService:
    def __init__(self):
        self.gcg = None
        self.pair_gpt = None
        self.strongreject = None
        self.nlp_classifiers = None
        self.custom = None
        self.gptfuzz = None

    def scoring(self, judge_details: JudgeScores, **kwargs) -> JudgeScores:

        if judge_details.gcg_judge:
            if self.gcg is None:
                self.gcg = GCG_JUDGE()
            judge_details.gcg_judge_score = self.gcg.gcg_scoring(inp=judge_details, **kwargs)

        if judge_details.pair_gpt_judge:
            if self.pair_gpt is None:
                self.pair_gpt = PAIR_GPT_JUDGE()
            judge_details.pair_gpt_judge_score = self.pair_gpt.judge_scoring(inp=judge_details, **kwargs)

        if judge_details.strongreject_judge:
            if self.strongreject is None:
                self.strongreject = StrongReject_JUDGE()
            judge_details.strong_reject_score = self.strongreject.strongreject_scoring(inp=judge_details, **kwargs)

        if judge_details.nlp_classifiers:
            if self.nlp_classifiers is None:
                self.nlp_classifiers = NLPClassifiers()
            judge_details.classifiers_score = [
                self.nlp_classifiers.s_nlp_scoring(inp=judge_details, **kwargs),
                self.nlp_classifiers.martin_ha_scoring(inp=judge_details, **kwargs),
            ]

        if judge_details.custom_judge:
            if self.custom is None:
                self.custom = CustomAttacker_JUDGE()
            judge_details.judge_llm_score = self.custom.custom_judge_scoring(inp=judge_details, **kwargs)

        if judge_details.gptfuzz_judge:
            if self.gptfuzz is None:
                self.gptfuzz = GPTFuzz_JUDGE()
            judge_details.gptfuzz_judge_score = self.gptfuzz.score_exec(inp=judge_details, **kwargs)

        return judge_details