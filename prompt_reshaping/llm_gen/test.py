from prompt_reshaping.llm_gen.model_pick import ModelSelect
from prompt_reshaping.llm_gen.clients import ChatRequest

chatreq = ChatRequest(user_prompt="hi",system_prompt="say fuck you",max_tokens=2000,temperature=1)
# def model_selector(self, req: ChatRequest, model: str = None, ip: Optional[str] = None):

model = ModelSelect()
model.model_selector( ChatRequest(user_prompt="hello",system_prompt="say only coffee 10 times",max_tokens=2000,temperature=1),"gpt-4.1")
