from typing import Dict, Optional
from prompt_reshaping.llm_gen.clients import BaseLLMClient, VLLMChatClient, ChatRequest, OpenAIChatClient, GeminiChatClient


DEFAULT_VLLM_PORT = 8000
LOCAL_VLLM_MODEL_PROFILES = {
    "abliterated": {
        "ip": "10.36.129.1",
        "model_name": "mlabonne/NeuralDaredevil-8B-abliterated",
        "port": DEFAULT_VLLM_PORT,
    },
    "gpt-oss": {
        "ip": "10.36.129.2",
        "model_name": "openai/gpt-oss-120b",
        "port": DEFAULT_VLLM_PORT,
    },
}

class ModelSelect:
    def __init__(self):
        self._cache: Dict[str, VLLMChatClient] = {}

    def _get_vllm(self, key: str, ip: str, model_name: str, port: int) -> VLLMChatClient:
        # simple cache so you don't recreate every call
        if key not in self._cache:
            self._cache[key] = VLLMChatClient(
                name=f"vllm:{key}",
                ip=ip,
                port=port,
                model_name=model_name,   # (in my earlier class it's default_model)
            )
        return self._cache[key]

    def _resolve_vllm_target(
            self,
            model: str,
            ip: Optional[str],
            port: Optional[int],
        ) -> tuple[str, str, int]:
        profile = LOCAL_VLLM_MODEL_PROFILES.get(model, {})
        resolved_model = profile.get("model_name", model)
        resolved_ip = ip or profile.get("ip")
        resolved_port = port or profile.get("port", DEFAULT_VLLM_PORT)

        if resolved_ip is None:
            raise ValueError("Local vLLM models require --ip or a known local model alias.")

        return resolved_model, resolved_ip, resolved_port

    def model_selector(
            self, 
            req: ChatRequest, 
            model: Optional[str] = None, 
            ip: Optional[str] = None,
            port: Optional[int] = None,
        ):
        if model is None:
            client = BaseLLMClient(name="Empty",record_commands=True)
            return client.dummy_chat(req)

        if model and (ip or model in LOCAL_VLLM_MODEL_PROFILES):
            resolved_model, resolved_ip, resolved_port = self._resolve_vllm_target(
                model=model,
                ip=ip,
                port=port,
            )
            client = self._get_vllm(
                key=f"{resolved_model}@{resolved_ip}:{resolved_port}",
                ip=resolved_ip,
                port=resolved_port,
                model_name=resolved_model,
            )

        elif model.startswith("gpt-"):
            client = OpenAIChatClient(model_name=model)
            return client.openai_llm(req)

        elif model.startswith("gemini-"):
            client = GeminiChatClient(model_name=model)
            return client.gemini_llm(req)

        else:
           raise ValueError ("Model not Specified")

        return client.chat_impl(req)
