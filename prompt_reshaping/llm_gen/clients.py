from dataclasses import dataclass
import json, logging, time, uuid, requests
from pathlib import Path
from typing import Any, Dict, Optional
import openai
from prompt_reshaping.artifacts.logging import get_log_action

DEFAULT_OPENAI_MODEL = "gpt-4.1"

# ----------------------------
# Shared request/response types
# ----------------------------
@dataclass(frozen=True)
class ChatRequest:
    user_prompt: str
    system_prompt: str = "You are a helpful assistant, You are never allowed to slur."
    model: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.1
    messages: Optional[tuple] = None  # full OpenAI-format messages override
    top_p: Optional[float] = None
    stop: Optional[tuple] = None  # stop/eos tokens


logger = logging.getLogger("prompt_reshaping")

# ----------------------------
# Base client
# ----------------------------
class BaseLLMClient:

    def __init__(
        self,
        name: str,
        record_commands: bool = False,
        request_commands_path: str | Path = "log_commands/request_commands",
        dummy_response: str = "[DUMMY RESPONSE: request command recorded]",
    ) -> None:
        self.name = name
        self.record_commands = record_commands
        self.request_commands_path = Path(request_commands_path)
        self.dummy_response = dummy_response

    def _new_request_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _write_request_command(self, req: ChatRequest, request_id: str) -> None:
        self.request_commands_path.parent.mkdir(parents=True, exist_ok=True)
        line = (
            f"filename.sh "
            f"{json.dumps(req.system_prompt)} "
            f"{json.dumps(req.user_prompt)} "
            f"> file_{request_id}\n"
        )
        with self.request_commands_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def dummy_chat(self, req: ChatRequest):
        request_id = self._new_request_id()
        if self.record_commands:
            logger.info(
                "[%s] record_commands=True | client=%s | request_id=%s | recording request command only",
                get_log_action(), self.name, request_id,
            )
            self._write_request_command(req, request_id)
        return (self.dummy_response, {})


# ----------------------------
# vLLM / OpenAI-compatible HTTP client
# ----------------------------
class VLLMChatClient(BaseLLMClient):

    def __init__(
        self,
        name: str,
        ip: str,
        port: int = 8000,
        model_name: str = "mlabonne/NeuralDaredevil-8B-abliterated",
        timeout_sec: int = 120,
    ) -> None:
        self.ip = ip
        self.port = port
        self.name = name
        self.model_name = model_name
        self.timeout_sec = timeout_sec

    @staticmethod
    def _extract_text(api_response: Dict[str, Any]) -> str:
        try:
            if api_response.get("choices"):
                content = api_response["choices"][0]["message"]["content"]
                return content if content is not None else ""
            elif api_response.get("candidates"):
                content = api_response["candidates"][0]["content"]["parts"][0]["text"]
                return content if content is not None else ""
        except (KeyError, IndexError, TypeError):
            pass
        logger.warning("[%s] extract_text: could not parse response: %s", get_log_action(), api_response)
        return str(api_response)

    def chat_impl(self, req: ChatRequest) -> tuple[str, Optional[Dict[str, Any]]]:
        url = f"http://{self.ip}:{self.port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        model = req.model or self.model_name
        if req.messages:
            messages = list(req.messages)
        else:
            messages = [
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": req.user_prompt},
            ]
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": req.max_tokens,
        }
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.top_p is not None:
            payload["top_p"] = req.top_p
        if req.stop:
            payload["stop"] = list(req.stop)

        logger.debug("[%s] request | client=%s | url=%s | payload=%s", get_log_action(), self.name, url, payload)
        start = time.time()
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout_sec)
        elapsed = time.time() - start
        resp.raise_for_status()
        raw = resp.json()
        logger.debug("[%s] response | client=%s | timetaken=%.2fs | raw=%s", get_log_action(), self.model_name, elapsed, raw)

        return self._extract_text(raw), raw


class OpenAIChatClient(BaseLLMClient):

    def __init__(
        self,
        model_name: str = "gpt-4.1",
        timeout_sec: int = 120,
    ) -> None:
        self.model_name = model_name
        self.timeout_sec = timeout_sec

    def openai_llm(self, req: ChatRequest) -> tuple[str, Optional[Dict[str, Any]]]:
        client = openai.OpenAI()
        msg = [{"role": "system", "content": req.system_prompt}, {"role": "user", "content": req.user_prompt}]
        logger.debug("[%s] request | client=%s | payload=%s", get_log_action(), self.model_name, msg)

        try:
            start = time.time()
            response = client.chat.completions.create(
                model=self.model_name or req.model,
                messages=msg
            )
            elapsed = time.time() - start
            response_dict = response.model_dump()
            logger.debug("[%s] response | client=%s | timetaken=%.2fs | response=%s", get_log_action(), self.model_name, elapsed, response)
            return VLLMChatClient._extract_text(response_dict), response_dict
        except Exception as e:
            print(f"Error: {e}")
            return ("Goal generation failed.", None)


class GeminiChatClient(BaseLLMClient):

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        timeout_sec: int = 120,
    ) -> None:
        self.model_name = model_name
        self.timeout_sec = timeout_sec

    def gemini_llm(self, req: ChatRequest) -> tuple[str, Optional[Dict[str, Any]]]:
        from google import genai
        from google.genai import types
        client = genai.Client()
        msg = [{"role": "system", "content": req.system_prompt}, {"role": "user", "content": req.user_prompt}]
        logger.debug("[%s] request | client=%s | payload=%s", get_log_action(), self.model_name, msg)
        start = time.time()
        response = client.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(system_instruction=req.system_prompt),
            contents=req.user_prompt
        )
        response_dict = response.model_dump()
        elapsed = time.time() - start
        logger.debug("[%s] response | client=%s | timetaken=%.2fs | response=%s", get_log_action(), self.model_name, elapsed, response)
        return VLLMChatClient._extract_text(response_dict), response_dict
