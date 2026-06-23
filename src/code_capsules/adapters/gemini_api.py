"""
GeminiAPIClient: drives a Gemini model as a coding agent via google-genai.

Uses google.genai (google-genai package, ≥0.4) with function calling.
Conversation history is built from genai.types.Content alternating
user/model/function/function_response parts.

Requires GOOGLE_API_KEY or GEMINI_API_KEY in the environment. Auto-registered
only if google.genai is importable.

Default model: gemini-2.5-flash. Override per-task via VariantConfig.model.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from code_capsules.api import InvocationResult
from code_capsules.adapters._api_agent import (
    APIBasedAgent, TOOL_DEFS, _ProviderResponse,
)


class GeminiAPIClient(APIBasedAgent):
    name = "gemini_api"
    default_model = "gemini-2.5-flash"

    def __init__(self, default_model: Optional[str] = None):
        super().__init__()
        if default_model:
            self.default_model = default_model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from google import genai
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            self._client = genai.Client(api_key=key)
        return self._client

    # ── Message-shape adapters ─────────────────────────────────────────────
    # Gemini's "messages" are list[Content] where each Content has role +
    # parts. We use the dict form for simplicity.

    def _initial_messages(self, prompt: str, system: str) -> list[dict]:
        # System instruction goes in the config (handled in _call_api),
        # not in the contents list. We just seed the user prompt.
        return [{"role": "user", "parts": [{"text": prompt}]}]

    def _append_user_message(self, messages: list[dict], prompt: str) -> list[dict]:
        return messages + [{"role": "user", "parts": [{"text": prompt}]}]

    def _append_assistant_message(self, messages: list[dict], text: str,
                                   tool_calls: list[dict],
                                   raw: Any = None) -> list[dict]:
        # Prefer the raw Content object returned by the API: it natively carries
        # per-part `thought_signature` values, which Gemini 3.x REQUIRES to be
        # echoed back on functionCall parts (otherwise the next turn 400s with
        # "Function call is missing a thought_signature"). Reconstructing parts
        # from dicts drops the signature, so append the raw turn verbatim when
        # available. Falls back to dict reconstruction for older paths / 2.x.
        if raw is not None:
            return messages + [raw]
        parts: list[dict] = []
        if text:
            parts.append({"text": text})
        for tc in tool_calls:
            parts.append({
                "function_call": {
                    "name": tc["name"],
                    "args": tc.get("args", {}),
                }
            })
        return messages + [{"role": "model", "parts": parts}]

    def _append_tool_results(self, messages: list[dict],
                              results: list) -> list[dict]:
        # One user-role message containing all the function_response parts
        parts = []
        for tc_raw, output_text, _ in results:
            parts.append({
                "function_response": {
                    "name": tc_raw["name"],
                    "response": {"output": output_text},
                }
            })
        return messages + [{"role": "user", "parts": parts}]

    # ── API call ───────────────────────────────────────────────────────────

    def _call_api(self, model: str, messages: list[dict], tools: list[dict],
                  max_tokens: int) -> _ProviderResponse:
        client = self._ensure_client()
        from google.genai import types as gtypes

        gemini_tools = [gtypes.Tool(function_declarations=[
            gtypes.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters_json_schema=t["parameters"],
            )
            for t in tools
        ])]

        cfg = gtypes.GenerateContentConfig(
            system_instruction=self._system_prompt,
            tools=gemini_tools,
            max_output_tokens=max_tokens,
        )

        # Retry transient capacity/rate errors with exponential backoff. Preview
        # models (e.g. gemini-3.x-preview) 503 under load; a per-call agent step
        # should not die on a transient spike.
        import time as _time
        last_exc = None
        for attempt in range(5):
            try:
                resp = client.models.generate_content(
                    model=model, contents=messages, config=cfg,
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                s = str(exc)
                transient = any(t in s for t in
                                ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL"))
                if transient and attempt < 4:
                    _time.sleep(2 ** attempt * 3)  # 3,6,12,24s
                    continue
                return _ProviderResponse(
                    text="", tool_calls=[], input_tokens=0, output_tokens=0,
                    stop_reason="error", error=s,
                )
        if last_exc is not None:
            return _ProviderResponse(
                text="", tool_calls=[], input_tokens=0, output_tokens=0,
                stop_reason="error", error=str(last_exc),
            )

        text = ""
        tool_calls: list[dict] = []
        candidate = (resp.candidates or [None])[0]
        if candidate is not None and candidate.content is not None:
            for i, part in enumerate(candidate.content.parts or []):
                if getattr(part, "text", None):
                    text += part.text
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    tool_calls.append({
                        "name": fc.name,
                        "args": dict(fc.args or {}),
                        # Gemini doesn't return call_ids; synthesize for messaging
                        "call_id": f"{fc.name}_{i}",
                    })

        usage = getattr(resp, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0 if usage else 0
        output_tokens = (
            getattr(usage, "candidates_token_count", 0) or 0
        ) if usage else 0
        # Gemini context cache hits (when active). prompt_token_count is the
        # TOTAL input; cached_content_token_count is the subset that hit the
        # explicitly-configured cache. Implicit caching on Gemini 2.5 is also
        # surfaced here when the API enables it.
        cached_input_tokens = (
            getattr(usage, "cached_content_token_count", 0) or 0
        ) if usage else 0

        # Gemini stop reasons: STOP / MAX_TOKENS / SAFETY / etc.
        finish_reason = ""
        if candidate is not None:
            fr = getattr(candidate, "finish_reason", None)
            finish_reason = str(fr).rsplit(".", 1)[-1].lower() if fr else ""
        stop_reason = "tool_use" if tool_calls else (finish_reason or "end_turn")

        return _ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            stop_reason=stop_reason,
            # Carry the raw model turn so the history preserves Gemini 3.x
            # thought_signatures on functionCall parts (see _append_assistant_message).
            raw_message=(candidate.content if candidate is not None else None),
        )


__all__ = ["GeminiAPIClient"]
