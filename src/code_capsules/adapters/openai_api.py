"""
OpenAIAPIClient: drives a GPT model as a coding agent via Chat Completions.

Uses the openai Python SDK (>=1.0). Tool use through function calling.
Conversation state lives in the messages list; sessions are kept in
_api_agent._SESSIONS for resume support.

Requires OPENAI_API_KEY in the environment. Auto-registered only if the
SDK is importable.

Default model: gpt-4o. Override per-task via VariantConfig.model.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from code_capsules.api import InvocationResult
from code_capsules.adapters._api_agent import (
    APIBasedAgent, TOOL_DEFS, _ProviderResponse,
)


class OpenAIAPIClient(APIBasedAgent):
    name = "openai_api"
    default_model = "gpt-4o"

    def __init__(self, default_model: Optional[str] = None):
        super().__init__()
        if default_model:
            self.default_model = default_model
        # Lazy SDK init; defer the client until first use so import-time
        # doesn't fail when the package is missing.
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client

    # ── Message-shape adapters ─────────────────────────────────────────────

    def _initial_messages(self, prompt: str, system: str) -> list[dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]

    def _append_user_message(self, messages: list[dict], prompt: str) -> list[dict]:
        return messages + [{"role": "user", "content": prompt}]

    def _append_assistant_message(self, messages: list[dict], text: str,
                                   tool_calls: list[dict],
                                   raw: Any = None) -> list[dict]:
        msg: dict = {"role": "assistant", "content": text or None}
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["call_id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc.get("args", {})),
                    },
                }
                for tc in tool_calls
            ]
        return messages + [msg]

    def _append_tool_results(self, messages: list[dict],
                              results: list) -> list[dict]:
        out = list(messages)
        for tc_raw, output_text, _output_size in results:
            out.append({
                "role": "tool",
                "tool_call_id": tc_raw["call_id"],
                "content": output_text,
            })
        return out

    # ── API call ───────────────────────────────────────────────────────────

    def _call_api(self, model: str, messages: list[dict], tools: list[dict],
                  max_tokens: int) -> _ProviderResponse:
        client = self._ensure_client()
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=openai_tools,
                max_completion_tokens=max_tokens,
            )
        except Exception as exc:
            return _ProviderResponse(
                text="", tool_calls=[], input_tokens=0, output_tokens=0,
                stop_reason="error", error=str(exc),
            )

        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""
        raw_tcs = msg.tool_calls or []
        tool_calls: list[dict] = []
        for tc in raw_tcs:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({
                "name": tc.function.name,
                "args": args,
                "call_id": tc.id,
            })

        usage = resp.usage
        # Chat Completions exposes cache hits under prompt_tokens_details.
        # prompt_tokens is the TOTAL (cached + fresh).
        cached_input_tokens = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached_input_tokens = getattr(details, "cached_tokens", 0) or 0
        return _ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cached_input_tokens=cached_input_tokens,
            stop_reason=choice.finish_reason or "end_turn",
        )


__all__ = ["OpenAIAPIClient"]
