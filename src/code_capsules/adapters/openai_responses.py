"""
OpenAIResponsesClient: OpenAI Responses API driver for the GPT-5 codex family.

The Chat Completions endpoint refuses gpt-5-codex (and other reasoning /
codex variants) with 404; those models are only served via /v1/responses.
This client drives the Responses API with the same Read/Write/Edit/Bash
tool loop as OpenAIAPIClient.

Differences from the Chat Completions client:
  - tool format is flat ({type, name, description, parameters}) - no nested
    "function" key
  - input items can be messages OR function_call / function_call_output;
    each turn re-sends the full history (server-side state via
    `previous_response_id` is a future optimization)
  - response output is a list of typed items; function_call items carry
    name + arguments (JSON string) + call_id
  - reasoning models emit a separate `reasoning` item we capture but don't
    pass back (the model reconstructs reasoning internally on next call)

Auto-registered as "openai_responses" iff openai SDK is importable.
Default model: gpt-5-codex (coding-specialized; pricing $1.25/$10.00 per M
verified 2026-05-24 via artificialanalysis.ai).
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from code_capsules.adapters._api_agent import (
    APIBasedAgent, TOOL_DEFS, _ProviderResponse,
)


class OpenAIResponsesClient(APIBasedAgent):
    name = "openai_responses"
    default_model = "gpt-5-codex"

    def __init__(self, default_model: Optional[str] = None):
        super().__init__()
        if default_model:
            self.default_model = default_model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client

    # ── Message-shape adapters ─────────────────────────────────────────────
    # Responses API "input" is a list of items: messages, function_calls,
    # and function_call_outputs. Each turn we re-send the full list.

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
        # Re-emit the model's actions as function_call items so the next
        # request has full history. Assistant text becomes an assistant message.
        out = list(messages)
        if text:
            out.append({"role": "assistant", "content": text})
        for tc in tool_calls:
            out.append({
                "type": "function_call",
                "name": tc["name"],
                "arguments": json.dumps(tc.get("args", {})),
                "call_id": tc["call_id"],
            })
        return out

    def _append_tool_results(self, messages: list[dict],
                              results: list) -> list[dict]:
        out = list(messages)
        for tc_raw, output_text, _output_size in results:
            out.append({
                "type": "function_call_output",
                "call_id": tc_raw["call_id"],
                "output": output_text,
            })
        return out

    # ── API call ───────────────────────────────────────────────────────────

    def _call_api(self, model: str, messages: list[dict], tools: list[dict],
                  max_tokens: int) -> _ProviderResponse:
        client = self._ensure_client()
        responses_tools = [
            {
                "type": "function",
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            }
            for t in tools
        ]
        try:
            # max_output_tokens minimum is 16; bound up to caller's value
            mot = max(16, max_tokens)
            resp = client.responses.create(
                model=model,
                input=messages,
                tools=responses_tools,
                max_output_tokens=mot,
            )
        except Exception as exc:
            return _ProviderResponse(
                text="", tool_calls=[], input_tokens=0, output_tokens=0,
                stop_reason="error", error=str(exc),
            )

        text = ""
        tool_calls: list[dict] = []
        for item in resp.output:
            itype = getattr(item, "type", "")
            if itype == "function_call":
                try:
                    args = json.loads(getattr(item, "arguments", "") or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({
                    "name": item.name,
                    "args": args,
                    "call_id": item.call_id,
                })
            elif itype == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") in ("output_text", "text"):
                        text += getattr(c, "text", "") or ""
            # 'reasoning' items emitted by codex models are skipped - they're
            # internal trace the model reconstructs on follow-up turns.

        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) or 0 if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0 if usage else 0
        # OpenAI Responses API exposes cache hits under input_tokens_details.
        # input_tokens is the TOTAL (cached + fresh); cached subset is billed
        # at ~10% of fresh rate by OpenAIPublicPricing.
        cached_input_tokens = 0
        details = getattr(usage, "input_tokens_details", None) if usage else None
        if details is not None:
            cached_input_tokens = getattr(details, "cached_tokens", 0) or 0
        stop_reason = "tool_use" if tool_calls else "end_turn"

        return _ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            stop_reason=stop_reason,
        )


__all__ = ["OpenAIResponsesClient"]
