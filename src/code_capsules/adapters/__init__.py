"""
ModelClient implementations shipped with Code-Capsules.

Four impls:
  - ClaudeCLIClient      - drives Anthropic's Claude CLI via subprocess (the
                           runtime calibrated in the paper).
  - OpenAIAPIClient      - drives a GPT model via the openai SDK with a
                           multi-turn tool loop (Read/Write/Edit/Bash).
                           Auto-registered iff `openai` is installed.
  - GeminiAPIClient      - drives a Gemini model via google.genai SDK with
                           the same tool loop. Auto-registered iff
                           `google.genai` is installed.
  - GenericAgentClient   - skeleton/contract documentation. NOT auto-registered;
                           users instantiate with a custom invoke() and register
                           manually.

To add your own (Aider, OpenAI Codex CLI, local llama.cpp, MCP-based agent):

    from code_capsules.api import InvocationResult
    from code_capsules.api.registry import register

    class MyAgentClient:
        name = "my_agent"
        def invoke(self, prompt, *, cwd, max_turns, **kwargs) -> InvocationResult:
            # ...your agent runtime here...
            return InvocationResult(text=..., tool_calls=..., ...)

    register("model_client", MyAgentClient())

Then in policy.yaml or VariantConfig.extra, reference your client by name:

    extra:
      model_client: my_agent
"""
from __future__ import annotations

from code_capsules.adapters.claude_cli import ClaudeCLIClient
from code_capsules.adapters.generic_agent import GenericAgentClient


def register_builtins() -> None:
    """Register the framework's shipped model clients.

    Anthropic CLI client is always registered. OpenAI / Gemini clients are
    registered if and only if their vendor SDK is importable - keeps the
    framework light for users who don't need cross-vendor (paper §"Cross-vendor
    extension via ModelClient"). The vendor SDK import is deferred to
    ``_ensure_client`` (so the adapter modules import without the SDK), which
    means a module-level try/except can't enforce the "iff importable"
    contract; ``importlib.util.find_spec`` is what gates registration.
    """
    import importlib.util

    def _importable(module: str) -> bool:
        # find_spec on a dotted name imports the parent package, which raises
        # ModuleNotFoundError when the parent is absent (e.g. "google.genai"
        # with no "google" installed - the default-install state). Treat any
        # import failure as "not available".
        try:
            return importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            return False

    from code_capsules.api.registry import register
    register("model_client", ClaudeCLIClient())

    if _importable("openai"):
        from code_capsules.adapters.openai_api import OpenAIAPIClient
        from code_capsules.adapters.openai_responses import OpenAIResponsesClient
        register("model_client", OpenAIAPIClient())          # chat-completions
        register("model_client", OpenAIResponsesClient())    # responses API (gpt-5-codex)

    if _importable("google.genai"):
        from code_capsules.adapters.gemini_api import GeminiAPIClient
        register("model_client", GeminiAPIClient())


__all__ = ["ClaudeCLIClient", "GenericAgentClient", "register_builtins"]
