"""
Built-in CostModel implementations — one per vendor at public-pricing rates.

Shipped:
  - AnthropicPublicPricing   (Claude family)
  - OpenAIPublicPricing       (GPT-4o / o1 / o3 family)
  - GeminiPublicPricing       (Gemini 2.0 / 2.5 family)

All use the public, listed-price rates as of `pricing_date`. Users should
write their own subclass / replacement for:
  - prompt caching (often 50-90% off input on cache hits)
  - batch API discounts (often 50% off)
  - enterprise / committed-use contracts
  - regional pricing variation
  - model versions newer than the stamped date

Each CostModel conforms to the CostModel Protocol from code_capsules.api.
"""
from __future__ import annotations


# ── Anthropic ────────────────────────────────────────────────────────────────

# Public API pricing as of pricing_date. Update when pricing changes.
# Source: anthropic.com/pricing (public page).
_ANTHROPIC_PUBLIC_USD_PER_M = {
    "claude-haiku-4-5":   {"input": 1.00,  "output": 5.00},
    "haiku-4-5":          {"input": 1.00,  "output": 5.00},
    "haiku":              {"input": 1.00,  "output": 5.00},

    "claude-sonnet-4-6":  {"input": 3.00,  "output": 15.00},
    "sonnet-4-6":         {"input": 3.00,  "output": 15.00},
    "sonnet":             {"input": 3.00,  "output": 15.00},

    "claude-opus-4-7":    {"input": 15.00, "output": 75.00},
    "opus-4-7":           {"input": 15.00, "output": 75.00},
    "opus":               {"input": 15.00, "output": 75.00},
}


class AnthropicPublicPricing:
    """Public Anthropic API rates as of pricing_date.

    Caveats users should override for:
    - prompt caching: input cost can drop 90% on cache hits
    - batch API: 50% discount for async batched requests
    - enterprise contracts: negotiated rates
    - pricing changes: update pricing_date + rates when Anthropic changes them
    """

    name = "anthropic_public"
    pricing_date = "2026-05-27"

    # Anthropic prompt-cache reads are billed at 10% of fresh input rate.
    # Cache writes are 1.25x — we don't differentiate here; assume reads.
    _CACHE_DISCOUNT = 0.10

    def cost(self, input_tokens: int, output_tokens: int, model: str,
             cached_input_tokens: int = 0) -> float:
        rates = _ANTHROPIC_PUBLIC_USD_PER_M.get(model)
        if rates is None:
            short = model.lower().split("-2025")[0].split("-2026")[0]
            rates = _ANTHROPIC_PUBLIC_USD_PER_M.get(short)
        if rates is None:
            rates = _ANTHROPIC_PUBLIC_USD_PER_M["sonnet"]  # conservative fallback
        cached = max(0, min(cached_input_tokens, input_tokens))
        fresh = input_tokens - cached
        rate_in = rates["input"] / 1_000_000.0
        return (fresh * rate_in
                + cached * rate_in * self._CACHE_DISCOUNT
                + output_tokens * rates["output"] / 1_000_000.0)


# ── OpenAI ───────────────────────────────────────────────────────────────────

# Public API pricing as of pricing_date. Source: platform.openai.com/docs/pricing.
# Reference rates for general agent-runtime use; verify before deployment.
_OPENAI_PUBLIC_USD_PER_M = {
    # GPT-5 family (current generation as of pricing_date).
    # Verified 2026-05-24 against artificialanalysis.ai. Cached-input rates
    # noted alongside — about 90% off for GPT-5 family. Users with prompt
    # caching should subclass CostModel to honor that.
    "gpt-5":              {"input": 1.25,  "output": 10.00},   # cached input $0.125
    "gpt-5-codex":        {"input": 1.25,  "output": 10.00},   # cached input $0.125
    "gpt-5.1":            {"input": 1.25,  "output": 10.00},
    "gpt-5.1-codex":      {"input": 1.25,  "output": 10.00},
    # gpt-5.2+ assumed to hold the gpt-5 flagship tier (approx — verify on use).
    "gpt-5.5":            {"input": 1.25,  "output": 10.00},   # approx (gpt-5 tier; verify)
    "gpt-5.5-pro":        {"input": 5.00,  "output": 25.00},   # approx (pro tier; verify)
    "gpt-5-mini":         {"input": 0.25,  "output": 2.00},    # cached input $0.03
    "gpt-5-nano":         {"input": 0.05,  "output": 0.40},    # output rate approx
    "gpt-5-pro":          {"input": 5.00,  "output": 25.00},   # pricing approx (verify)

    # GPT-4o family (legacy as of pricing_date but still available)
    "gpt-4o":             {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":        {"input": 0.15,  "output": 0.60},

    # o1 / o3 reasoning family (legacy)
    "o1":                 {"input": 15.00, "output": 60.00},
    "o1-mini":            {"input": 3.00,  "output": 12.00},
    "o3":                 {"input": 10.00, "output": 40.00},
    "o3-mini":            {"input": 1.10,  "output": 4.40},
}


class OpenAIPublicPricing:
    """Public OpenAI API rates as of pricing_date.

    For agent runtimes that drive OpenAI models (custom tool loop, Assistants
    API). Override for: cached input pricing (50% on hits >=1024 tokens),
    batch API (50% off), enterprise contracts.
    """

    name = "openai_public"
    pricing_date = "2026-05-27"

    # OpenAI's automatic prompt cache discounts cached input to 10% of fresh
    # rate (verified via artificialanalysis.ai for the GPT-5 family).
    _CACHE_DISCOUNT = 0.10

    def cost(self, input_tokens: int, output_tokens: int, model: str,
             cached_input_tokens: int = 0) -> float:
        rates = _OPENAI_PUBLIC_USD_PER_M.get(model)
        if rates is None:
            short = model.lower().split("-2024")[0].split("-2025")[0].split("-preview")[0]
            rates = _OPENAI_PUBLIC_USD_PER_M.get(short)
        if rates is None:
            rates = _OPENAI_PUBLIC_USD_PER_M["gpt-4o"]
        cached = max(0, min(cached_input_tokens, input_tokens))
        fresh = input_tokens - cached
        rate_in = rates["input"] / 1_000_000.0
        return (fresh * rate_in
                + cached * rate_in * self._CACHE_DISCOUNT
                + output_tokens * rates["output"] / 1_000_000.0)


# ── Google Gemini ────────────────────────────────────────────────────────────

# Public API pricing as of pricing_date. Source: ai.google.dev/pricing.
_GEMINI_PUBLIC_USD_PER_M = {
    "gemini-2.0-flash":      {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075,"output": 0.30},
    "gemini-2.5-pro":        {"input": 1.25, "output": 10.00},  # ≤200k context tier
    "gemini-2.5-flash":      {"input": 0.30, "output": 2.50},
    # Gemini 3.x (approx — held at the 2.5-pro tier pending verification).
    "gemini-3.1-pro-preview":{"input": 1.25, "output": 10.00},  # approx (verify)
    "gemini-3.5-flash":      {"input": 0.30, "output": 2.50},   # approx (verify)
}


class GeminiPublicPricing:
    """Public Google Gemini API rates as of pricing_date.

    Gemini 2.5 Pro uses a tiered price; this class returns the ≤200k-context
    rate. Override for: long-context premium tier, batch API discounts.
    """

    name = "gemini_public"
    pricing_date = "2026-05-27"

    # Gemini 2.5 family (Flash + Pro) context-cache reads are 10% of fresh
    # input rate per ai.google.dev/gemini-api/docs/pricing (verified
    # 2026-05-27): Flash cache $0.03 vs fresh $0.30; Pro cache $0.125 vs
    # fresh $1.25. Earlier Gemini 1.5 ~25% rate no longer applies.
    _CACHE_DISCOUNT = 0.10

    def cost(self, input_tokens: int, output_tokens: int, model: str,
             cached_input_tokens: int = 0) -> float:
        rates = _GEMINI_PUBLIC_USD_PER_M.get(model)
        if rates is None:
            short = model.lower().split("-exp")[0].split("-preview")[0]
            rates = _GEMINI_PUBLIC_USD_PER_M.get(short)
        if rates is None:
            rates = _GEMINI_PUBLIC_USD_PER_M["gemini-2.5-flash"]
        cached = max(0, min(cached_input_tokens, input_tokens))
        fresh = input_tokens - cached
        rate_in = rates["input"] / 1_000_000.0
        return (fresh * rate_in
                + cached * rate_in * self._CACHE_DISCOUNT
                + output_tokens * rates["output"] / 1_000_000.0)


__all__ = [
    "AnthropicPublicPricing",
    "OpenAIPublicPricing",
    "GeminiPublicPricing",
]
