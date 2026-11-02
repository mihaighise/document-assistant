"""Thin wrapper around the Claude API, so graph nodes only see two calls.

`structured()` returns a validated Pydantic object (for classify/grade/rewrite) and
`text()` returns prose (for the final answer). Tests swap in a fake with the same
two methods, so the graph runs without network access or an API key.
"""

from typing import Literal, Protocol, TypeVar

import anthropic
from pydantic import BaseModel

from sklearn_assistant.config import LLM_MODEL

T = TypeVar("T", bound=BaseModel)
# Effort controls how much the model thinks before answering (and so cost/latency).
# Routing decisions are simple and use "low"; the answer uses "medium".
Effort = Literal["low", "medium", "high"]

# If a safety classifier declines a request, the API re-runs it on a fallback model it
# picks by refusal category, instead of returning the refusal to us.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMRefusal(RuntimeError):
    """The model (and its fallback) declined the request."""


class LLM(Protocol):
    """What the graph needs from a language model."""

    def structured(self, system: str, user: str, schema: type[T], effort: Effort) -> T: ...

    def text(self, system: str, user: str, effort: Effort) -> str: ...


class ClaudeLLM:
    """LLM implementation backed by the Anthropic SDK."""

    def __init__(self, model: str = LLM_MODEL, client: anthropic.Anthropic | None = None):
        # Anthropic() reads ANTHROPIC_API_KEY (or another configured credential) from the
        # environment. The SDK already retries 429/5xx/connection errors twice with backoff.
        self.client = client or anthropic.Anthropic()
        self.model = model

    def structured(self, system: str, user: str, schema: type[T], effort: Effort) -> T:
        """One call whose output is constrained to `schema`'s JSON shape."""
        # parse() turns the Pydantic class into a JSON schema, constrains decoding to it,
        # and validates the reply back into an instance (response.parsed_output).
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=2048,  # the schemas here are small (a label, a query, a few ints)
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config={"effort": effort},  # merged with the schema by parse()
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        _raise_on_refusal(response)
        return response.parsed_output

    def text(self, system: str, user: str, effort: Effort) -> str:
        """One free-text call; returns the concatenated text blocks."""
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,  # room for an explanation plus a code example
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        _raise_on_refusal(response)
        # The reply can also hold thinking blocks (empty by default); keep only text.
        return "".join(block.text for block in response.content if block.type == "text")


def _raise_on_refusal(response) -> None:
    """Check stop_reason before reading content: a refusal has no usable output."""
    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        raise LLMRefusal(f"request declined (category: {category})")
