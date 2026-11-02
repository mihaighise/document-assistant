"""Thin wrapper around the Claude API, so graph nodes only see two calls.

`structured()` returns a validated Pydantic object (for classify/grade/rewrite) and
`text()` returns prose (for the final answer). Tests swap in a fake with the same
two methods, so the graph runs without network access or an API key.

How an LLM call works, in short:
- The model is a remote service. Each call is one HTTPS request to Anthropic's
  Messages API: we send text, the model generates a reply, we get it back. Nothing
  is remembered between calls, so every call must include all the context it needs
  (here: the question plus the retrieved doc chunks).
- A request has a *system prompt* (standing instructions: the role and rules, e.g.
  "answer only from the sources and cite them") and *messages* (the conversation; here
  always a single user message holding the actual input).
- Text is processed in *tokens* (roughly ¾ of a word). Cost is per token, for both what
  we send (input) and what the model writes (output), and `max_tokens` caps the output.
"""

from typing import Literal, Protocol, TypeVar

import anthropic
from pydantic import BaseModel

from sklearn_assistant.config import LLM_MODEL

# T stands for "whichever Pydantic class the caller passes in". It lets type checkers know
# that structured(..., schema=Grades) returns a Grades object, not just "some BaseModel".
T = TypeVar("T", bound=BaseModel)
# Effort controls how much the model thinks before answering (and so cost/latency).
# Routing decisions are simple and use "low"; the answer uses "medium".
# (The model reasons internally before writing its reply; more effort = more reasoning
# tokens, which are billed like output tokens, but usually better results on hard tasks.)
Effort = Literal["low", "medium", "high"]

# If a safety classifier declines a request, the API re-runs it on a fallback model it
# picks by refusal category, instead of returning the refusal to us.
# This feature is in beta, so it has to be switched on per request with this name.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMRefusal(RuntimeError):
    """The model (and its fallback) declined the request."""


# A Protocol describes an interface: any class with these two methods counts as an LLM,
# without inheriting from anything. graph.py only depends on this interface, which is
# what lets the tests pass in FakeLLM (scripted replies) instead of ClaudeLLM.
class LLM(Protocol):
    """What the graph needs from a language model."""

    def structured(self, system: str, user: str, schema: type[T], effort: Effort) -> T: ...

    def text(self, system: str, user: str, effort: Effort) -> str: ...


class ClaudeLLM:
    """LLM implementation backed by the Anthropic SDK."""

    def __init__(self, model: str = LLM_MODEL, client: anthropic.Anthropic | None = None):
        # Anthropic() reads ANTHROPIC_API_KEY (or another configured credential) from the
        # environment. The SDK already retries 429/5xx/connection errors twice with backoff.
        # (config.py has already copied the key from .env into the environment by now.)
        # One client is reused for every call; it keeps the HTTP connection open.
        self.client = client or anthropic.Anthropic()
        self.model = model

    def structured(self, system: str, user: str, schema: type[T], effort: Effort) -> T:
        """One call whose output is constrained to `schema`'s JSON shape."""
        # Why structured output: the graph must *act* on these replies (route to "rewrite",
        # keep chunks 1 and 3). Free text like "Sources 1 and 3 look relevant" would need
        # fragile parsing. Instead the reply is forced to be JSON matching the schema,
        # e.g. {"relevant": [1, 3]}, and arrives as a ready-to-use Python object.
        # parse() turns the Pydantic class into a JSON schema, constrains decoding to it,
        # and validates the reply back into an instance (response.parsed_output).
        response = self.client.beta.messages.parse(
            model=self.model,          # which Claude model answers
            max_tokens=2048,  # the schemas here are small (a label, a query, a few ints)
            system=system,             # the standing instructions for this kind of call
            # The input itself, as the one message of a fresh conversation.
            messages=[{"role": "user", "content": user}],
            output_format=schema,      # the Pydantic class the reply must match
            output_config={"effort": effort},  # merged with the schema by parse()
            betas=[FALLBACK_BETA],     # switch on the beta feature used on the next line
            fallbacks="default",       # let the API pick a fallback model on refusal
        )
        _raise_on_refusal(response)
        return response.parsed_output

    def text(self, system: str, user: str, effort: Effort) -> str:
        """One free-text call; returns the concatenated text blocks."""
        # Same request as above minus the schema: the reply is prose (markdown with
        # citations and maybe a code block), which is shown to the user as-is.
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
        # A reply is a list of content blocks, not a single string.
        # The reply can also hold thinking blocks (empty by default); keep only text.
        return "".join(block.text for block in response.content if block.type == "text")


def _raise_on_refusal(response) -> None:
    """Check stop_reason before reading content: a refusal has no usable output."""
    # stop_reason says why generation ended: "end_turn" (finished normally), "max_tokens"
    # (hit the cap), or "refusal" (declined even after the fallback). Only the last one
    # leaves us with nothing to use, so it becomes an exception the caller can see.
    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        raise LLMRefusal(f"request declined (category: {category})")
