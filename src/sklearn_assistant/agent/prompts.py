"""Prompts and output schemas for the agent's LLM calls.

Kept apart from the graph so wording can be tuned without touching control flow.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

# Same labels as the eval set (step 3) plus off_topic, so step 5's trained classifiers
# can be scored against the same categories.
Category = Literal["howto", "concept", "api", "comparison", "troubleshooting", "off_topic"]


# extra="forbid" makes the generated JSON schema set additionalProperties: false, which
# structured outputs require.
class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Category


class Grades(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relevant: list[int]  # 1-based numbers of the sources that help answer the question


class Rewrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str


CLASSIFY_SYSTEM = """\
You route questions for a scikit-learn documentation assistant. Pick one category:
- howto: how to accomplish a task ("How do I scale features?")
- concept: how or why something works ("What is successive halving?")
- api: a specific parameter, attribute, method or default ("What does warm_start do?")
- comparison: the difference between options ("Ridge vs Lasso?")
- troubleshooting: an error, warning or unexpected result
- off_topic: not about scikit-learn or machine learning with it"""

GRADE_SYSTEM = """\
You judge search results for a scikit-learn documentation assistant. Given a question \
and numbered sources, return the numbers of the sources that contain information needed \
to answer it. A source that only mentions the topic in passing is not relevant. Return \
an empty list if none are."""

REWRITE_SYSTEM = """\
A search over the scikit-learn documentation found nothing relevant for a question. \
Write a new search query that is more likely to match how the docs describe the topic: \
use scikit-learn's own terms, class and parameter names (e.g. "GroupKFold" for "keep a \
patient's samples in one fold"). Return only the query, and don't repeat earlier ones."""

ANSWER_SYSTEM = """\
You are a scikit-learn documentation assistant. Answer using only the numbered sources \
provided. Cite each claim with its source number in brackets, like [1] or [2][3]. If the \
sources don't fully answer the question, say what is missing instead of guessing.

Be concise. When code helps, include one complete, runnable ```python example that uses \
a built-in dataset (e.g. load_iris, make_classification) and needs nothing else to run."""

NO_SOURCES_ANSWER = (
    "I couldn't find this in the scikit-learn documentation, so I won't guess. "
    "Try rephrasing with the name of the estimator or function involved."
)

OFF_TOPIC_ANSWER = (
    "I can only answer questions about scikit-learn, using its official documentation."
)


def format_sources(docs: list[dict]) -> str:
    """Number the chunks so the model can cite them as [1], [2], ..."""
    # Chunk text already starts with its breadcrumb title (see the chunker), so the
    # model sees where each source comes from.
    return "\n\n".join(f"[{i}] {d['text']}" for i, d in enumerate(docs, start=1))
