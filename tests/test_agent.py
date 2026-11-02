from dataclasses import dataclass

from sklearn_assistant.agent import prompts
from sklearn_assistant.agent.graph import MAX_REWRITES, ask, build_graph
from sklearn_assistant.agent.sandbox import extract_code, run_python

DOCS = [{"id": f"c{i}", "text": f"Title {i}\n\nbody {i}", "title": f"Title {i}",
         "url": f"https://scikit-learn.org/1.9/page{i}.html#s"} for i in range(1, 6)]


@dataclass
class FakeResult:
    chunk: dict


class FakeRetriever:
    """Returns the same 5 chunks for any query and records the queries."""

    def __init__(self):
        self.queries = []

    def retrieve(self, query, k=5, mode="hybrid"):
        self.queries.append(query)
        return [FakeResult(d) for d in DOCS[:k]]


class FakeLLM:
    """Scripted replies: a fixed category, a queue of grades, a queue of answers."""

    def __init__(self, category="howto", grades=(), answers=("An answer [1].",)):
        self.category, self.grades, self.answers = category, list(grades), list(answers)
        self.answer_prompts = []

    def structured(self, system, user, schema, effort):
        if schema is prompts.Classification:
            return schema(category=self.category)
        if schema is prompts.Grades:
            return schema(relevant=self.grades.pop(0))
        return schema(query=f"rewritten {user.count('- ')}")  # Rewrite

    def text(self, system, user, effort):
        self.answer_prompts.append(user)
        return self.answers.pop(0)


def test_relevant_docs_are_answered_and_cited_in_order():
    retriever = FakeRetriever()
    state = ask(build_graph(FakeLLM(grades=[[3, 1, 9]]), retriever), "How do I scale?")
    # Out-of-range 9 is dropped; citations are renumbered 1..n in retrieval order.
    assert [c["title"] for c in state["citations"]] == ["Title 1", "Title 3"]
    assert [c["n"] for c in state["citations"]] == [1, 2]
    assert state["trace"][0] == "classify: howto" and state["trace"][-1] == "answer"
    assert retriever.queries == ["How do I scale?"]


def test_off_topic_is_declined_without_searching():
    retriever = FakeRetriever()
    state = ask(build_graph(FakeLLM(category="off_topic"), retriever), "Best pizza?")
    assert state["answer"] == prompts.OFF_TOPIC_ANSWER
    assert retriever.queries == []


def test_query_is_rewritten_until_something_relevant_is_found():
    retriever = FakeRetriever()
    state = ask(build_graph(FakeLLM(grades=[[], [2]]), retriever), "q")
    assert retriever.queries == ["q", "rewritten 1"]
    assert [c["title"] for c in state["citations"]] == ["Title 2"]


def test_gives_up_without_guessing_after_max_rewrites():
    retriever = FakeRetriever()
    llm = FakeLLM(grades=[[]] * (MAX_REWRITES + 1))
    state = ask(build_graph(llm, retriever), "q")
    assert len(retriever.queries) == MAX_REWRITES + 1
    assert state["answer"] == prompts.NO_SOURCES_ANSWER
    assert llm.answer_prompts == []  # no LLM answer without sources


def test_failing_code_is_sent_back_for_one_fix():
    broken = "Try this [1]:\n```python\nraise ValueError('boom')\n```"
    fixed = "Try this [1]:\n```python\nprint('ok')\n```"
    llm = FakeLLM(grades=[[1]], answers=[broken, fixed])
    state = ask(build_graph(llm, FakeRetriever(), check_code=True), "q")
    assert state["code_status"] == "passed" and state["answer"] == fixed
    assert "ValueError: boom" in llm.answer_prompts[1]  # the fix prompt shows the error


def test_code_that_stays_broken_ships_with_a_warning():
    broken = "```python\nraise ValueError('boom')\n```"
    llm = FakeLLM(grades=[[1]], answers=[broken, broken])
    state = ask(build_graph(llm, FakeRetriever(), check_code=True), "q")
    assert state["code_status"] == "failed"
    assert state["answer"].endswith("ValueError: boom")


def test_sandbox_reports_success_errors_and_timeouts():
    assert run_python("print(1 + 1)").output.strip() == "2"
    failed = run_python("import sklearn\nsklearn.nope()")
    assert not failed.ok and "AttributeError" in failed.output
    assert run_python("while True: pass", timeout=1).output.startswith("Timed out")


def test_extract_code_joins_blocks_and_skips_doctests():
    md = "```python\nx = 1\n```\ntext\n```py\nprint(x)\n```\n```python\n>>> x\n```"
    assert extract_code(md) == "x = 1\n\n\nprint(x)\n"
