"""The question-answering agent as a LangGraph state machine.

    classify ─┬─ off_topic ──────────────────────────────> decline ──> END
              └─> retrieve ─> grade ─┬─ relevant ─────────> answer ─┬─> END
                     ^               ├─ none, retry left ─> rewrite │
                     └───────────────┘                              ├─ code ─> check_code ─┬─> END
                                     └─ none, no retries ─> no_answer ──> END              │
                                                         answer <── fails, fix left ───────┘

Each node is a function that reads the shared state and returns only the keys it
changes; LangGraph merges them in. Routing functions pick the next node from state.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from sklearn_assistant.agent import prompts
from sklearn_assistant.agent.llm import LLM
from sklearn_assistant.agent.sandbox import extract_code, run_python
from sklearn_assistant.retrieval.hybrid import HybridRetriever

K = 5                # chunks retrieved per search, the cutoff step 3 measured (recall@5)
MAX_REWRITES = 2     # extra searches with rewritten queries before giving up
MAX_CODE_FIXES = 1   # times the answer may be regenerated after its code fails
# Step 3: dense recall@5 0.601 vs hybrid 0.553, so the agent uses dense until fusion improves.
RETRIEVAL_MODE = "dense"


class AgentState(TypedDict, total=False):
    """Everything the nodes share. total=False: keys appear as nodes fill them in."""

    question: str
    category: str
    query: str                 # current search query: the question, then rewrites
    queries: list[str]         # every query tried, so rewrites don't repeat themselves
    docs: list[dict]           # chunks from the latest search
    relevant: list[dict]       # the subset the grader kept; these are what get cited
    rewrites: int
    answer: str
    citations: list[dict]      # [{"n": 1, "title": ..., "url": ...}], numbered as in the answer
    code_error: str | None     # traceback from the last failed code run
    code_fixes: int
    code_status: str           # "passed" | "failed" | "skipped" (checker off or no code)
    # Annotated with operator.add: each node's list is *appended* to the existing one
    # instead of replacing it, giving a step-by-step log of the run.
    trace: Annotated[list[str], operator.add]


def build_graph(llm: LLM, retriever: HybridRetriever, check_code: bool = False):
    """Wire the nodes into a compiled graph. Dependencies are passed in so tests can fake them."""

    def classify(state: AgentState) -> dict:
        result = llm.structured(prompts.CLASSIFY_SYSTEM, state["question"],
                                prompts.Classification, effort="low")
        # The first search uses the question as-is; rewrites happen only if it fails.
        return {"category": result.category, "query": state["question"],
                "queries": [state["question"]], "rewrites": 0, "code_fixes": 0,
                "trace": [f"classify: {result.category}"]}

    def retrieve(state: AgentState) -> dict:
        results = retriever.retrieve(state["query"], k=K, mode=RETRIEVAL_MODE)
        docs = [r.chunk for r in results]
        return {"docs": docs, "trace": [f"retrieve: {len(docs)} chunks for {state['query']!r}"]}

    def grade(state: AgentState) -> dict:
        # One call grades all K chunks at once: cheaper and faster than one call per chunk.
        user = f"Question: {state['question']}\n\nSources:\n{prompts.format_sources(state['docs'])}"
        grades = llm.structured(prompts.GRADE_SYSTEM, user, prompts.Grades, effort="low")
        # Drop out-of-range numbers and duplicates; keep retrieval order.
        keep = sorted({n for n in grades.relevant if 1 <= n <= len(state["docs"])})
        relevant = [state["docs"][n - 1] for n in keep]
        return {"relevant": relevant,
                "trace": [f"grade: {len(relevant)}/{len(state['docs'])} relevant"]}

    def rewrite(state: AgentState) -> dict:
        tried = "\n".join(f"- {q}" for q in state["queries"])
        user = f"Question: {state['question']}\n\nQueries already tried:\n{tried}"
        result = llm.structured(prompts.REWRITE_SYSTEM, user, prompts.Rewrite, effort="low")
        return {"query": result.query, "queries": state["queries"] + [result.query],
                "rewrites": state["rewrites"] + 1, "trace": [f"rewrite: {result.query!r}"]}

    def answer(state: AgentState) -> dict:
        user = (f"Question type: {state['category']}\nQuestion: {state['question']}\n\n"
                f"Sources:\n{prompts.format_sources(state['relevant'])}")
        # Second pass after a failed code check: show the model its answer and the error.
        if state.get("code_error"):
            user += (f"\n\nYour previous answer was:\n{state['answer']}\n\n"
                     f"Its code failed with:\n{state['code_error']}\n"
                     "Rewrite the answer with code that runs.")
        text = llm.text(prompts.ANSWER_SYSTEM, user, effort="medium")
        # Citation numbers in the answer match positions in `relevant` (format_sources).
        citations = [{"n": i, "title": d["title"], "url": d["url"]}
                     for i, d in enumerate(state["relevant"], start=1)]
        return {"answer": text, "citations": citations, "code_error": None,
                "code_status": "skipped", "trace": ["answer"]}

    def check_code(state: AgentState) -> dict:
        result = run_python(extract_code(state["answer"]))
        if result.ok:
            return {"code_status": "passed", "trace": ["check_code: passed"]}
        # Count the attempt here so route_after_check sees the updated number.
        update = {"code_error": result.output, "code_status": "failed",
                  "code_fixes": state["code_fixes"] + 1, "trace": ["check_code: failed"]}
        if update["code_fixes"] > MAX_CODE_FIXES:
            # Out of fix attempts: the answer ships as-is, so warn the reader.
            last_line = result.output.strip().splitlines()[-1] if result.output.strip() else ""
            update["answer"] = (state["answer"] + "\n\n> Note: the code above failed when "
                                f"run automatically: {last_line}")
        return update

    def decline(state: AgentState) -> dict:
        return {"answer": prompts.OFF_TOPIC_ANSWER, "citations": [], "trace": ["decline"]}

    def no_answer(state: AgentState) -> dict:
        # A fixed reply rather than an LLM call: with no relevant sources, the model
        # could only answer from memory, which this assistant promises not to do.
        return {"answer": prompts.NO_SOURCES_ANSWER, "citations": [], "trace": ["no_answer"]}

    # Routing functions return the name of the next node (or END).
    def route_after_classify(state: AgentState) -> str:
        return "decline" if state["category"] == "off_topic" else "retrieve"

    def route_after_grade(state: AgentState) -> str:
        if state["relevant"]:
            return "answer"
        return "rewrite" if state["rewrites"] < MAX_REWRITES else "no_answer"

    def route_after_answer(state: AgentState) -> str:
        # Only run the checker when it's enabled and there is code to run.
        return "check_code" if check_code and extract_code(state["answer"]) else END

    def route_after_check(state: AgentState) -> str:
        if state["code_status"] == "failed" and state["code_fixes"] <= MAX_CODE_FIXES:
            return "answer"
        return END

    graph = StateGraph(AgentState)
    for name, fn in [("classify", classify), ("retrieve", retrieve), ("grade", grade),
                     ("rewrite", rewrite), ("answer", answer), ("check_code", check_code),
                     ("decline", decline), ("no_answer", no_answer)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify, ["decline", "retrieve"])
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, ["answer", "rewrite", "no_answer"])
    graph.add_edge("rewrite", "retrieve")  # the loop: search again with the new query
    graph.add_conditional_edges("answer", route_after_answer, ["check_code", END])
    graph.add_conditional_edges("check_code", route_after_check, ["answer", END])
    graph.add_edge("decline", END)
    graph.add_edge("no_answer", END)
    return graph.compile()


def ask(graph, question: str) -> AgentState:
    """Run one question through a compiled graph and return the final state."""
    return graph.invoke({"question": question, "trace": []})
