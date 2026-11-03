"""Step 4: ask the agent a question from the command line.

    python scripts/ask.py "How do I handle class imbalance in RandomForest?"
    python scripts/ask.py --check-code "How do I plot a learning curve?"

Needs ANTHROPIC_API_KEY in .env (see .env.example) or in the environment.
"""

import argparse

from sklearn_assistant.agent.graph import ask, build_graph
from sklearn_assistant.agent.llm import ClaudeLLM
from sklearn_assistant.config import CLASSIFIER_MODEL
from sklearn_assistant.retrieval.hybrid import HybridRetriever


def main() -> None:
    """Answer one question, then print the sources and the steps the agent took."""
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--check-code", action="store_true",
                        help="run the answer's code locally and fix it once if it fails "
                             "(executes model-written code on this machine)")
    args = parser.parse_args()

    classifier_llm = ClaudeLLM(model=CLASSIFIER_MODEL, fallback=False)

    graph = build_graph(ClaudeLLM(), HybridRetriever(),
                        check_code=args.check_code, classifier_llm=classifier_llm)
    state = ask(graph, args.question)

    print(state["answer"])
    if state.get("citations"):
        print("\nSources:")
        for c in state["citations"]:
            print(f"  [{c['n']}] {c['title']}\n      {c['url']}")
    if args.check_code:
        print(f"\nCode check: {state.get('code_status', 'no_code')}")
    print("\nSteps: " + " -> ".join(state["trace"]))


if __name__ == "__main__":
    main()
