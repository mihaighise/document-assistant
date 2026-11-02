"""Step 5: score Claude as the question classifier, on the same test set as TF-IDF + LR.

    python scripts/eval_llm_classifier.py                 # Opus 5.5 and Haiku 5.5
    python scripts/eval_llm_classifier.py --models haiku

Runs the agent's exact classify call (same prompt, schema and effort as the `classify`
node) once per test question, and logs accuracy, macro-F1, latency and measured cost
to the MLflow experiment "classifier". There is no training: the LLM only reads the
category definitions in the prompt. Needs ANTHROPIC_API_KEY (see .env.example).
"""

import argparse
import hashlib

from sklearn_assistant.agent import prompts
from sklearn_assistant.agent.llm import ClaudeLLM, LLMRefusal
from sklearn_assistant.classifier.data import load_test
from sklearn_assistant.evaluation.classifier import evaluate, print_report
from sklearn_assistant.tracking import log_classifier_run

MODELS = {"opus": "claude-opus-5-5", "haiku": "claude-haiku-5-5"}
EFFORT = "low"  # same as the classify node in agent/graph.py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = parser.parse_args()
    test = load_test()

    results = {}
    for name in args.models:
        model = MODELS[name]
        # Haiku 5.5 has no server-side fallback (the API rejects the parameter for it).
        llm = ClaudeLLM(model=model, fallback=name != "haiku")

        def predict(question: str) -> str:
            try:
                return llm.structured(prompts.CLASSIFY_SYSTEM, question,
                                      prompts.Classification, effort=EFFORT).category
            except LLMRefusal:
                # Counted as a wrong answer rather than crashing the whole run.
                return "refused"

        print(f"\n=== {model} ({len(test)} calls, one at a time) ===")
        # Questions go one at a time (no threads) so latency matches the agent, where
        # each user question is classified on its own.
        summary, rows, matrix = evaluate(predict, test)
        # Cost from the token counts the API reported for these exact calls.
        cost_per_1k = llm.cost_usd() / llm.calls * 1000
        metrics = {**summary, "cost_per_1k_usd": cost_per_1k, "total_cost_usd": llm.cost_usd(),
                   "input_tokens_per_call": llm.input_tokens / llm.calls,
                   "output_tokens_per_call": llm.output_tokens / llm.calls}
        log_classifier_run(
            f"llm_{name}",
            # The prompt's hash flags runs made with different category wording.
            params={"model": model, "effort": EFFORT, "n_test": len(test),
                    "prompt_sha": hashlib.sha1(prompts.CLASSIFY_SYSTEM.encode()).hexdigest()[:12]},
            metrics=metrics, rows=rows, matrix=matrix,
        )
        print_report(summary, rows, matrix)
        print(f"\nTokens per call: {metrics['input_tokens_per_call']:.0f} in, "
              f"{metrics['output_tokens_per_call']:.0f} out. Cost: "
              f"${llm.cost_usd():.4f} for this run, ${cost_per_1k:.2f} per 1k questions")
        results[name] = metrics

    print("\nmodel   accuracy  macro-F1  latency_ms  $/1k")
    for name, m in results.items():
        print(f"{name:7} {m['accuracy']:8.3f} {m['macro_f1']:9.3f} {m['latency_ms']:11.0f} "
              f"{m['cost_per_1k_usd']:6.2f}")


if __name__ == "__main__":
    main()
