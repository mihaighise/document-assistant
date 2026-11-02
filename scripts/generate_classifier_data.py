"""Step 5: generate labeled training questions for the question classifier with Claude.

    python scripts/generate_classifier_data.py            # ~1,800 questions, ~120 API calls

Writes datasets/classifier/train.jsonl, which is committed: LLM output isn't
reproducible, so the file *is* the dataset. Needs ANTHROPIC_API_KEY (see .env.example).

Why generate: the only hand-labeled questions are the 81 in the retrieval eval set,
which are too few to train on and must stay unseen as the test set.
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, ConfigDict

from sklearn_assistant.agent.llm import ClaudeLLM
from sklearn_assistant.agent.prompts import CATEGORY_DEFINITIONS
from sklearn_assistant.classifier.data import CATEGORIES, drop_leakage, load_test
from sklearn_assistant.config import CLASSIFIER_TRAIN_PATH
from sklearn_assistant.retrieval.hybrid import load_chunks

PER_CALL = 15        # questions per API call
CALLS_PER_CATEGORY = 20
TOPICS_PER_CALL = 8

SYSTEM = f"""\
You write realistic user questions for training a question classifier for a scikit-learn \
documentation assistant. The categories are:
{CATEGORY_DEFINITIONS}

Write questions the way real users type them. Vary length (3 to 40 words), tone and \
phrasing. Some name scikit-learn classes, functions or parameters; many describe the \
problem in plain words instead. A few are terse, lowercase or have small typos. Every \
question must clearly belong to the requested category."""

# Off-topic questions can't be seeded from the docs, so they rotate through themes.
# Most are near misses (other Python/data tools a scikit-learn user also touches), which
# are the hard cases; the rest are plainly unrelated.
OFF_TOPIC_THEMES = [
    "pandas data wrangling", "numpy array operations", "matplotlib and seaborn plotting",
    "PyTorch or TensorFlow deep learning", "LLMs, Hugging Face and prompt engineering",
    "Python language basics and packaging", "SQL and databases", "git, Docker and deployment",
    "web development", "general knowledge, travel, cooking or small talk",
]


class Generated(BaseModel):
    model_config = ConfigDict(extra="forbid")
    questions: list[str]


def user_prompt(category: str, rng: random.Random, titles: list[str], call: int) -> str:
    """The request for one call: which category, and topics to keep the questions varied."""
    if category == "off_topic":
        themes = [OFF_TOPIC_THEMES[(call + i) % len(OFF_TOPIC_THEMES)] for i in range(3)]
        return (f"Write {PER_CALL} off_topic questions, spread over these themes: "
                f"{', '.join(themes)}. Don't mention scikit-learn.")
    # Seeding with random doc section titles stops every call from asking about the
    # same famous topics (cross-validation, random forests) and makes the questions
    # cover the corpus the assistant actually answers from.
    topics = "\n".join(f"- {t}" for t in rng.sample(titles, TOPICS_PER_CALL))
    return (f"Write {PER_CALL} {category} questions. Base them on these documentation "
            f"topics (use some, not necessarily all):\n{topics}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls-per-category", type=int, default=CALLS_PER_CATEGORY)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rng = random.Random(0)
    # Breadcrumb titles like "Ensembles > Gradient-boosted trees > Usage" (minus the
    # "(part 2/3)" suffix of split sections), deduplicated.
    titles = sorted({c["title"].split(" (part ")[0] for c in load_chunks()})
    jobs = [(cat, user_prompt(cat, rng, titles, i))
            for cat in CATEGORIES for i in range(args.calls_per_category)]

    llm = ClaudeLLM()
    rows: list[dict] = []
    # The calls are independent and each mostly waits on the network, so threads run
    # several at once (~8x faster than one after another).
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(llm.structured, SYSTEM, prompt, Generated, "low"): cat
                   for cat, prompt in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            cat = futures[future]
            try:
                rows += [{"question": q.strip(), "category": cat}
                         for q in future.result().questions if q.strip()]
            except Exception as e:  # one failed call shouldn't lose the other ~119
                print(f"  call failed ({cat}): {e}")
            print(f"\r{done}/{len(jobs)} calls", end="", flush=True)
    print()

    # Exact duplicates (ignoring case and spacing), keeping the first label seen.
    seen, unique = set(), []
    for r in rows:
        key = " ".join(r["question"].lower().split())
        if key not in seen:
            seen.add(key)
            unique.append(r)
    rows = unique
    kept = drop_leakage(rows, [q.question for q in load_test()])
    print(f"{len(rows)} unique questions, {len(rows) - len(kept)} dropped as near-copies "
          f"of test questions, {len(kept)} kept")

    rng.shuffle(kept)
    CLASSIFIER_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASSIFIER_TRAIN_PATH, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    for cat in CATEGORIES:
        print(f"  {cat:16} {sum(r['category'] == cat for r in kept)}")


if __name__ == "__main__":
    main()
