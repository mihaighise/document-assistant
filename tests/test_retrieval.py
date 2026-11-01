from sklearn_assistant.retrieval.bm25 import BM25Index, tokenize
from sklearn_assistant.retrieval.hybrid import rrf


def test_identifiers_are_indexed_whole_and_in_parts():
    assert tokenize("fit_transform") == ["fit_transform", "fit", "transform"]
    assert tokenize("GridSearchCV") == ["gridsearchcv", "grid", "search", "cv"]
    # Single-letter parts and stopwords are dropped.
    assert tokenize("the n_jobs") == ["n_jobs", "jobs"]


def test_dotted_paths_split_into_modules_and_names():
    tokens = tokenize("sklearn.model_selection.GridSearchCV")
    assert {"sklearn", "model_selection", "model", "gridsearchcv", "search"} <= set(tokens)


def test_bm25_prefers_exact_identifier():
    chunks = [
        {"id": "ft", "text": "fit_transform fits the transformer then transforms X"},
        {"id": "t", "text": "transform applies an already fitted transformer"},
        {"id": "x", "text": "RandomForestClassifier supports class_weight"},
    ]
    results = BM25Index(chunks).search("what does fit_transform do", k=3)
    assert results[0][0] == "ft"
    assert "x" not in [cid for cid, _ in results]


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf([["a", "b", "c"], ["b", "d"]])
    ids = [cid for cid, _ in fused]
    # "b" is 2nd and 1st, beating "a" which only one retriever found.
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c", "d"}
    assert fused[0][1] == 1 / 62 + 1 / 61
