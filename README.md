# document-assistant
Document Assistant for scikit-learn using RAG + LangChain + LangGraph.

Answers questions like *"how do I handle class imbalance in RandomForest?"* or *"what's the
difference between fit_transform and transform?"* using the official scikit-learn docs, with
citations to the exact page section.

## Setup

Requires Python 3.11 or 3.12 (PyTorch, needed later for fine-tuning, has no wheels for newer
versions yet).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

## Pipeline

### 1. Ingest + section-based chunking

```bash
.venv/bin/python scripts/ingest.py
```

Downloads the official offline HTML docs and writes section-level chunks to
`data/processed/chunks.jsonl` (`data/` is gitignored).

**Design decisions**

- **HTML build, not the `.rst` sources on GitHub.** The API reference pages (e.g. every
  parameter of `GridSearchCV`) are generated from docstrings at build time, so they don't exist
  in the `.rst` files. The HTML also gives real URLs and section anchors to cite.
- **Version-pinned citations.** The docs are v1.9.1, and citation URLs point at
  `https://scikit-learn.org/1.9/...` rather than `/stable/`, so links keep matching the indexed
  text when a new release comes out.
- **Chunk by section, not by fixed size.** Sphinx wraps every heading in a
  `<section id="...">`, so each section becomes a chunk with URL `page.html#section-id`.
  Subsection text goes into its own chunk, not the parent's. Each chunk's text starts with its
  breadcrumb title (e.g. `Ensembles > Gradient-boosted trees > Usage`) so the embedding sees
  the context.
- **Special cases:**
  - API pages: one chunk for the class, one per method (`GridSearchCV > fit`).
  - Glossary: one chunk per term (`fit_transform`).
  - Code blocks are kept verbatim inside ``` fences.
  - Gallery script output is truncated (some examples print pages of warnings).
  - Sections over ~1500 tokens are split on paragraph boundaries; near-empty sections
    (headings, link lists) are dropped.
- **Excluded pages:** release notes, developer/contributor guides, team and governance pages,
  install instructions — they're about the project, not about using the library.

**Result:** 5,374 chunks — 3,041 API, 1,458 gallery examples, 681 user guide, 194 glossary.
Median chunk ~870 characters, max 6,194.

### 2. Hybrid retrieval (planned)

```bash
.venv/bin/python scripts/build_index.py
```

Builds a dense index (embeddings in Chroma) and a lexical index (BM25) over the same chunks,
and merges their results at query time.

```mermaid
flowchart LR
    subgraph build["Index build (once)"]
        C[(chunks.jsonl)] --> E[Embed with<br/>bge-small-en-v1.5]
        E --> V[(Chroma<br/>data/chroma/)]
        C --> T[Code-aware<br/>tokenizer]
        T --> B[(BM25 index)]
    end

    subgraph query["Query time"]
        Q[/Question/] --> QE[Embed question]
        Q --> QT[Tokenize question]
        QE -->|top 20| V
        QT -->|top 20| B
        V --> F[Reciprocal Rank Fusion<br/>score = Σ 1 / 60 + rank]
        B --> F
        F --> R[/Top-k chunks with<br/>title, url, page, doc_type/]
    end
```

**Why both searches**

| | Dense (embeddings) | Lexical (BM25) |
|---|---|---|
| Matches | meaning: *"skewed classes"* → `class_weight` | exact names: `fit_transform`, `n_jobs` |
| Misses | close identifiers (`fit_transform` vs. `transform`) | synonyms and paraphrases |

**Design decisions**

- **Local embedding model (`BAAI/bge-small-en-v1.5`).** No API key or cost, re-indexing is
  free, and step 6 fine-tunes the embeddings, which needs an open model.
- **Code-aware BM25 tokenization.** Identifiers are indexed whole and split into parts
  (`fit_transform` → `fit_transform`, `fit`, `transform`), so both exact and partial matches score.
- **Reciprocal Rank Fusion.** Uses only ranks, so BM25 scores and cosine similarities never
  need to be put on the same scale.
- **Chunk length vs. model limit.** BGE truncates input at 512 tokens, so the chunker's split
  threshold drops from ~1500 to ~450 tokens to keep every chunk fully embedded.
- **Plain libraries behind one `retrieve(query, k)` function.** Step 3's evaluation gets
  full control over retrieval; step 4 wraps it as a LangChain retriever.

## Roadmap

- [x] 1. Ingest + section-based chunking
- [ ] 2. Embeddings + Chroma, BM25 + hybrid retrieval
- [ ] 3. Evaluation set (~75 questions) + recall@5 / MRR, tracked in MLflow
- [ ] 4. LangGraph agent: classify → retrieve → grade/rewrite → answer (+ code sandbox)
- [ ] 5. Question classifier: TF-IDF + LR vs. fine-tuned BERT vs. LLM
- [ ] 6. Fine-tuned embeddings + cross-encoder reranker
- [ ] 7. Answer evaluation (faithfulness, correctness)
- [ ] 8. FastAPI + Docker, Streamlit UI, tracing

The evaluation set comes right after retrieval (before the agent) so the plain-embeddings
baseline gets measured first, and every later improvement shows up as a number in the final
comparison table.

## Layout

```
src/sklearn_assistant/
  config.py            paths, docs URL, citation base URL
  ingest/download.py   fetch + unzip the offline docs
  ingest/chunker.py    HTML -> section-level chunks
  retrieval/dense.py   embed chunks -> Chroma; query            (planned)
  retrieval/bm25.py    code-aware tokenizer + BM25 index        (planned)
  retrieval/hybrid.py  RRF fusion -> ranked chunks              (planned)
scripts/ingest.py      step 1 entry point
scripts/build_index.py step 2 entry point                       (planned)
tests/                 pytest suite
```
