from sklearn_assistant.ingest.chunker import chunk_page, is_included

LONG = "Some explanatory text about the estimator. " * 5

PAGE = f"""
<html><body><article class="bd-article">
<section id="ensembles"><h1>Ensembles<a class="headerlink">#</a></h1>
  <p>{LONG}</p>
  <section id="usage"><h2>Usage</h2>
    <p>{LONG}</p>
    <div class="highlight"><pre>&gt;&gt;&gt; clf.fit(X,
...     y)</pre></div>
  </section>
</section>
</article></body></html>
"""


def test_sections_become_separate_chunks_with_anchors():
    chunks = chunk_page(PAGE, "modules/ensemble.html")
    assert [c.url.split("#")[1] for c in chunks] == ["ensembles", "usage"]
    assert chunks[1].title == "Ensembles > Usage"
    # Parent chunk must not swallow the subsection's text.
    assert "clf.fit" not in chunks[0].text


def test_code_blocks_keep_their_whitespace():
    usage = chunk_page(PAGE, "modules/ensemble.html")[1]
    assert "```python\n>>> clf.fit(X,\n...     y)\n```" in usage.text


def test_api_methods_are_split_out():
    html = f"""<article class="bd-article"><section id="gs"><h1>GridSearchCV</h1>
      <dl class="py class"><dt id="sklearn.GridSearchCV">GridSearchCV</dt><dd><p>{LONG}</p>
        <dl class="py method"><dt id="sklearn.GridSearchCV.fit">fit(X)</dt>
          <dd><p>{LONG}</p></dd></dl>
      </dd></dl></section></article>"""
    chunks = chunk_page(html, "modules/generated/sklearn.GridSearchCV.html")
    titles = {c.title for c in chunks}
    assert titles == {"GridSearchCV", "GridSearchCV > fit"}
    assert all(c.doc_type == "api" for c in chunks)


def test_project_pages_are_excluded():
    assert is_included("modules/ensemble.html")
    assert not is_included("whats_new/v1.5.html")
    assert not is_included("developers/contributing.html")
