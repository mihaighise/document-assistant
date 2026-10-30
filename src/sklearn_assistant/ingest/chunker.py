"""Turn the Sphinx HTML docs into section-level chunks with citable URLs.

Sphinx wraps every heading in a <section id="..."> element, so each section maps
to a chunk whose URL is `page.html#section-id`. Text belonging to nested
subsections goes into their own chunks, not the parent's.

Two kinds of pages get special handling because they don't use sections:
- API pages: each method (`dl.py.method`) becomes its own chunk.
- The glossary: each term (`dl.glossary > dt`) becomes its own chunk.
"""

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from sklearn_assistant.config import DOCS_BASE_URL

# Pages that are about the project rather than about using the library.
EXCLUDE_PREFIXES = (
    "_", "whats_new", "developers", "notebooks", "lite", "binder", "testimonials",
    "about", "governance", "install", "search", "sg_execution_times", "contents",
    "index", "related_projects", "roadmap", "presentations", "support",
    "min_dependency", "maintainers", "institutional_support", "preface",
    "documentation", "communication_team", "contributor_experience_team",
)
# Gallery cross-links, download buttons, and other navigation that isn't content.
NOISE_SELECTORS = (
    "a.headerlink", "script", "style", "img", "section#gallery-examples",
    "div.sphx-glr-download-link-note", "div.sphx-glr-footer", "div.sphx-glr-thumbnails",
    "div.sphx-glr-thumbcontainer", "p.sphx-glr-timing", "div.sphx-glr-script-out img",
)
# Elements that start a new line when rendered; everything else is treated as inline.
BLOCK_TAGS = ("p", "li", "dt", "dd", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "div",
              "blockquote", "table", "ul", "ol", "dl")

MAX_CHARS = 6000   # ~1500 tokens; longer sections get split on paragraph boundaries
MIN_CHARS = 80     # sections with less body text than this are just headings/link lists
MAX_OUTPUT_CHARS = 1000  # gallery script output can be pages of repeated warnings


@dataclass
class Chunk:
    """One retrievable unit of documentation, plus the metadata needed to cite it."""

    id: str
    text: str        # breadcrumb title + body; this is what gets embedded
    title: str       # "Ensembles > Gradient-boosted trees > Usage"
    url: str         # version-pinned URL with section anchor
    page: str        # path relative to the docs root
    doc_type: str    # user_guide | api | example | glossary

    def to_dict(self) -> dict:
        """Plain dict for writing to JSONL."""
        return asdict(self)


def is_included(rel_path: str) -> bool:
    """Whether a page (path relative to the docs root) should be indexed."""
    return not rel_path.startswith(EXCLUDE_PREFIXES)


def doc_type_for(rel_path: str) -> str:
    """Classify a page by where it lives in the docs tree.

    Stored as metadata so retrieval can later filter or boost by type
    (e.g. prefer examples for code questions).
    """
    if rel_path.startswith(("modules/generated/", "api/")):
        return "api"
    if rel_path.startswith("auto_examples/"):
        return "example"
    if rel_path == "glossary.html":
        return "glossary"
    return "user_guide"


def element_to_text(el: Tag) -> str:
    """Render HTML to plain text, keeping code blocks verbatim inside ``` fences.

    Code is swapped out for placeholders first so the whitespace normalization
    below can't mangle its indentation, then restored at the end.
    """
    # Work on a copy so callers' trees are left untouched.
    el = BeautifulSoup(str(el), "lxml")
    code_blocks: list[str] = []
    for pre in el.find_all("pre"):
        code = pre.get_text().rstrip()
        if pre.find_parent("div", class_="sphx-glr-script-out") and len(code) > MAX_OUTPUT_CHARS:
            code = code[:MAX_OUTPUT_CHARS] + "\n[... output truncated]"
        code_blocks.append(code)
        pre.replace_with(NavigableString(f"\n\x00CODE{len(code_blocks) - 1}\x00\n"))
    # API parameter lists: "X<span class=classifier>array-like" -> "X : array-like"
    for span in el.select("span.classifier"):
        span.insert_before(NavigableString(" : "))
    # Source line-wraps inside paragraphs are not real line breaks.
    for s in el.find_all(string=True):
        s.replace_with(re.sub(r"\s+", " ", s))
    # Block elements end with a newline; inline ones (<code>, <a>, ...) stay on the same line.
    for tag in el.find_all(BLOCK_TAGS):
        tag.append(NavigableString("\n"))

    text = el.get_text("")
    lines = [line.strip() for line in text.split("\n")]
    # Nested blocks produce runs of blank lines; keep at most one between paragraphs.
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return re.sub(r"\x00CODE(\d+)\x00",
                  lambda m: f"```python\n{code_blocks[int(m.group(1))]}\n```", text)


def split_long(body: str) -> list[str]:
    """Split an oversized body into pieces of at most MAX_CHARS.

    Greedily packs whole paragraphs into each piece so splits fall on natural
    boundaries wherever possible.
    """
    if len(body) <= MAX_CHARS:
        return [body]
    parts, current = [], ""
    for para in body.split("\n\n"):
        if current and len(current) + len(para) > MAX_CHARS:
            parts.append(current)
            current = ""
        current = f"{current}\n\n{para}" if current else para
    if current:
        parts.append(current)
    # A single paragraph can still exceed the limit (e.g. a huge code block).
    return [p[i:i + MAX_CHARS] for p in parts for i in range(0, len(p), MAX_CHARS)]


def heading_text(section: Tag) -> str:
    """Text of a section's own heading (not those of its subsections)."""
    h = section.find(re.compile(r"^h[1-6]$"), recursive=False)
    return h.get_text(" ", strip=True) if h else ""


def make_chunks(title: str, body: str, url: str, rel_path: str, doc_type: str) -> list[Chunk]:
    """Build the chunk(s) for one section, splitting it if it is too long.

    The title is prepended to the text so the embedding sees the section's
    context (e.g. which estimator a "Parameters" block belongs to). IDs are
    derived from the URL, so they stay stable across re-ingestion runs.
    """
    if len(body) < MIN_CHARS:
        return []
    parts = split_long(body)
    chunks = []
    for i, part in enumerate(parts):
        part_title = title if len(parts) == 1 else f"{title} (part {i + 1}/{len(parts)})"
        chunk_id = hashlib.sha1(f"{url}|{i}".encode()).hexdigest()[:16]
        chunks.append(Chunk(chunk_id, f"{part_title}\n\n{part}", part_title, url,
                            rel_path, doc_type))
    return chunks


def chunk_page(html: str, rel_path: str) -> list[Chunk]:
    """Chunk one HTML page. `rel_path` is the page's path relative to the docs root."""
    soup = BeautifulSoup(html, "lxml")
    # The main content; this skips the navbar, sidebars and footer.
    article = soup.select_one("article.bd-article")
    if article is None:
        return []
    for sel in NOISE_SELECTORS:
        for el in article.select(sel):
            el.decompose()

    page_url = DOCS_BASE_URL + rel_path
    doc_type = doc_type_for(rel_path)
    page_title = heading_text(article.find("section")) if article.find("section") else ""
    chunks: list[Chunk] = []

    # API methods: pull each out of the tree so the class chunk stays focused.
    for method in article.select("dl.py.method"):
        dt = method.find("dt")
        anchor = dt.get("id", "") if dt else ""
        name = anchor.rsplit(".", 1)[-1] if anchor else "method"
        title = f"{page_title} > {name}"
        chunks += make_chunks(title, element_to_text(method), f"{page_url}#{anchor}",
                              rel_path, doc_type)
        method.decompose()

    # Glossary: one chunk per term.
    for glossary in article.select("dl.glossary"):
        for dt in glossary.find_all("dt", recursive=False):
            dd = dt.find_next_sibling("dd")
            term = dt.get_text(" ", strip=True)
            body = element_to_text(dd) if dd else ""
            chunks += make_chunks(f"Glossary > {term}", body, f"{page_url}#{dt.get('id', '')}",
                                  rel_path, doc_type)
        glossary.decompose()

    def walk(section: Tag, breadcrumb: list[str]) -> None:
        """Emit a chunk for this section's own text, then recurse into subsections."""
        crumbs = breadcrumb + [heading_text(section)]
        # Detach subsections so this section's body contains only its own text.
        subsections = section.find_all("section", recursive=False)
        for sub in subsections:
            sub.extract()
        # The heading is already in the breadcrumb title; don't repeat it in the body.
        heading = section.find(re.compile(r"^h[1-6]$"), recursive=False)
        if heading:
            heading.decompose()
        body = element_to_text(section)
        url = f"{page_url}#{section['id']}" if section.get("id") else page_url
        chunks.extend(make_chunks(" > ".join(c for c in crumbs if c), body, url,
                                  rel_path, doc_type))
        for sub in subsections:
            walk(sub, crumbs)

    for top in article.find_all("section", recursive=False):
        walk(top, [])
    return chunks


def chunk_docs(docs_dir: Path) -> list[Chunk]:
    """Chunk every included HTML page under `docs_dir`, in a deterministic order."""
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.rglob("*.html")):
        rel_path = path.relative_to(docs_dir).as_posix()
        if is_included(rel_path):
            chunks += chunk_page(path.read_text(encoding="utf-8"), rel_path)
    return chunks
