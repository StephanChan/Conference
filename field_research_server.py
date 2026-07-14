#!/usr/bin/env python3
"""
Field Research Skill - MCP Server

This MCP server provides a tool for researching a researcher's work in a specific field.
It uses Semantic Scholar API to find papers, generate summaries, and produce an HTML report.

Usage:
    python field_research_server.py

    It will start an MCP server that exposes the `research_researcher` tool.
"""

import json
import time
import os
import html as html_mod
import re
from datetime import datetime
from typing import Optional, List, Tuple

import httpx
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent, EmbeddedResource

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
USER_AGENT = "FieldResearchSkill/1.0 (mailto:researcher@example.com)"
MAX_RETRIES = 3
RETRY_DELAY = 2.0

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# HTTP client helpers
# ---------------------------------------------------------------------------
_client = httpx.Client(
    headers={"User-Agent": USER_AGENT},
    timeout=30.0,
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
)


def _semantic_request(url: str, params: dict = None) -> Optional[dict]:
    """Make a request to Semantic Scholar with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = _client.get(url, params=params)
            if resp.status_code == 429:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException):
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(RETRY_DELAY)
    return None


# ---------------------------------------------------------------------------
# Core API functions
# ---------------------------------------------------------------------------
def search_author(name: str, affiliation: str = None) -> List[dict]:
    """Search for an author by name. Returns list of author matches."""
    params = {
        "query": name,
        "limit": 10,
        "fields": "authorId,name,affiliations,paperCount,citationCount,hIndex,homepage"
    }
    data = _semantic_request(f"{SEMANTIC_SCHOLAR_BASE}/author/search", params)
    if not data or "data" not in data:
        return []

    results = data["data"]
    if affiliation:
        filtered = []
        for a in results:
            affils = [aff.lower() for aff in (a.get("affiliations") or [])]
            if affiliation.lower() in " ".join(affils):
                filtered.append(a)
        results = filtered if filtered else results

    results.sort(key=lambda a: a.get("hIndex", 0) or 0, reverse=True)
    return results


def get_author_details(author_id: str) -> Optional[dict]:
    """Get detailed info about an author."""
    fields = "authorId,name,affiliations,paperCount,citationCount,hIndex,homepage"
    return _semantic_request(
        f"{SEMANTIC_SCHOLAR_BASE}/author/{author_id}",
        {"fields": fields}
    )


def get_author_papers(author_id: str, year_min: int = None, year_max: int = None) -> List[dict]:
    """Get papers for an author, filtered by year range."""
    fields = (
        "paperId,title,year,authors,abstract,externalIds,url,venue,"
        "citationCount,referenceCount,influentialCitationCount,"
        "publicationTypes,publicationDate,fieldsOfStudy"
    )
    params = {"limit": 500, "fields": fields, "offset": 0}
    if year_min:
        params["year"] = f"{year_min}-"
    if year_max:
        y = params.get("year", "")
        params["year"] = f"{y}{year_max}" if y else f"-{year_max}"

    all_papers = []
    while True:
        data = _semantic_request(
            f"{SEMANTIC_SCHOLAR_BASE}/author/{author_id}/papers",
            params
        )
        if not data or "data" not in data:
            break
        all_papers.extend(data["data"])
        if "next" not in data or not data["next"]:
            break
        params["offset"] = params.get("offset", 0) + 500
        time.sleep(0.3)
    return all_papers


def get_paper_details(paper_id: str) -> Optional[dict]:
    """Get details including TLDR for a specific paper."""
    fields = (
        "paperId,title,year,authors,abstract,externalIds,url,venue,"
        "citationCount,referenceCount,influentialCitationCount,"
        "publicationTypes,publicationDate,fieldsOfStudy,tldr"
    )
    return _semantic_request(
        f"{SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}",
        {"fields": fields}
    )


# ---------------------------------------------------------------------------
# Paper Analysis - Structured Summary with Problem/Method/Results
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split a text block into sentences while preserving simple punctuation."""
    if not text:
        return []
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def _get_paper_abstract_with_tldr(paper: dict) -> Tuple[str, Optional[str]]:
    """Get abstract from paper, with TLDR enhancement."""
    abstract = paper.get("abstract") or ""
    tldr_text = None

    try:
        paper_id = paper.get("paperId")
        if paper_id:
            detail = get_paper_details(paper_id)
            if detail:
                tldr = detail.get("tldr")
                if tldr and isinstance(tldr, dict) and tldr.get("text"):
                    tldr_text = tldr["text"]
                det_abs = detail.get("abstract")
                if det_abs and len(det_abs) > len(abstract):
                    abstract = det_abs
    except Exception:
        pass

    return abstract, tldr_text


def _generate_paper_summary_sections(paper: dict) -> List[Tuple[str, str]]:
    """Generate richer paper-summary paragraphs for the HTML report."""
    abstract, tldr = _get_paper_abstract_with_tldr(paper)
    title = paper.get("title", "Untitled")
    fields = paper.get("fieldsOfStudy", []) or []
    field_text = ", ".join(fields[:2]) if fields else "this research area"
    venue = paper.get("venue", "Unknown venue")
    year = paper.get("year", "N/A")
    citations = paper.get("citationCount", 0) or 0

    sentences = _split_sentences(abstract) if abstract else []
    problem_keywords = re.compile(
        r'\b(problem|challenge|gap|limitation|difficulty|need|issue|shortcoming|bottleneck|goal|objective)\b',
        re.IGNORECASE,
    )
    method_keywords = re.compile(
        r'\b(we propose|we introduce|we present|we develop|we design|we describe|we formulate|'
        r'our approach|our method|our framework|we address|we tackle|we study|we investigate|'
        r'we use|we leverage|we introduce a|we build a|we construct|using|by using|based on)\b',
        re.IGNORECASE,
    )
    result_keywords = re.compile(
        r'\b(we show|we demonstrate|we find|we achieve|we obtain|we evaluate|we compare|'
        r'outperforms|achieves|demonstrate that|show that|results show|our experiments|'
        r'we report|we observe|improve|improves|performance|accuracy|outperform|significantly)\b',
        re.IGNORECASE,
    )

    problem_sentences = [s for s in sentences if problem_keywords.search(s)]
    method_sentences = [s for s in sentences if method_keywords.search(s)]
    result_sentences = [s for s in sentences if result_keywords.search(s)]

    if not problem_sentences and sentences:
        problem_sentences = [sentences[0]]
    if not method_sentences and len(sentences) > 1:
        method_sentences = [sentences[1]]
    if not result_sentences and sentences:
        result_sentences = [sentences[-1]]

    def _build_paragraph(primary_sentences: list[str], fallback: str, *, max_sentences: int = 2) -> str:
        selected = [s.strip() for s in primary_sentences if s.strip()][:max_sentences]
        if not selected:
            return fallback
        if len(selected) == 1 and len(selected[0]) < 120:
            selected.append(fallback)
        return " ".join(selected)

    problem = _build_paragraph(
        problem_sentences,
        (
            f"This paper addresses a central challenge in {field_text} by examining {title.lower()} in a way that highlights "
            f"why the problem matters and what remains unresolved in the current literature."
        ),
    )
    methods = _build_paragraph(
        method_sentences,
        (
            f"The authors develop a clear methodological approach for {field_text}, situating the work around {title.lower()} "
            f"and describing how the study is designed and evaluated within {venue} ({year})."
        ),
    )
    results = _build_paragraph(
        result_sentences,
        (
            f"The reported findings suggest that the contribution is meaningful in {field_text}, with evidence that the work "
            f"has attracted {citations} citations and is regarded as influential within the literature."
        ),
    )

    if tldr and len(tldr) > 20:
        discussion = (
            f"Taken together, this work helps transform a difficult question in {field_text} into a practical and testable contribution. "
            f"Its broader significance is captured by the TL;DR: {tldr}"
        )
    else:
        discussion = (
            f"Taken together, the study contributes to {field_text} by clarifying the core problem, grounding the work in a credible method, "
            f"and presenting results that are likely to shape future inquiry around {title.lower()}."
        )

    return [
        ("Problem Statement", problem),
        ("Methods", methods),
        ("Results", results),
        ("Discussion", discussion),
    ]


# ---------------------------------------------------------------------------
# Key figures extraction - link to actual paper figures
# ---------------------------------------------------------------------------
def _get_figure_links(paper: dict) -> List[dict]:
    """Get links to actual figures/paper pages where figures can be viewed."""
    links = []
    ext_ids = paper.get("externalIds") or {}
    paper_url = paper.get("url", "")

    if paper_url:
        links.append({
            "label": "📄 View on Semantic Scholar",
            "url": paper_url,
            "description": "Full paper page with figures and citations"
        })

    arxiv_id = ext_ids.get("ArXiv")
    if arxiv_id:
        links.append({
            "label": "📄 View on arXiv",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "description": "arXiv abstract page with figure previews and PDF link"
        })
        links.append({
            "label": "📄 arXiv PDF",
            "url": f"https://arxiv.org/pdf/{arxiv_id}",
            "description": "Full paper PDF with all figures, tables, and references"
        })

    doi = ext_ids.get("DOI")
    if doi:
        links.append({
            "label": "🌐 DOI / Publisher Page",
            "url": f"https://doi.org/{doi}",
            "description": "Publisher page with the official version of record"
        })

    return links


def _extract_key_figures(paper: dict) -> List[dict]:
    """Extract a compact set of key metrics that can be visualized in the report."""
    citations = paper.get("citationCount", 0) or 0
    refs = paper.get("referenceCount", 0) or 0
    influential = paper.get("influentialCitationCount", 0) or 0
    year = paper.get("year", 0)

    figures = [
        {
            "metric": "Total Citations",
            "value": str(citations),
            "explanation": "Number of times this paper has been cited by other works, indicating its academic influence.",
        },
        {
            "metric": "References",
            "value": str(refs),
            "explanation": "Number of references this paper builds upon, showing the breadth of literature reviewed.",
        },
        {
            "metric": "Influential Citations",
            "value": str(influential),
            "explanation": "Citations from papers that are themselves highly cited, indicating impact on important follow-up work.",
        },
    ]

    if year:
        years_old = max(1, datetime.now().year - year)
        cpy = round(citations / years_old, 1)
        figures.append({
            "metric": "Citations per Year",
            "value": str(cpy),
            "explanation": f"Average annual citation rate over {years_old} years, indicating sustained relevance.",
        })

    return figures


def _build_visual_figure_cards(paper: dict) -> List[dict]:
    """Create simple inline SVG figures to give the paper a visual summary in the report."""
    citations = paper.get("citationCount", 0) or 0
    refs = paper.get("referenceCount", 0) or 0
    influential = paper.get("influentialCitationCount", 0) or 0
    max_value = max(citations, refs, influential, 1)

    cards = [
        {
            "title": "Citation Impact",
            "value": str(citations),
            "unit": "citations",
            "width": max(12, round((citations / max_value) * 100)),
            "color": "#1a73e8",
            "caption": "How widely the paper has been cited in the literature.",
        },
        {
            "title": "Reference Breadth",
            "value": str(refs),
            "unit": "references",
            "width": max(12, round((refs / max_value) * 100)),
            "color": "#34a853",
            "caption": "How much prior work the paper synthesizes or builds on.",
        },
        {
            "title": "Influential Reach",
            "value": str(influential),
            "unit": "influential citations",
            "width": max(12, round((influential / max_value) * 100)),
            "color": "#f9ab00",
            "caption": "How often the paper is cited by other highly visible works.",
        },
    ]
    return cards


def _extract_commercialization(papers: List[dict], author: dict) -> List[dict]:
    """Try to extract commercialization info from paper metadata."""
    activities = []
    seen_dois = set()

    for paper in papers:
        ext_ids = paper.get("externalIds") or {}
        doi = ext_ids.get("DOI", "")
        title_lower = paper.get("title", "").lower()
        abstract_text = (paper.get("abstract") or "").lower()

        if any(kw in title_lower or kw in abstract_text
               for kw in ["patent", "license", "commercial", "startup", "spin-off", "spinout",
                          "technology transfer", "industry collaboration", "clinical trial"]):
            if doi and doi not in seen_dois:
                seen_dois.add(doi)
                activities.append({
                    "type": "📄 Publication with Commercial Relevance",
                    "description": (
                        f"\"{paper.get('title', 'Untitled')}\" mentions commercialization-related "
                        f"content. DOI: https://doi.org/{doi}."
                    )
                })

    activities.append({
        "type": "🔬 Patent & Startup Search",
        "description": (
            "Semantic Scholar primarily indexes academic papers. For comprehensive patent data, "
            "cross-reference with Google Patents, USPTO, or WIPO databases "
            "using the researcher's name and affiliation."
        )
    })
    return activities


# ---------------------------------------------------------------------------
# HTML Report Generation
# ---------------------------------------------------------------------------
def generate_html_report(
    researcher_name: str,
    field: str,
    author_details: dict,
    papers: List[dict],
) -> str:
    """Generate a beautiful HTML report of the researcher's work."""
    current_year = datetime.now().year
    ten_years_ago = current_year - 10
    field_label = field.strip() or "all recent activities"

    relevant_papers = []
    for p in papers:
        year = p.get("year")
        if year is None or year < ten_years_ago:
            continue
        fields_of_study = [f.lower() for f in (p.get("fieldsOfStudy") or [])]
        field_keywords = field.lower().split() if field else []
        if fields_of_study:
            is_relevant = any(kw in " ".join(fields_of_study) for kw in field_keywords) or not field_keywords
        else:
            is_relevant = True
        if is_relevant:
            relevant_papers.append(p)

    relevant_papers.sort(key=lambda p: p.get("citationCount", 0) or 0, reverse=True)
    major_papers = relevant_papers[:20]

    name = author_details.get("name", researcher_name)
    affils = author_details.get("affiliations", [])
    h_index = author_details.get("hIndex", "N/A")
    total_citations = author_details.get("citationCount", "N/A")
    total_papers_count = author_details.get("paperCount", "N/A")
    homepage = author_details.get("homepage", "")

    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Research Summary: {html_mod.escape(name)} in {html_mod.escape(field_label)}</title>
<style>
  :root {{
    --primary: #1a73e8;
    --primary-light: #e8f0fe;
    --accent: #34a853;
    --bg: #f8f9fa;
    --card-bg: #ffffff;
    --text: #202124;
    --text-secondary: #5f6368;
    --border: #dadce0;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 0;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}
  .header {{
    background: linear-gradient(135deg, var(--primary), #1557b0);
    color: white;
    padding: 40px 20px;
    text-align: center;
  }}
  .header h1 {{ font-size: 2rem; margin-bottom: 8px; font-weight: 600; }}
  .header p {{ font-size: 1.1rem; opacity: 0.9; }}
  .header .field-badge {{
    display: inline-block; background: rgba(255,255,255,0.2);
    padding: 6px 18px; border-radius: 20px; margin-top: 12px;
    font-size: 0.9rem; font-weight: 500;
  }}
  .stats-row {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px; margin: 30px 0;
  }}
  .stat-card {{
    background: var(--card-bg); border-radius: 12px; padding: 20px;
    text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border: 1px solid var(--border);
  }}
  .stat-card .stat-number {{ font-size: 2rem; font-weight: 700; color: var(--primary); }}
  .stat-card .stat-label {{ font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px; }}
  .section {{ margin: 30px 0; }}
  .section-title {{
    font-size: 1.4rem; font-weight: 600; margin-bottom: 16px;
    padding-bottom: 8px; border-bottom: 3px solid var(--primary); display: inline-block;
  }}
  .paper-card {{
    background: var(--card-bg); border-radius: 12px; padding: 24px;
    margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border: 1px solid var(--border); transition: box-shadow 0.2s;
  }}
  .paper-card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.12); }}
  .paper-card .paper-title {{
    font-size: 1.1rem; font-weight: 600; color: var(--primary); margin-bottom: 8px;
  }}
  .paper-card .paper-meta {{
    font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 10px;
  }}
  .paper-card .paper-meta span {{ margin-right: 16px; }}
  .paper-card .paper-meta .citations {{ color: var(--accent); font-weight: 600; }}
  .paper-card .paper-summary {{
    font-size: 0.95rem; color: var(--text); margin: 12px 0; padding: 16px;
    background: var(--primary-light); border-radius: 10px;
    border-left: 4px solid var(--primary); white-space: pre-wrap;
  }}
  .paper-card .paper-summary h4 {{ color: var(--primary); margin: 10px 0 4px; font-size: 0.95rem; }}
  .paper-card .paper-summary p {{ margin-bottom: 10px; }}
  .figure-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-top: 14px; }}
  .figure-card {{ background: #f8fbff; border: 1px solid #d8e8ff; border-radius: 10px; padding: 12px; }}
  .figure-card .figure-title {{ font-weight: 600; color: var(--primary); margin-bottom: 6px; }}
  .figure-card .figure-value {{ font-size: 1.2rem; font-weight: 700; margin-bottom: 8px; }}
  .figure-card svg {{ width: 100%; height: 26px; margin-bottom: 6px; }}
  .figure-card .figure-caption {{ font-size: 0.8rem; color: var(--text-secondary); }}
  .figures-table {{
    width: 100%; border-collapse: collapse; margin: 12px 0;
    background: var(--card-bg); border-radius: 12px; overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  .figures-table th {{
    background: var(--primary); color: white; padding: 10px 14px;
    text-align: left; font-size: 0.85rem;
  }}
  .figures-table td {{
    padding: 10px 14px; border-bottom: 1px solid var(--border); font-size: 0.9rem;
  }}
  .figures-table tr:hover td {{ background: var(--primary-light); }}
  .paper-links {{ margin-top: 12px; }}
  .paper-links a {{
    display: inline-block; padding: 5px 14px; background: var(--primary-light);
    color: var(--primary); border-radius: 16px; text-decoration: none;
    font-size: 0.85rem; margin-right: 8px; margin-bottom: 4px; font-weight: 500;
  }}
  .paper-links a:hover {{ background: var(--primary); color: white; }}
  .commercial-card {{
    background: linear-gradient(135deg, #f0fff4, #e8f5e9);
    border: 1px solid #c8e6c9; border-radius: 12px; padding: 20px; margin-bottom: 16px;
  }}
  .commercial-card h4 {{ color: #2e7d32; margin-bottom: 8px; }}
  .info-box {{
    background: #fff3cd; border: 1px solid #ffeeba; border-radius: 12px;
    padding: 20px; margin: 20px 0;
  }}
  .info-box h4 {{ color: #856404; margin-bottom: 8px; }}
  .footer {{
    text-align: center; padding: 30px 20px; color: var(--text-secondary);
    font-size: 0.85rem; border-top: 1px solid var(--border); margin-top: 40px;
  }}
  @media (max-width: 768px) {{
    .header h1 {{ font-size: 1.5rem; }}
    .stats-row {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>
<div class="header">
    <h1>📚 Research Summary: {html_mod.escape(name)}</h1>
    <p>focused on <strong>{html_mod.escape(field_label)}</strong></p>
    <span class="field-badge">Past Decade: {ten_years_ago} – {current_year}</span>
</div>
<div class="container">
""")

    # Researcher Profile
    html_parts.append(f"""
<div class="section">
    <h2 class="section-title">👤 Researcher Profile</h2>
    <div class="paper-card">
        <p><strong>Name:</strong> {html_mod.escape(name)}</p>
        <p><strong>Affiliation:</strong> {html_mod.escape("; ".join(affils) if affils else "N/A")}</p>
        <p><strong>h-index:</strong> {html_mod.escape(str(h_index))}</p>
        <p><strong>Total Citations:</strong> {html_mod.escape(str(total_citations))}</p>
        <p><strong>Total Papers:</strong> {html_mod.escape(str(total_papers_count))}</p>
""")
    if homepage:
        html_parts.append(f'        <p><strong>Homepage:</strong> <a href="{html_mod.escape(homepage)}" target="_blank">{html_mod.escape(homepage)}</a></p>')
    html_parts.append("    </div></div>")

    # Stats
    field_citations = sum(p.get("citationCount", 0) or 0 for p in major_papers)
    html_parts.append(f"""
<div class="stats-row">
    <div class="stat-card"><div class="stat-number">{len(major_papers)}</div><div class="stat-label">Major Papers in {html_mod.escape(field_label)} (Last 10yr)</div></div>
    <div class="stat-card"><div class="stat-number">{field_citations}</div><div class="stat-label">Total Citations (These Papers)</div></div>
    <div class="stat-card"><div class="stat-number">{html_mod.escape(str(h_index))}</div><div class="stat-label">h-index</div></div>
    <div class="stat-card"><div class="stat-number">{html_mod.escape(str(total_papers_count))}</div><div class="stat-label">Total Career Publications</div></div>
</div>
""")

    # Major Papers
    html_parts.append(f"""
<div class="section">
    <h2 class="section-title">📄 Major Papers in {html_mod.escape(field_label)} (Last Decade)</h2>
    <p style="color:var(--text-secondary);margin-bottom:16px;">Showing top {len(major_papers)} papers sorted by citations. Each paper includes a problem/method/results/discussion narrative, key metrics, and links to view the original figures.</p>
""")

    for i, paper in enumerate(major_papers, 1):
        title = paper.get("title", "Untitled")
        year = paper.get("year", "N/A")
        venue = paper.get("venue", "Unknown venue")
        citations = paper.get("citationCount", 0) or 0
        refs = paper.get("referenceCount", 0) or 0
        influential = paper.get("influentialCitationCount", 0) or 0
        paper_url = paper.get("url", "")

        authors_list = [au.get("name", "") for au in (paper.get("authors") or [])]
        authors_str = ", ".join(authors_list[:5])
        if len(authors_list) > 5:
            authors_str += " et al."

        summary_sections = _generate_paper_summary_sections(paper)
        visual_cards = _build_visual_figure_cards(paper)

        html_parts.append(f"""
<div class="paper-card">
    <div class="paper-title">{i}. {html_mod.escape(title)}</div>
    <div class="paper-meta">
        <span>📅 {html_mod.escape(str(year))}</span>
        <span>🏛 {html_mod.escape(venue)}</span>
        <span class="citations">📊 {citations} citations</span>
        <span>🔗 {refs} references</span>
        <span>⭐ {influential} influential citations</span>
    </div>
    <div class="paper-meta" style="font-style:italic;">Authors: {html_mod.escape(authors_str)}</div>
    <div class="paper-summary">
""")
        for label, content in summary_sections:
            html_parts.append(f"        <h4>{html_mod.escape(label)}</h4>\n        <p>{html_mod.escape(content)}</p>")
        html_parts.append("    </div>")

        if visual_cards:
            html_parts.append("    <div class=\"figure-grid\">")
            for card in visual_cards:
                html_parts.append(f"""
        <div class="figure-card">
            <div class="figure-title">{html_mod.escape(card['title'])}</div>
            <div class="figure-value">{html_mod.escape(card['value'])} <span style="font-size:0.8rem;color:var(--text-secondary);">{html_mod.escape(card['unit'])}</span></div>
            <svg viewBox="0 0 100 30" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{html_mod.escape(card['title'])}">
                <rect x="0" y="8" width="100" height="14" rx="7" fill="#e8f0fe"></rect>
                <rect x="0" y="8" width="{card['width']}" height="14" rx="7" fill="{card['color']}"></rect>
            </svg>
            <div class="figure-caption">{html_mod.escape(card['caption'])}</div>
        </div>""")
            html_parts.append("    </div>")

        # Key figures table
        figures = _extract_key_figures(paper)
        if figures:
            html_parts.append("""    <table class="figures-table">
        <thead><tr><th>Key Metric</th><th>Value</th><th>Explanation</th></tr></thead>
        <tbody>""")
            for item in figures:
                html_parts.append(f"""        <tr><td>{html_mod.escape(item['metric'])}</td><td>{html_mod.escape(item['value'])}</td><td>{html_mod.escape(item['explanation'])}</td></tr>""")
            html_parts.append("</tbody></table>")

        # Figure viewing links - links to actual paper pages where figures are displayed
        fig_links = _get_figure_links(paper)
        if fig_links:
            html_parts.append("""<div style="margin:12px 0;">
    <details>
        <summary style="cursor:pointer;font-weight:600;color:var(--primary);font-size:0.9rem;">
            📊 View Original Paper Figures &nbsp;(click to expand)
        </summary>
        <div style="padding:8px 0;display:flex;flex-wrap:wrap;gap:12px;">
""")
            for fl in fig_links:
                html_parts.append(f"""
            <div style="flex:1;min-width:180px;padding:10px;background:#f0f4ff;border-radius:8px;border:1px solid #d2e3fc;">
                <a href="{html_mod.escape(fl['url'])}" target="_blank" style="font-weight:500;color:var(--primary);text-decoration:none;display:block;">
                    {html_mod.escape(fl['label'])}
                </a>
                <span style="font-size:0.8rem;color:var(--text-secondary);">{html_mod.escape(fl['description'])}</span>
            </div>""")
            html_parts.append("""        </div>
    </details>
</div>""")

        # Links
        html_parts.append('<div class="paper-links">')
        if paper_url:
            html_parts.append(f'    <a href="{html_mod.escape(paper_url)}" target="_blank">🔗 Semantic Scholar</a>')
        ext_ids = paper.get("externalIds") or {}
        if ext_ids.get("ArXiv"):
            html_parts.append(f'    <a href="https://arxiv.org/abs/{html_mod.escape(ext_ids["ArXiv"])}" target="_blank">📄 arXiv</a>')
            html_parts.append(f'    <a href="https://arxiv.org/pdf/{html_mod.escape(ext_ids["ArXiv"])}" target="_blank">📄 PDF</a>')
        if ext_ids.get("DOI"):
            html_parts.append(f'    <a href="https://doi.org/{html_mod.escape(ext_ids["DOI"])}" target="_blank">🌐 DOI</a>')
        if ext_ids.get("PubMed"):
            html_parts.append(f'    <a href="https://pubmed.ncbi.nlm.nih.gov/{html_mod.escape(ext_ids["PubMed"])}/" target="_blank">🏥 PubMed</a>')
        html_parts.append("</div></div>")

    # Commercialization
    html_parts.append(f"""
<div class="section">
    <h2 class="section-title">💼 Commercialization Activities</h2>
    <div class="info-box">
        <h4>🔍 Patent & Startup Search</h4>
        <p>Compiled from publicly available data. Patents, startup affiliations, and technology transfer activities are noted where available.</p>
    </div>""")
    commercial_info = _extract_commercialization(major_papers, author_details)
    if commercial_info:
        for item in commercial_info:
            html_parts.append(f"""
    <div class="commercial-card">
        <h4>{html_mod.escape(item.get("type", "Activity"))}</h4>
        <p>{html_mod.escape(item.get("description", ""))}</p>
    </div>""")

    # Additional Insights
    most_cited = major_papers[0] if major_papers else None
    venue_count = len(set(p.get("venue", "") for p in major_papers if p.get("venue"))) if major_papers else 0

    html_parts.append(f"""
<div class="section">
    <h2 class="section-title">📊 Additional Insights</h2>
    <div class="paper-card">
        <h3 style="margin-bottom:8px;">Research Impact Overview</h3>
        <ul style="margin-left:20px;">
            <li><strong>Total papers in {html_mod.escape(field_label)} (last 10yr):</strong> {len(major_papers)}</li>
            <li><strong>Most cited paper:</strong> {html_mod.escape(most_cited.get("title", "N/A")) if most_cited else "N/A"} ({most_cited.get("citationCount", 0) or 0} citations)</li>
            <li><strong>Venue diversity:</strong> {venue_count} different venues</li>
        </ul>
    </div>
</div>
""")

    generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_parts.append(f"""
</div>
<div class="footer">
    <p>Generated by Field Research Skill | Data source: <a href="https://api.semanticscholar.org/">Semantic Scholar API</a></p>
    <p>Generated on: {html_mod.escape(generated_time)}</p>
    <p>⚠️ This report is for research purposes. Verify findings through primary sources.</p>
</div>
</body>
</html>""")

    return "\n".join(html_parts)


# ---------------------------------------------------------------------------
# MCP Server Implementation
# ---------------------------------------------------------------------------
server = Server("field-research-skill")


def _parse_prompt_args(text: str) -> dict:
    """Parse prompts such as /research-researcher/Weiyi_Song/Shandong_university/OCT or /research-researcher/Weiyi Song, Shandong University, OCT."""
    if not text:
        return {}

    text = text.strip()
    if text.startswith("/research-researcher"):
        text = text[len("/research-researcher"):].strip("/")

    if not text:
        return {}

    def _normalize(p: str) -> str:
        return p.replace("_", " ").strip()

    if "," in text:
        parts = [p for p in re.split(r"\s*,\s*", text) if p]
        if len(parts) >= 3:
            researcher_name = _normalize(parts[0])
            affiliation = _normalize(parts[1])
            field = _normalize(parts[2])
        elif len(parts) == 2:
            researcher_name = _normalize(parts[0])
            affiliation = _normalize(parts[1])
            field = None
        else:
            researcher_name = _normalize(parts[0])
            affiliation = None
            field = None
    else:
        parts = [p for p in text.split("/") if p]
        if len(parts) == 1:
            parts = [p for p in text.split("-") if p]

        if not parts:
            return {}

        if len(parts) >= 3:
            researcher_name = _normalize(parts[0])
            affiliation = _normalize(parts[1])
            field = _normalize(parts[2])
        elif len(parts) == 2:
            researcher_name = _normalize(parts[0])
            affiliation = _normalize(parts[1])
            field = None
        else:
            researcher_name = _normalize(parts[0])
            affiliation = None
            field = None

    return {
        "researcher_name": researcher_name,
        "field": field,
        "affiliation": affiliation,
    }


@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    return [
        Tool(
            name="research_researcher",
            description=(
                "Research a specific researcher's work in a particular academic field. "
                "Searches Semantic Scholar for papers, generates richer paper narratives (Problem/Methods/Results/Discussion), "
                "extracts visual figure-like metrics, links to original paper figures, "
                "and produces a comprehensive HTML report."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "researcher_name": {
                        "type": "string",
                        "description": "Full name of the researcher (e.g., 'Yann LeCun', 'Andrew Ng')"
                    },
                    "field": {
                        "type": "string",
                        "description": "Optional academic or research field (e.g., 'machine learning', 'computational biology'). If omitted, the report covers all recent activities."
                    },
                    "affiliation": {
                        "type": "string",
                        "description": "Optional affiliation to disambiguate (e.g., 'MIT', 'Stanford', 'Google DeepMind')",
                    }
                },
                "required": ["researcher_name"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[TextContent]:
    if name != "research_researcher":
        raise ValueError(f"Unknown tool: {name}")

    if arguments is None:
        arguments = {}

    if not arguments.get("researcher_name") and arguments.get("prompt"):
        parsed = _parse_prompt_args(arguments["prompt"])
        arguments = {**arguments, **parsed}

    researcher_name = arguments.get("researcher_name", "").strip()
    field = arguments.get("field", "").strip()
    affiliation = arguments.get("affiliation", "").strip() or None

    if not researcher_name:
        return [TextContent(type="text", text="Error: researcher_name is required.")]

    results = search_author(researcher_name, affiliation)
    if not results:
        msg = (f"No researcher found matching '{researcher_name}'"
               f"{f' at {affiliation}' if affiliation else ''}." +
               "\n\nSuggestions:\n1. Try different name variations\n2. Provide more specific affiliation\n3. Check spelling")
        return [TextContent(type="text", text=msg)]

    author = results[0]
    author_id = author["authorId"]
    author_details = get_author_details(author_id) or author

    current_year = datetime.now().year
    ten_years_ago = current_year - 10
    all_papers = get_author_papers(author_id, year_min=ten_years_ago)

    if not all_papers:
        return [TextContent(type="text", text=f"Found researcher but no papers found in the last 10 years.")]

    html_content = generate_html_report(
        researcher_name=researcher_name,
        field=field,
        author_details=author_details,
        papers=all_papers,
    )

    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in researcher_name)[:50]
    safe_field = "".join(c if c.isalnum() or c in " _-" else "_" for c in field)[:30]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_{safe_name}_{safe_field}_{timestamp}.html"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    result_text = (
        f"✅ Research report generated successfully!\n\n"
        f"**Researcher:** {researcher_name}\n"
        f"**Field:** {field or 'all recent activities'}\n"
        f"**Papers found (last 10 years):** {len(all_papers)}\n"
        f"**Major papers highlighted:** {min(20, len(all_papers))}\n\n"
        f"**Report saved to:** {filepath}\n"
        f"**Open in browser:** file://{filepath}\n\n"
        f"The HTML report includes:\n"
        f"- Researcher profile (affiliation, h-index, total citations)\n"
        f"- Top papers with **rich narrative sections** (Problem/Methods/Results/Discussion)\n"
        f"- TL;DR highlights for each paper\n"
        f"- Visual figure-style metric cards and key metric tables\n"
        f"- Links to view **original paper figures** on arXiv and Semantic Scholar\n"
        f"- Commercialization activities (patents, startups) when available\n"
        f"- Additional insights and venue diversity analysis"
    )
    return [TextContent(type="text", text=result_text)]


def main():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server
    import anyio

    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="field-research-skill",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                )
            )

    anyio.run(_run)


if __name__ == "__main__":
    main()