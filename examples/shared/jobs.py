"""Shared tool: search the web for jobs with Exa. Every call is logged so evalkit can judge the steps."""
import os

from exa_py import Exa

calls: list[dict] = []   # reset by each framework's answer(); becomes the trace `steps`


def search_jobs(query: str) -> str:
    """Search the web for current job postings. Pass a descriptive query, e.g. 'applied AI engineer jobs remote Europe'."""
    res = Exa(os.environ["EXA_API_KEY"]).search_and_contents(
        f"{query} job opening", num_results=6, text={"max_characters": 500})
    out = "\n".join(f"- {r.title} | {r.url} | {(r.text or '').strip()[:300]!r}" for r in res.results)
    calls.append({"tool": "search_jobs", "query": query, "results": out})
    return out or "No results."


SYSTEM = ("You find job postings. Always call search_jobs first. Reply with a numbered list of at most 5 "
          "jobs: title, company, URL, and one line on why it matches. Use ONLY URLs returned by the tool; "
          "never invent jobs or links. If nothing relevant is found, say so.")
