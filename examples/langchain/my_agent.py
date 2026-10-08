import sys; sys.path.insert(0, "../shared")
import jobs
from evalkit import record_trace
from langchain.agents import create_agent
from langchain_core.tools import tool

agent = create_agent("openai:gpt-6-luna", tools=[tool(jobs.search_jobs)], system_prompt=jobs.SYSTEM)


def answer(user_input: str) -> dict:
    """Offline evals call this: no trace is written, so eval runs don't pollute .traces/."""
    jobs.calls.clear()
    out = agent.invoke({"messages": [("user", user_input)]})["messages"][-1].text
    return {"output": out, "steps": list(jobs.calls)}


def serve(user_input: str) -> str:
    """Production entrypoint: same agent, plus a trace tagged with the git version for online evals."""
    r = answer(user_input)
    record_trace(user_input, r["output"], steps=r["steps"])
    return r["output"]
