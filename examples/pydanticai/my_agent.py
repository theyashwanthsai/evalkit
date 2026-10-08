import sys; sys.path.insert(0, "../shared")
import jobs
from evalkit import record_trace
from pydantic_ai import Agent

agent = Agent("openai:gpt-6-luna", instructions=jobs.SYSTEM, tools=[jobs.search_jobs])


def answer(user_input: str) -> dict:
    jobs.calls.clear()
    return {"output": agent.run_sync(user_input).output, "steps": list(jobs.calls)}


def serve(user_input: str) -> str:
    r = answer(user_input)
    record_trace(user_input, r["output"], steps=r["steps"])
    return r["output"]
