import sys; sys.path.insert(0, "../shared")
import jobs
from evalkit import record_trace
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

graph = create_react_agent(ChatOpenAI(model="gpt-6-luna"), [tool(jobs.search_jobs)], prompt=jobs.SYSTEM)


def answer(user_input: str) -> dict:
    jobs.calls.clear()
    out = graph.invoke({"messages": [("user", user_input)]})["messages"][-1].content
    return {"output": out, "steps": list(jobs.calls)}


def serve(user_input: str) -> str:
    r = answer(user_input)
    record_trace(user_input, r["output"], steps=r["steps"])
    return r["output"]
