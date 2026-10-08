import sys; sys.path.insert(0, "../shared")
import jobs
from crewai import LLM, Agent, Crew, Task
from crewai.tools import tool
import evalkit
from evalkit import record_trace

search_jobs = tool("search_jobs")(jobs.search_jobs)


def answer(user_input: str) -> dict:
    jobs.calls.clear()
    agent = Agent(role="Job scout", goal=jobs.SYSTEM, backstory="Finds real job postings.",
                  tools=[search_jobs], llm=LLM(model="openai/gpt-6-luna"))
    task = Task(description=user_input, expected_output="A numbered list of jobs.", agent=agent)
    out = Crew(agents=[agent], tasks=[task], verbose=False).kickoff().raw
    return {"output": out, "steps": list(jobs.calls)}


def serve(user_input: str) -> str:
    with evalkit.run():   # links traces if this agent is called by, or calls, another agent
        r = answer(user_input)
        record_trace(user_input, r["output"], steps=r["steps"])
    return r["output"]
