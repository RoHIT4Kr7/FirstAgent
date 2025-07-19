from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict
from langchain_perplexity import ChatPerplexity
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = ChatPerplexity(model="sonar", temperature=0.7)
# optional parameter


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("ENTER: ")
while user_input.lower() != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("ENTER: ")
