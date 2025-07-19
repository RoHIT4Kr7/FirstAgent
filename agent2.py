from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Union
from langchain_google_vertexai import ChatVertexAI
from langchain_perplexity import ChatPerplexity
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = ChatVertexAI(model_name="gemini-2.5-flash", project="n8n-local-463912")


def process(state: AgentState) -> AgentState:
    """This node will solve the request your input"""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []
user_input = input("ENTER: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("ENTER: ")
