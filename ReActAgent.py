from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langchain_google_vertexai import ChatVertexAI
from langchain_perplexity import ChatPerplexity
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int):
    """This is an addition function that adds two number together"""

    return a + b


@tool
def multiply(a: int, b: int):
    """This is an multiplication function that multiply two number together"""

    return a * b


tools = [add, multiply]

search_model = ChatPerplexity(model="sonar", temperature=0.7)
llm = ChatVertexAI(
    model_name="gemini-2.5-flash", project="n8n-local-463912"
).bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query to the best of your ability"
    )
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("agent", model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "agent")
app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [("user", "Add 40+12 and then Multiply the result with 12")]}
print_stream(app.stream(inputs, stream_mode="values"))
