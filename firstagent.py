from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
model = ChatVertexAI(model_name="gemini-2.5-flash", project="n8n-local-463912")
