# main.py
# This script creates a Telegram bot that uses a LangGraph agent to respond to messages.
# It uses FastAPI to create a web server and listens for incoming messages from Telegram via a webhook.

# --- 1. Import Necessary Libraries ---
import os
import requests
from dotenv import load_dotenv
from typing import Annotated, List
from typing_extensions import TypedDict
import operator

# FastAPI for creating the web server
from fastapi import FastAPI, Request

# LangChain and LangGraph for the AI agent
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# --- 2. Load Environment Variables ---
# Load secrets from a .env file in the same directory
# Your .env file should look like this:
# OPENAI_API_KEY="sk-..."
# TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


# --- 3. Define the LangGraph Agent State ---
# The state is a dictionary that holds the conversation history.
# The graph will pass this state between nodes.
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]


# --- 4. Define the LangGraph Nodes ---
# Nodes are functions that perform actions and modify the state.

# Initialize the Language Model (LLM)
# We'll use GPT-4o, but you can use other models.
model = ChatVertexAI(model_name="gemini-2.5-flash")


def call_model(state: AgentState):
    """
    This is the primary node of our agent. It takes the current conversation
    history from the state, invokes the LLM, and returns the LLM's response.
    """
    # Get the list of messages from the current state
    messages = state["messages"]

    # Invoke the model with the messages
    response = model.invoke(messages)

    # Return a dictionary with the new message to be added to the state
    # The `operator.add` in AgentState will append this to the existing list.
    return {"messages": [response]}


# --- 5. Build and Compile the Graph ---
# This is where we define the structure and flow of our agent.

# Create a new state graph
workflow = StateGraph(AgentState)

# Add the 'call_model' function as a node named "agent"
workflow.add_node("agent", call_model)

# Set the entry point of the graph to the "agent" node
workflow.set_entry_point("agent")

# Set the finish point of the graph. Since this is a simple conversational
# agent, the "agent" node is also the end.
workflow.set_finish_point("agent")

# Compile the graph into a runnable application. This is the object
# we will interact with from our web server.
langgraph_app = workflow.compile()

# --- 6. Set up the FastAPI Web Server ---
# This server will expose a URL (webhook) for Telegram to send messages to.

# Initialize the FastAPI application
app = FastAPI()


def send_message_to_telegram(chat_id: int, text: str):
    """
    Sends a message back to the user on Telegram.
    Splits the message into chunks if it's too long.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    max_length = 4000  # keep a buffer for formatting
    messages = [text[i : i + max_length] for i in range(0, len(text), max_length)]

    for part in messages:
        payload = {"chat_id": chat_id, "text": part, "parse_mode": "Markdown"}

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            print(f"Successfully sent part of response to chat_id {chat_id}")
        except requests.exceptions.RequestException as e:
            print(f"Error sending message to Telegram: {e}")


@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    """
    This is the main webhook endpoint. Telegram will send all updates here.
    """
    data = await request.json()

    if "message" not in data or "text" not in data["message"]:
        print("Received a non-message update, ignoring.")
        return {"status": "ok", "message": "ignored non-message update"}

    chat_id = data["message"]["chat"]["id"]
    user_message = data["message"]["text"]

    print(f"Received message from chat_id {chat_id}: '{user_message}'")

    inputs = {"messages": [HumanMessage(content=user_message)]}

    response = langgraph_app.invoke(inputs)

    ai_response = response["messages"][-1].content

    print(f"Agent generated response: '{ai_response}'")

    # Send the AI's response back to the user in Telegram
    send_message_to_telegram(chat_id, ai_response)

    # Return a 200 OK response to Telegram to acknowledge receipt of the update
    return {"status": "ok"}


@app.get("/")
def read_root():
    """
    A simple root endpoint to check if the server is running.
    You can access this by going to your server's URL in a browser.
    """
    return {"status": "online", "message": "LangGraph Telegram Bot is running"}


# --- How to Run This Server ---
# 1. Make sure you have a .env file with your API keys.
# 2. Install the required libraries:
#    pip install "langchain[llms]" langgraph langchain-openai fastapi uvicorn python-dotenv requests
# 3. Use ngrok to expose your local server to the internet:
#    ngrok http 8000
# 4. Set your Telegram webhook using the ngrok URL (replace <...> with your details):
#    curl -F "url=https://<your-ngrok-url>.ngrok-free.app/telegram-webhook" https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook
# 5. Run the FastAPI server with uvicorn:
#    uvicorn main:app --host 0.0.0.0 --port 8000
