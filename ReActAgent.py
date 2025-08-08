import os
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import gspread
from langchain_perplexity import ChatPerplexity

# Load environment variables from .env file
load_dotenv()

# --- Tool Definitions ---


def search_web(query: str):
    """
    A powerful tool that allows you to perform real-time internet searches for up-to-date information.
    Use this for current events, news, or any topic requiring the latest data.
    """
    pplx_api_key = os.getenv("PPLX_API_KEY")
    if not pplx_api_key:
        return "Perplexity API key (PPLX_API_KEY) is not configured."

    # Use Perplexity's online model to get answers from the web
    chat = ChatPerplexity(model="sonar", temperature=0.7, pplx_api_key=pplx_api_key)

    # Perplexity expects a list of messages, so we wrap the query
    messages = [HumanMessage(content=query)]
    result = chat.invoke(messages)

    return result.content


@tool
def contact_details(name: str):
    """
    Use this tool to look up a user's contact information (e.g., email address) from the 'Contacts' Google Sheet.
    Use this before send_mail if you need to retrieve a recipient's address based on their name.
    """
    try:
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path:
            return "Google Sheets credentials path is not configured."

        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
        client = gspread.authorize(creds)
        sheet = client.open("Contacts").sheet1
        records = sheet.get_all_records()

        for record in records:
            if record.get("Name") and record.get("Name").lower() == name.lower():
                return (
                    f"Contact found: Name - {record['Name']}, Email - {record['Email']}"
                )
        return f"Contact with name '{name}' not found."
    except FileNotFoundError:
        return f"Error: Credentials file not found at path: {creds_path}"
    except Exception as e:
        return f"An error occurred while accessing Google Sheets: {e}"


@tool
def send_mail(to: str, subject: str, message: str):
    """
    Use this tool to compose and send emails.
    Do not use this tool unless you have an explicit instruction from the user and have verified the recipient's email address.
    """
    try:
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")

        if not sender_email or not sender_password:
            return "Sender email or password is not configured."

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to, msg.as_string())

        return "Email sent successfully!"
    except Exception as e:
        return f"An error occurred while sending email: {e}"


# --- Agent State and Graph Definition ---


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# --- System Prompt ---
system_prompt_text = """You are a helpful and informative AI assistant.

You have access to the following tools:

1.  **search_web**: A powerful tool that allows you to perform real-time internet searches for up-to-date information.
    * **Use when**: The user asks about current events, news, or very recent developments; requests real-time data (e.g., weather, stock prices); asks for specific facts or statistics that might change frequently; the query implies a need for the most current information.
    * **Do NOT use when**: The question can be answered using your general knowledge; the user asks for creative writing or conversational responses.

2.  **contact_details**: Use this tool to look up a user's contact information (e.g., email address) from a Google Sheet.
    * **Use when**: You need to send a message or data to someone and need to find their email address first. Use this *before* `send_mail` if you only have a name.

3.  **send_mail**: Use this tool to compose and send emails on behalf of the user.
    * **Do NOT use unless**: You have the user's explicit instruction to send an email; you have verified the recipient's contact information (using `contact_details` if needed); the email content is finalized.

**Example of Proper Tool Usage Flow:**
If the user says: “Find out the latest news on AI and send it to Jane Doe.”

1.  Use `search_web` to get the latest news on AI.
2.  Use `contact_details` with the name "Jane Doe" to retrieve her email address.
3.  If the email is found, compose and send the email using `send_mail`.

Always choose the tool(s) that best fit the user's intent and execute them in a logical sequence.
"""

# --- LLM and Agent Executor Setup ---

# Initialize the model without the system prompt
llm = ChatVertexAI(model_name="gemini-2.5-flash")

tools = [search_web, contact_details, send_mail]
# Bind the tools to the LLM
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

# --- Graph Definition ---


# Define the nodes for the graph
def agent_node(state):
    """Invokes the LLM to generate a response or decide on a tool."""
    # Prepend the system message to the current conversation state
    messages_with_system_prompt = [SystemMessage(content=system_prompt_text)] + state[
        "messages"
    ]

    response = llm_with_tools.invoke(messages_with_system_prompt)
    return {"messages": [response]}


# Define the conditional logic for routing
def should_continue(state):
    """Determines the next step: call a tool or end the conversation."""
    if state["messages"][-1].tool_calls:
        return "continue"
    return "end"


# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
# Use the prebuilt ToolNode for the action node
workflow.add_node("action", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "action", "end": END},
)
workflow.add_edge("action", "agent")
app = workflow.compile()


# --- Main Execution Block ---
if __name__ == "__main__":
    print("Agent is ready. How can I help you? (Type 'exit' to quit)")

    # This list will store the entire conversation history
    conversation_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Add the user's message to the history
        conversation_history.append(HumanMessage(content=user_input))

        # Prepare the input for the graph with the full history
        inputs = {"messages": conversation_history}

        # Variable to hold the final AI message from the stream
        final_ai_message = None

        for output in app.stream(inputs):
            # stream() yields dictionaries with output from the node that just ran
            for key, value in output.items():
                if key == "agent" and value["messages"]:
                    # The last message from the agent node is the most recent one
                    last_message = value["messages"][-1]
                    if last_message.content:
                        print(f"Agent: {last_message.content}")
                    if last_message.tool_calls:
                        print("Agent is calling a tool...")

                    # Store the latest AI message
                    final_ai_message = last_message

        # After the stream is finished, add the final AI response to our history
        if final_ai_message:
            conversation_history.append(final_ai_message)

        print("\n---\n")
