from langgraph.graph import StateGraph, MessagesState, START
from langchain.chat_models import init_chat_model
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langchain_core.messages import SystemMessage,  HumanMessage

from dotenv import load_dotenv
from utils.logger import get_logger

from assignment_chat.prompts import return_instructions
from assignment_chat.tools_music import get_music_info
from assignment_chat.tools_books import recommend_books

_logs = get_logger(__name__)
load_dotenv(".env")
load_dotenv(".secrets")

chat_agent = init_chat_model(
    "openai:gpt-4o-mini",
)
tools = [get_music_info, recommend_books]

instructions = return_instructions()

def call_model(state: MessagesState):
    """LLM decides whether to call a tool or not"""
    response = chat_agent.bind_tools(tools).invoke( [SystemMessage(content=instructions)] + state["messages"])
    return {
        "messages": [response]
    }

def get_chat_agent():
    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    graph = builder.compile()
    return graph

