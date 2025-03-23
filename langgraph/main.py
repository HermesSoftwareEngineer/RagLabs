from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_google_vertexai import ChatVertexAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

# Carrega as variáveis de ambiente
load_dotenv()

@tool
def human_assistance(query: str) -> str:
    """Requerer assistência humana"""
    human_response = interrupt({"query": query})
    return human_response["data"]

tool_tavily = TavilySearchResults(max_results=2)
tools = [tool_tavily, human_assistance]

class State(TypedDict):
    # As menssagens são do tipo "list"
    # A função 'add_messages' define como as mensagens serão adicionadas
    # (no caso, as mensagens serão adicionadas, e não substituídas)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = ChatVertexAI(model_name="gemini-1.5-flash")

def chatbot(state: State):
    llm_with_tools = llm.bind_tools(tools)
    return {'messages': [llm_with_tools.invoke(state['messages'])]}

tools_node = ToolNode(tools)

# Definindo os nós
# O primeiro argumento é o nome exclusivo do nó
# O segundo argumento é a função ou objeto que será usado sempre
graph_builder.add_node('tools', tools_node)
graph_builder.add_node('chatbot', chatbot)

# Instruindo as pontes entre os nós
graph_builder.add_edge(START, 'chatbot')
graph_builder.add_conditional_edges('chatbot', tools_condition, {"tools": "tools", END: END})
graph_builder.add_edge('tools', 'chatbot')

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
