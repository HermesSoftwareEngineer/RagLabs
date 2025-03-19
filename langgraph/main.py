from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from langchain_google_vertexai import ChatVertexAI

class State(TypedDict):
    # As menssagens são do tipo "list"
    # A função 'add_messages' define como as mensagens serão adicionadas
    # (no caso, as mensagens serão adicionadas, e não substituídas)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = ChatVertexAI(model_name="gemini-1.5-flash")

def chatbot(state: State):
    return {'messages': [llm.invoke(state['messages'])]}

# O primeiro argumento é o nome exclusivo do nó
# O segundo argumento é a função ou objeto que será usado sempre
graph_builder.add_node('chatbot', chatbot)
graph_builder.add_edge(START, 'chatbot')
graph_builder.add_edge('chatbot', END)

graph = graph_builder.compile()

