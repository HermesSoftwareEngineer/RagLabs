from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state import AtendimentoState
from nodes import responder

graph_builder = StateGraph(AtendimentoState)

graph_builder.add_node("responder", responder)

graph_builder.add_edge(START, "responder")
graph_builder.add_edge("responder", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
