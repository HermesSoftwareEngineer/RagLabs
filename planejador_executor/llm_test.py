from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain_google_vertexai import ChatVertexAI
from typing import Annotated

llm = ChatVertexAI(model_name="gemini-1.5-flash")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    assunto: str

replanner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """O usuário quer conversar sobre o assunto: {assunto}"""
        ),
        (
            "placeholder",
            "{messages}"
        )
    ]
)

agent = replanner_prompt | llm

def responder(state: State):
    response = agent.invoke(state)
    return {"messages": response}

graph_builder = StateGraph(State)

graph_builder.add_node("responder", responder)

graph_builder.add_edge(START, "responder")
graph_builder.add_edge("responder", END)

app = graph_builder.compile()

response = app.invoke({"messages": "olá, tudo bem?", "assunto": "política"})
print(response)