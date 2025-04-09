from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain_google_vertexai import ChatVertexAI
from typing import Annotated, List, Tuple
import operator

llm = ChatVertexAI(model_name="gemini-1.5-flash")

class State(TypedDict):
    messages: Annotated[List[str], add_messages]
    input: str
    # plan: List[str]
    # past_steps: Annotated[List[Tuple], operator.add]

replanner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """O usuário quer conversar sobre o assunto: {input}
            A mensagem inicial do usuário foi: {messages}
            """
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

response = app.invoke({"messages": "olá, tudo bem?", "input": "política"})
print(response["messages"][-1].content)