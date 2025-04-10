from langgraph.prebuilt.tool_node import ToolNode
from tools import tools
from custom_types import State
from llms import llm
from pydantic import BaseModel, Field

tools_node = ToolNode(tools)

def consultar_ou_responder(state: State):
    response = llm.bind_tools(tools).invoke(state["messages"])
    # print(f"Resposta de consulta ou responder: ", response)
    return {"messages": response}

def responder(dados):
    print(f"dados: {dados}")
    return {"messages": dados.result}