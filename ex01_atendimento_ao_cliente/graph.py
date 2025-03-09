from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from vector_store import llm
from tools import buscar_documentacao
from langgraph.checkpoint.memory import MemorySaver

# Consultar ou responder
def consultar_ou_responder(state: MessagesState):
    """Consulta a documentação ou gera uma resposta direta"""
    response = llm.bind_tools([buscar_documentacao]).invoke(state['messages'])
    return {'messages': response}

# Função para gerar resposta
def generate(state: MessagesState):
    """Gera uma resposta com base no contexto e nas mensagens anteriores"""
    messages_tools = [m for m in reversed(state['messages']) if m.type == 'tool']
    context = '\n\n'.join(m.content for m in reversed(messages_tools))
    
    system_message = SystemMessage(
        'Você é um assistente de perguntas da Renner. Acesse o banco de dados (contexto) para lhe auxiliar nas respostas.'
        'Use no máximo 3 frases pare responder. Se você não souber a resposta, basta dizer que não sabe.\n' + context
    )

    conversation_messages = [
        m for m in state['messages']
        if m.type in ('human', 'ia')
        or (m.type == 'ia' and not m.tool_calls)
    ]

    prompt = [system_message] + conversation_messages

    return {'messages': llm.invoke(prompt)}

tools = ToolNode([buscar_documentacao])

# Iniciando o fluxo de trabalho
graph_builder = StateGraph(MessagesState)

# Configurando o fluxo de trabalho
graph_builder.add_node(consultar_ou_responder)
graph_builder.add_node(generate)
graph_builder.add_node(tools)

graph_builder.set_entry_point('consultar_ou_responder')
graph_builder.add_conditional_edges('consultar_ou_responder', tools_condition, {END: END, 'tools': 'tools'})

graph_builder.add_edge('tools', 'generate')
graph_builder.add_edge('generate', END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)