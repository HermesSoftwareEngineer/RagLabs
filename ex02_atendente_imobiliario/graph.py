from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from tools import buscar_imoveis, buscar_instrucoes
from vector_store import llm
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um atendente da Imobiliária Stylus. Responda com no máximo 3 frases. Se necessário, consulte o banco de dados.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Função para consultar dados ou responder
def consultar_ou_responder(state: MessagesState):
    """Função para consultar dados ou responder diretamente"""
    prompt = prompt_template.invoke(state['messages'])
    response = llm.bind_tools([buscar_imoveis, buscar_instrucoes]).invoke(prompt)
    return {'messages': response}

# Definindo as ferramentas como um nó
tools = ToolNode([buscar_imoveis, buscar_instrucoes])

# Função para gerar resposta
def gerar_resposta(state: MessagesState): 
    """Função para gerar resposta com base no contexto"""
    list_messages_tools = [m for m in reversed(state['messages']) if m.type == 'tool']
    docs_messages_toos = '\n\n'.join(m.content for m in reversed(list_messages_tools))

    system_message = SystemMessage(
        'Você é um assistente para tarefas de perguntas. Use os dados recuperados (contexto) para responder.'
        'Se você não souber a resposta, basta dizer que não sabe. Use no máximo 3 frases.\n\n' + docs_messages_toos
    )

    conversation_messages = [
        m for m in state['messages']
        if m.type in ('system', 'human')
        or (m.type == 'ia' and not m.tool_calls)
    ]
    
    prompt = [system_message] + conversation_messages

    return llm.invoke(prompt)

# Iniciando fluxo de trabalho
graph_builder = StateGraph(MessagesState)

# Adicionando os nós
graph_builder.add_node(consultar_ou_responder)
graph_builder.add_node(tools)
graph_builder.add_node(gerar_resposta)

graph_builder.set_entry_point('consultar_ou_responder')
graph_builder.add_conditional_edges('consultar_ou_responder', tools_condition, {END: END, 'tools': 'tools'})
graph_builder.add_edge('tools', 'gerar_resposta')
graph_builder.add_edge('gerar_resposta', END)

# Compilando fluxo
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)