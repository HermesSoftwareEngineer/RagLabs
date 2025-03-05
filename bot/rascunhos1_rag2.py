from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Carregando variáveis de ambiente
load_dotenv()

# Iniciando llm
llm = ChatVertexAI(model_name='gemini-1.5-flash')

# Iniciando vector_store
embeddings = VertexAIEmbeddings('textembedding-gecko-multilingual@001')
vector_store = InMemoryVectorStore(embeddings)

# Carregando a página web
loader = WebBaseLoader('https://g1.globo.com/economia/noticia/2025/03/03/trump-confirma-tarifas-de-25percent-para-mexico-e-canada-nesta-terca-feira.ghtml')
docs = loader.load()

# Iniciando cortes/divisões dos textos
text_splitters = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
all_splits = text_splitters.split_documents(docs)

# Adicionando diviões ao vector_store
vector_store.add_documents(all_splits)

# Iniando grafo (fluxo de trabalho)
graph_builder = StateGraph(MessagesState)

@tool(response_format='content_and_artifact')
def retrieve(query: str):
    """Recuperando informações relacionadas a consulta"""
    retireved_documents = vector_store.similarity_search(query, k=2)
    serialized = '\n\n'.join(
        (f"Source: {doc.metadata},\n Content: {doc.page_content}")
        for doc in retireved_documents
    )

    return retireved_documents, serialized

def query_or_responde(state: MessagesState):
    """Gerar chamada de ferramenta para recuperar ou responder"""
    response = llm.bind_tools([retrieve]).invoke(state['messages'])
    return {'messages': [response]}

# Recuperando dados
tools = ToolNode([retrieve])

# Função que gera a resposta
def generate(state: MessagesState):
    """Gerar resposta"""
    tool_messages = [m for m in reversed(state['messages']) if m.type == 'tool']
    docs_content = "\n\n".join(m.content for m in reversed(tool_messages))

    system_message = SystemMessage(
        'Você é um assistente para tarefas de perguntas. Use os dados recuperados (contexto) para responder.'
        'Se você não souber a resposta, basta dizer que não sabe. Use no máximo 3 frases.\n\n' + docs_content
    )

    conversation_messages = [
        message for message in state['messages']
        if message.type in ('human', 'system')
        or (message.type == 'ai' and not message.tool_calls)
    ]

    prompt = [system_message] + conversation_messages

    return {"messages": llm.invoke(prompt)}

# COncluindo o fluxo de trabalho
graph_builder.add_node(query_or_responde)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point('query_or_responde')
graph_builder.add_conditional_edges("query_or_responde", tools_condition, {END: END, 'tools': 'tools'})

graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)