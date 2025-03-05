from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, MessagesState, END
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition


# Carregando variáveis de ambiente
load_dotenv()

# Iniciando llm
llm = ChatVertexAI(model_name='gemini-1.5-flash')

# Iniciando vector_store
embeddings = VertexAIEmbeddings('textembedding-gecko-multilingual@001')
vector_store = InMemoryVectorStore(embeddings)

# Carregando a página web
loader = WebBaseLoader('https://python.langchain.com/docs/tutorials/qa_chat_history/')
docs = loader.load()

# Iniciando cortes/divisões dos textos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Adicionando diviões ao vector_store
vector_store.add_documents(all_splits)

# Iniciando grafo (fluxo de trabalho)
graph_builder = StateGraph(MessagesState)

# Recuperando dados
@tool(response_format='content_and_artifact')
def retrieve(query: str):
    """Recuperando dados de acordo com a consulta"""
    retrieved_documents = vector_store.similarity_search(query, k=2)
    serialized = '\n\n'.join(
        f'Source: {doc.metadada},\n Content: {doc.page_content}'
        for doc in retrieved_documents
    )
    return retrieved_documents, serialized

# Função de consultar ou responder
def query_or_respond(state: MessagesState):
    response = llm.bind_tools([retrieve]).invoke(state['messages'])
    return {'messages': response}

# Função que gera a resposta
def generate(state: MessagesState):
    """Gerar resposta"""
    tool_messages = [m.page_content for m in reversed(state['messages']) if m.type == 'tool']
    docs_context = '\n\n'.join(m.content for m in reversed(tool_messages))

    system_message = SystemMessage(
        'Você é um assistente de perguntas. Utilize os dados recuperados (contexto) para responder as perguntas.'
        'Responda da melhor maneira possível. Utilize no máximo 3 frases.\n\n' + docs_context
    )

    conversation_messages = [
        m for m in state['messages']
        if m.type in ('human', 'system')
        or (m.type == 'ia' and not m.tool_calls)
    ]

    prompt = [system_message] + conversation_messages

    return {'messages': llm.invoke(prompt)}

# Habilitando o nó de ferramenta
tools = ToolNode([retrieve])

# Concluindo o fluxo de trabalho
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.add_conditional_edges('query_or_respond', tools_condition, {END: END})