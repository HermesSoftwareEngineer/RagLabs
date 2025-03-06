from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, MessagesState, END
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

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
text_splliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splliter.split_documents(docs)

# Adicionando diviões ao vector_store
vector_store.add_documents(all_splits)

# Iniciando grafo (fluxo de trabalho)
graph_builder = StateGraph(MessagesState)

# Ferramenta de recuperação de dados
@tool(response_format='content_and_artifact')
def retrieve(query: str):
    """Função que recupera os dados para contexto"""
    retrieved_documents = vector_store.similarity_search(query, k=2)
    serialized = '\n\n'.join(
        f'Source: {doc.metadata},\nContent: {doc.page_content}'
        for doc in retrieved_documents
    )
    return retrieved_documents, serialized

# Função para consultar ou respondar
def query_or_respond(state: MessagesState):
    """Consultar ou responder"""
    response = llm.bind_tools([retrieve]).invoke(state['messages'])
    return {'messages': [response]}

# Formalizando tools como um nó
tools = ToolNode([retrieve])

# Função que gera a resposta
def generate(state: MessagesState):
    """Gerando a resposta"""
    tools_messages = [message for message in reversed(state['messages']) if message.type == 'tool']
    docs_content = '\n\n'.join(m.content for m in reversed(tools_messages))

    system_message = SystemMessage(
        'Você é um assistente de perguntas. Utilize os dados recuperados (contexto) para lhe auxiliar nas respostas.'
        'Utilize no máximo 3 frases para responder. Se você não souber a resposta, basta dizer que não sabe.' + docs_content
    )

    conversation_messages = [
        message for message in state['messages']
        if message.type in ('human', 'ia')
        or (message.type == 'ia' and not message.tool_calls)
    ]

    prompt = [system_message] + conversation_messages
    return llm.invoke(prompt)

# Concluindo e compilando o fluxo de trabalho
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point('query_or_respond')
graph_builder.add_conditional_edges('query_or_respond', tools_condition, {END: END, 'tools': 'tools'})
graph_builder.add_edge('tools', 'generate')
graph_builder.add_edge('generate', END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)