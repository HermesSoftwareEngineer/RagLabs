from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain import hub
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START

# Carregando variáveis de ambiente
load_dotenv()

# Iniciando LLM
llm = ChatVertexAI(model_name='gemini-1.5-flash')

# Iniciando vector store
embeddings = VertexAIEmbeddings('textembedding-gecko-multilingual@001')
vector_store = InMemoryVectorStore(embeddings)

# Carregando documento
loader = TextLoader(r"C:\Users\Asus\PROJETOS_DEV\RagLabs\bot\arquivo.txt", encoding="utf-8")
docs = loader.load()

# Iniciando o texto spliter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Adicionando documents ao vector_store
vector_store.add_documents(all_splits)

# Iniciando prompt padrão para RAG
prompt = hub.pull('rlm/rag-prompt')

# Iniciando classe padrão
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Função para recuperar dados
def retrieve(state: State):
    retrieved_documents = vector_store.similarity_search(state['question'])
    return {'context': retrieved_documents}

# Função para gerar resposta
def generate(state: State):
    docs_context = '\n\n'.join(doc.page_content for doc in state['context'])
    messages = prompt.invoke({'question': state['question'], 'context': docs_context})
    response = llm.invoke(messages)
    return {'answer': response}

# Preparando e compilando o fluxo de trabalho
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, 'retrieve')
graph = graph_builder.compile()