from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain import hub
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START

# Carregando as variáveis de ambiente
load_dotenv()

# Iniciando o llm
llm = ChatVertexAI(model_name='gemini-1.5-flash')

# Iniciando a IA incorporadora que vetoriza os dados
embeddings = VertexAIEmbeddings('textembedding-gecko-multilingual@001')
vector_store = InMemoryVectorStore(embeddings)

# Carregando a página WEB
loader = WebBaseLoader('https://sitedopastor.com.br/ilustracoes/')
docs = loader.load()

# Iniciando o quebrador de textos
text_splitters = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
all_splits = text_splitters.split_documents(docs)

# Vetorizando a página Web
vector_store.add_documents(all_splits)

# Carregando prompt padrão
prompt = hub.pull('rlm/rag-prompt')

# Configurando STATE padrão
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Recuperando dados
def retrieve(state: State):
    retrieved_documents = vector_store.similarity_search(state['question'])
    return {'context': retrieved_documents}

# Gerando resposta
def generate(state: State):
    docs_context = '\n\n'.join(doc.page_content for doc in state['context'])
    messages = prompt.invoke({'question': state['question'], 'context': docs_context})
    response = llm.invoke(messages)
    return {'answer': response}

# Compilando e preparando fluxo de trabalho
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, 'retrieve')
graph = graph_builder.compile()