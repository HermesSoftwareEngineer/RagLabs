from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from langchain import hub
from langgraph.graph import StateGraph, START
import bs4

# Inciando o llm
llm = ChatVertexAI(model_name='gemini-1.5-flash')

# Iniciando o vector store (vetor de armazenamento)
embeddings = VertexAIEmbeddings(model='textembedding-gecko-multilingual@001')
vector_store = InMemoryVectorStore(embeddings)

# Iniciando o carregador de documentos
loader = WebBaseLoader(
    web_path=('https://lilianweng.github.io/posts/2023-06-23-agent/',),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=('post-content', 'post-title', 'post-header')
        )
    ),
)

docs = loader.load()

# Divisão do texto em partes menores (Chuking)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Definindo o prompt já padronizado
prompt = hub.pull('rlm/rag-prompt')

# Indexando os blocos de texto
_ = vector_store.add_documents(all_splits)

# Definição do estado da aplicação
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Definição do fluxo de trabalho
# Recuperação dos documentos
def retrieve (state: State):
    retrieved_documents = vector_store.similarity_search(state['question'])
    return {'context': retrieved_documents}

# Geração da resposta
def generate (state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state['context'])
    messages = prompt.invoke(
        {'question': state['question'], 'context': docs_content}
    )
    response = llm.invoke(messages)
    return {'answer': response}

# Fluxo de trabalho
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, 'retrieve')
graph = graph_builder.compile()