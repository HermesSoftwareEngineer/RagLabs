from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START
from langchain import hub
from typing_extensions import TypedDict, List
from langchain_core.documents import Document

# Iniciando o llm
llm = ChatVertexAI(model_name='gemini-1.5-flash')

# Inicializa o armazenamento vetorial
embeddings = VertexAIEmbeddings('textembedding-gecko-multilingual@001')
vector_store = InMemoryVectorStore(embeddings)

# Carrega os documentos Web
loader = WebBaseLoader('https://gshow.globo.com/cultura-pop/filmes/oscar/2025/noticia/globo-homenageia-elenco-e-equipe-de-ainda-estou-aqui-em-los-angeles.ghtml')
docs = loader.load()

# Divide o texto em partes menores
text_splliter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
all_splits = text_splliter.split_documents(docs)

# Adiciona os documentos no armazenamento vetorial
vector_store.add_documents(all_splits)

# Definição do estado da aplicação
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Definindo o prompt já padronizado
prompt = hub.pull('rlm/rag-prompt')

# Função para recuperar os documentos relevantes
def retrieve(state: State):
    state['context'] = vector_store.similarity_search(state['question'])
    return state

# Função para gerar a resposta
def generate(state: State):
    docs_content = '\n\n'.join(doc.page_content for doc in state['context'])
    messages = prompt.invoke(
        {'question': state['question'],'context': docs_content}
    )
    state['answer'] = llm.invoke(messages)
    return state

# Compilando o fluxo de trabalho
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, 'retrieve')
graph = graph_builder.compile()