from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from langchain import hub
from langgraph.graph import StateGraph, START

# Iniciando o LLM
llm = ChatVertexAI(model_name='gemini-1.5-flash')

# Iniciando o armazenamento de vetores
embbedings = VertexAIEmbeddings('textembedding-gecko-multilingual@001')
vector_store = InMemoryVectorStore(embbedings)

# Carregando a página Web
loader = WebBaseLoader('https://gshow.globo.com/globoplay/noticia/ainda-estou-aqui-registra-melhor-bilheteria-entre-indicados-a-melhor-filme-internacional-do-oscar-veja-numeros.ghtml')
docs = loader.load()

# Cortando os documentos em pedaços
text_splitters = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
all_splits = text_splitters.split_documents(docs)

# Adicionando os pedaços ao vector_store (vetorizando os pedaços)
vector_store.add_documents(all_splits)

# Carregando o prompt padrão para RAG
prompt = hub.pull('rlm/rag-prompt')

# Definindo o estado padrão STATE
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Recuperando os dados
def retrieve(state: State):
    retrieved_documents = vector_store.similarity_search(state['question'])
    return {'context': retrieved_documents}

# Gerando a resposta
def generate(state: State):
    docs_content = '\n\n'.join(doc.page_content for doc in state['context'])
    messages = prompt.invoke({'question': state['question'], 'context': docs_content})
    answer = llm.invoke(messages)
    return {'answer': answer}
    
# Compilando e preparando o fluxo de trabalho
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, 'retrieve')
graph = graph_builder.compile()