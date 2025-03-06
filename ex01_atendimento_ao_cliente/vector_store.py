from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Iniciando variáveis de ambiente
load_dotenv()

# Iniciando LLM
llm = ChatVertexAI(model_name='geminit-1.5-flash')

# Iniciando o vector_store
embeddings = VertexAIEmbeddings('textembedding-gecko-multilingual@001')
vector_store = InMemoryVectorStore(embeddings)

# Carregando o documento
loader = WebBaseLoader('https://atendimento.lojasrenner.com.br/hc/pt-br')
docs = loader.load()

# Dividindo o documento em partes menores
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Adicionando documentos ao vector_store
vector_store.add_documents(all_splits)