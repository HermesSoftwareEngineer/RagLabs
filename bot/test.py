from langchain_community.document_loaders import TextLoader
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Carregando variáveis de ambiente
load_dotenv()

# Iniciando o vector store
embeddings = VertexAIEmbeddings('textembedding-gecko-multilingual@001')
vector_store = InMemoryVectorStore(embeddings)

# Carregando documento
loader = TextLoader(r"C:\Users\Asus\PROJETOS_DEV\RagLabs\bot\arquivo.txt", encoding="utf-8")
docs = loader.load()

# Divindo em partes menores
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
all_splits = text_splitter.split_documents(docs)

# Votorizando
vector_store.add_documents(all_splits)

for doc in all_splits:
    print('-'*30)
    print('OUTRA PARTE')
    print(doc.page_content)