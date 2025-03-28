from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import TextLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Carregando variáveis de ambiente
load_dotenv()

# Iniciando o llm
llm = ChatVertexAI(model_name='gemini-1.5-flash')

# Iniciando os vector_stores
# embeddings = VertexAIEmbeddings('textembedding-gecko-multilingual@001')
# vector_store_instrucoes = InMemoryVectorStore(embeddings)
# vector_store_imoveis = InMemoryVectorStore(embeddings)

# # Carregando documentos
# loader_instrucoes = TextLoader(r'C:\Users\hermes.barbosa\PROJETOS_DEV\RagLabs\stylusbot_teste1\faq.txt')
# docs_instrucoes = loader_instrucoes.load()

# loader_imoveis = UnstructuredExcelLoader(r'C:\Users\hermes.barbosa\PROJETOS_DEV\RagLabs\stylusbot_teste1\dados_imoveis.xlsx')
# docs_imoveis = loader_imoveis.load()

# # Dividindo os documentos
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
# splits_instrucoes = text_splitter.split_documents(docs_instrucoes)
# splits_imoveis = text_splitter.split_documents(docs_imoveis)

# # Adicionando documentos aos vetores
# vector_store_instrucoes.add_documents(splits_instrucoes)
# vector_store_imoveis.add_documents(splits_imoveis)