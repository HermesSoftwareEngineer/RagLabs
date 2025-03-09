from vector_store import vector_store
from langchain.tools import tool

@tool(response_format='content_and_artifact')
def buscar_documentacao(query: str):
    """Função que recupera os dados para contexto"""
    retrieved_documents = vector_store.similarity_search(query, k=2)
    serialized = '\n\n'.join(
        f'Source: {doc.metadata},\nContent: {doc.page_content}'
        for doc in retrieved_documents
    )
    return retrieved_documents, serialized