from langchain.tools import tool
from vector_store import vector_store_imoveis, vector_store_instrucoes

@tool(response_format='content_and_artifact')
def buscar_imoveis(query: str):
    """Ferramenta para consultar imóveis disponíveis"""
    retrieved_documents = vector_store_imoveis.similarity_search(query, k=2)
    serialized = '\n\n'.join(
        f"Source: {doc.metadata},\nContent: {doc.page_content}"
        for doc in retrieved_documents
    )

    return retrieved_documents, serialized

@tool(response_format='content_and_artifact')
def buscar_instrucoes(query: str):
    """Ferramenta para buscar intruções sobre aluguel, documentação, caução, garantias, visitas, dúvidas frequentes, etc..."""
    retrieved_documents = vector_store_instrucoes.similarity_search(query, k=2)
    serialized= '\n\n'.join(
        f"Source: {doc.metadata},\nContent: {doc.page_content}"
        for doc in retrieved_documents
    )

    return retrieved_documents, serialized