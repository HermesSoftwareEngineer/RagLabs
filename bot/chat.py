from llm import graph
from langchain_core.messages import HumanMessage

while True:
    query = input('Usuário: ')
    if query == 'sair':
        break
    input_messages = HumanMessage(query)
    response = graph.invoke({'question': query})
    print('IA RAG: ', response['answer'].content)