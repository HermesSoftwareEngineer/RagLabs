from llm import graph
from langchain_core.messages import HumanMessage

def chat():
    while True:
        query = input("Você: ")
        if query.lower() in ['sair', 'exit', 'quit']:
            print("Encerrando o chat.")
            break
        result = graph.invoke({'question': query})
        print("Bot:", result['answer'].content)

chat()