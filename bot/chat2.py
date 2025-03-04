from rag2 import graph
from langchain_core.messages import HumanMessage

config = {'configurable': {'thread_id': 'abc123'}}

def chat():
    while True:
        query = input("Você: ")
        if query.lower() in ['sair', 'exit', 'quit']:
            print("Encerrando o chat.")
            break
        message = [HumanMessage(query)]
        result = graph.invoke({'messages': message}, config)
        print("Bot:", result['messages'][-1].content)

chat()